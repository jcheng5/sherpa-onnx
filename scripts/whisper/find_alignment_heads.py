#!/usr/bin/env python3
"""
Analyze alignment heads for Whisper models using L2 norm scoring.

This script analyzes cross-attention patterns to:
1. Compute L2 norm scores for each attention head (per arxiv 2509.09987)
2. Analyze score distributions across models and audio samples
3. Help determine optimal L2 thresholds for dynamic head selection

The L2 norm score measures how "peaked" the attention distribution is,
which correlates with alignment quality. Higher scores indicate heads
that focus on specific audio frames rather than spreading attention.

Usage:
    # Analyze a single model/audio pair
    python find_alignment_heads.py --model tiny --audio test.wav

    # Analyze L2 threshold across multiple models
    python find_alignment_heads.py --analyze-threshold --audio-dir test_wavs/

    # Compare with fixed alignment heads
    python find_alignment_heads.py --model tiny --audio test.wav --compare-fixed
"""

import argparse
import glob
import importlib.util
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import whisper
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim

# Import load_model from the base export script.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "export_onnx", os.path.join(_script_dir, "export-onnx.py")
)
_export_onnx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_export_onnx)

load_model = _export_onnx.load_model

del _script_dir, _spec, _export_onnx


# Models to analyze for threshold determination
MODELS_FOR_THRESHOLD_ANALYSIS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    # Note: distil models require separate downloads
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Analyze alignment heads using L2 norm scoring"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., tiny, base, small, medium, distil-small.en)",
    )
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="Directory containing audio files for batch analysis",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of top heads to report"
    )
    parser.add_argument(
        "--analyze-threshold",
        action="store_true",
        help="Analyze L2 score distributions across multiple models",
    )
    parser.add_argument(
        "--compare-fixed",
        action="store_true",
        help="Compare L2 scores with model's fixed alignment heads",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models for threshold analysis",
    )
    return parser.parse_args()


class AttentionCaptureHook:
    """Hook to capture cross-attention weights from all layers/heads."""

    def __init__(self):
        self.attention_weights: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.handles = []

    def hook_fn(self, layer_idx: int):
        def fn(module, input, output):
            # output is (attn_output, attn_weights) when output_attentions=True
            # But whisper doesn't use output_attentions, so we need to compute manually
            pass
        return fn

    def clear(self):
        self.attention_weights.clear()


def compute_cross_attention_weights(
    model: whisper.Whisper,
    audio_path: str,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[int], str]:
    """
    Run transcription and capture cross-attention weights from all heads.

    Returns:
        attention_weights: Dict mapping (layer, head) to attention matrix [n_tokens, n_audio_frames]
        token_ids: List of decoded token IDs
        text: Transcribed text
    """
    # Load and preprocess audio
    audio = load_audio(audio_path)
    audio = pad_or_trim(audio)

    n_mels = model.dims.n_mels
    mel = log_mel_spectrogram(audio, n_mels=n_mels).to(model.device)

    # Encode audio
    audio_features = model.encoder(mel.unsqueeze(0))

    # Get tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual,
        num_languages=getattr(model, 'num_languages', None) or (99 if model.is_multilingual else None),
        task="transcribe",
    )

    # Initial tokens (SOT sequence + no_timestamps)
    # The no_timestamps token is required for proper decoding
    tokens = list(tokenizer.sot_sequence) + [tokenizer.no_timestamps]

    # Storage for attention weights per (layer, head)
    all_attention_weights: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

    n_layers = len(model.decoder.blocks)
    n_heads = model.dims.n_text_head

    print(f"Model has {n_layers} decoder layers with {n_heads} attention heads each")

    # Decode with attention capture
    max_tokens = 448  # max context length

    for i in range(max_tokens):
        tokens_tensor = torch.tensor([tokens]).to(model.device)

        # We need to manually run through decoder blocks to capture attention
        x = model.decoder.token_embedding(tokens_tensor) + model.decoder.positional_embedding[:tokens_tensor.shape[1]]
        x = x.to(audio_features.dtype)

        for layer_idx, block in enumerate(model.decoder.blocks):
            # Self-attention (we don't need this for alignment)
            x = x + block.attn(block.attn_ln(x), mask=model.decoder.mask)[0]

            # Cross-attention - compute manually to get weights
            cross_attn = block.cross_attn
            ln_output = block.cross_attn_ln(x)

            q = cross_attn.query(ln_output)
            k = cross_attn.key(audio_features)
            v = cross_attn.value(audio_features)

            # Reshape for multi-head attention
            batch_size, n_ctx, n_state = q.shape
            head_dim = n_state // n_heads

            q = q.view(batch_size, n_ctx, n_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(batch_size, -1, n_heads, head_dim).permute(0, 2, 1, 3)
            v = v.view(batch_size, -1, n_heads, head_dim).permute(0, 2, 1, 3)

            # Compute attention weights
            scale = head_dim ** -0.25
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            attn_weights = torch.softmax(qk.float(), dim=-1)  # [batch, heads, n_ctx, n_audio]

            # Store attention weights for each head (only the last token's attention)
            for head_idx in range(n_heads):
                # Get attention from the last decoded token
                weights = attn_weights[0, head_idx, -1, :].detach().cpu().numpy()
                all_attention_weights[(layer_idx, head_idx)].append(weights)

            # Compute attention output
            attn_output = (attn_weights.to(v.dtype) @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            attn_output = cross_attn.out(attn_output)
            x = x + attn_output

            # MLP
            x = x + block.mlp(block.mlp_ln(x))

        x = model.decoder.ln(x)
        logits = (x @ model.decoder.token_embedding.weight.T).float()

        # Get next token
        next_token = logits[0, -1].argmax().item()

        if next_token == tokenizer.eot:
            break

        tokens.append(next_token)

    # Convert to numpy arrays [n_tokens, n_audio_frames]
    attention_matrices = {}
    for key, weights_list in all_attention_weights.items():
        attention_matrices[key] = np.stack(weights_list, axis=0)

    # Decode text
    text = tokenizer.decode(tokens[len(tokenizer.sot_sequence):])

    return attention_matrices, tokens, text


def compute_l2_norm_score(attention: np.ndarray) -> float:
    """
    Compute L2 norm score for an attention matrix.

    This follows the approach from arxiv 2509.09987 "Whisper Has an Internal
    Word Aligner". The L2 norm score measures how "peaked" the attention
    distribution is - higher scores indicate heads that focus on specific
    audio frames rather than spreading attention uniformly.

    The score is computed as:
        score = sum(row_norms) + sum(col_norms)

    where row_norms and col_norms are L2 norms of each row and column.

    Args:
        attention: Attention matrix of shape (n_tokens, n_frames)

    Returns:
        L2 norm score (higher = better alignment head)
    """
    n_tokens, n_frames = attention.shape

    if n_tokens < 2 or n_frames < 2:
        return 0.0

    # Compute row norms (one per token)
    row_norms = np.linalg.norm(attention, axis=1)
    row_sum = np.sum(row_norms)

    # Compute column norms (one per frame)
    col_norms = np.linalg.norm(attention, axis=0)
    col_sum = np.sum(col_norms)

    return row_sum + col_sum


def compute_monotonicity_score(attention: np.ndarray) -> float:
    """
    Compute how monotonically increasing the attention pattern is.

    For each token, find the frame with maximum attention (argmax).
    A good alignment head should have these argmax positions increasing
    monotonically (or nearly so) as tokens progress.

    Returns a score between 0 and 1, where 1 is perfectly monotonic.
    """
    n_tokens, n_frames = attention.shape

    if n_tokens < 2:
        return 0.0

    # Get the frame with maximum attention for each token
    peak_positions = np.argmax(attention, axis=1)

    # Count how many times position increases (or stays same)
    increases = 0
    for i in range(1, len(peak_positions)):
        if peak_positions[i] >= peak_positions[i - 1]:
            increases += 1

    monotonicity = increases / (len(peak_positions) - 1)
    return monotonicity


def compute_diagonal_score(attention: np.ndarray) -> float:
    """
    Compute how diagonal the attention pattern is.

    A diagonal pattern means token i attends mostly to audio frame i*scale,
    where scale = n_frames / n_tokens.
    """
    n_tokens, n_frames = attention.shape

    if n_tokens < 2:
        return 0.0

    # Expected diagonal positions
    scale = n_frames / n_tokens
    expected_positions = np.arange(n_tokens) * scale

    # Actual peak positions
    peak_positions = np.argmax(attention, axis=1)

    # Compute correlation between expected and actual
    if np.std(peak_positions) < 1e-6:
        return 0.0

    correlation = np.corrcoef(expected_positions, peak_positions)[0, 1]

    # Handle NaN
    if np.isnan(correlation):
        return 0.0

    return max(0, correlation)  # Only positive correlations indicate good alignment


def analyze_attention_heads(
    attention_matrices: Dict[Tuple[int, int], np.ndarray],
    top_k: int = 10,
    sort_by: str = "l2",
) -> List[Tuple[Tuple[int, int], float, float, float, float]]:
    """
    Analyze all attention heads and rank them by alignment quality.

    Args:
        attention_matrices: Dict mapping (layer, head) to attention matrix
        top_k: Number of top heads to return
        sort_by: Metric to sort by ("l2", "monotonic", "diagonal", "combined")

    Returns:
        List of ((layer, head), l2_score, mono_score, diag_score, combined_score)
        sorted by the specified metric.
    """
    results = []

    for (layer, head), attention in attention_matrices.items():
        l2_score = compute_l2_norm_score(attention)
        mono_score = compute_monotonicity_score(attention)
        diag_score = compute_diagonal_score(attention)
        combined_score = (mono_score + diag_score) / 2
        results.append(((layer, head), l2_score, mono_score, diag_score, combined_score))

    # Sort by specified metric (descending)
    sort_indices = {"l2": 1, "monotonic": 2, "diagonal": 3, "combined": 4}
    sort_idx = sort_indices.get(sort_by, 1)
    results.sort(key=lambda x: x[sort_idx], reverse=True)

    return results[:top_k]


def get_fixed_alignment_heads(model) -> Optional[List[Tuple[int, int]]]:
    """Extract fixed alignment heads from model metadata if available."""
    if not hasattr(model, "alignment_heads") or model.alignment_heads is None:
        return None

    try:
        ah = model.alignment_heads
        if hasattr(ah, "indices"):
            indices = ah.indices()
            return list(zip(indices[0].tolist(), indices[1].tolist()))
    except Exception:
        pass

    return None


def analyze_threshold_distribution(
    model_name: str,
    audio_files: List[str],
    top_k: int = 20,
) -> Dict:
    """
    Analyze L2 score distribution for a model across multiple audio files.

    Returns statistics about L2 scores for determining optimal threshold.
    """
    print(f"\nAnalyzing {model_name}...")

    try:
        model = load_model(model_name)
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return {}

    n_layers = len(model.decoder.blocks)
    n_heads = model.dims.n_text_head
    total_heads = n_layers * n_heads
    print(f"  Model has {n_layers} layers x {n_heads} heads = {total_heads} total heads")

    # Get fixed alignment heads for comparison
    fixed_heads = get_fixed_alignment_heads(model)
    if fixed_heads:
        print(f"  Fixed alignment heads: {fixed_heads}")

    all_l2_scores = []
    fixed_head_scores = []
    non_fixed_head_scores = []

    for audio_path in audio_files:
        if not os.path.exists(audio_path):
            continue

        try:
            attention_matrices, tokens, text = compute_cross_attention_weights(
                model, audio_path
            )
            print(f"  Audio: {os.path.basename(audio_path)} -> {len(tokens)} tokens")

            for (layer, head), attention in attention_matrices.items():
                l2_score = compute_l2_norm_score(attention)
                all_l2_scores.append(l2_score)

                if fixed_heads and (layer, head) in fixed_heads:
                    fixed_head_scores.append(l2_score)
                else:
                    non_fixed_head_scores.append(l2_score)

        except Exception as e:
            print(f"  Failed on {audio_path}: {e}")
            continue

    if not all_l2_scores:
        return {}

    result = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "total_heads": total_heads,
        "n_samples": len(all_l2_scores) // total_heads,
        "all_scores": {
            "min": min(all_l2_scores),
            "max": max(all_l2_scores),
            "mean": statistics.mean(all_l2_scores),
            "median": statistics.median(all_l2_scores),
            "stdev": statistics.stdev(all_l2_scores) if len(all_l2_scores) > 1 else 0,
        },
    }

    if fixed_head_scores:
        result["fixed_head_scores"] = {
            "count": len(fixed_heads),
            "min": min(fixed_head_scores),
            "max": max(fixed_head_scores),
            "mean": statistics.mean(fixed_head_scores),
        }

    if non_fixed_head_scores:
        result["non_fixed_scores"] = {
            "min": min(non_fixed_head_scores),
            "max": max(non_fixed_head_scores),
            "mean": statistics.mean(non_fixed_head_scores),
        }

    # Print summary
    print(f"\n  L2 Score Distribution:")
    print(f"    Min: {result['all_scores']['min']:.2f}")
    print(f"    Max: {result['all_scores']['max']:.2f}")
    print(f"    Mean: {result['all_scores']['mean']:.2f}")
    print(f"    Median: {result['all_scores']['median']:.2f}")
    print(f"    Stdev: {result['all_scores']['stdev']:.2f}")

    if "fixed_head_scores" in result:
        print(f"\n  Fixed Alignment Heads ({result['fixed_head_scores']['count']}):")
        print(f"    Mean L2: {result['fixed_head_scores']['mean']:.2f}")
        print(f"    Range: [{result['fixed_head_scores']['min']:.2f}, {result['fixed_head_scores']['max']:.2f}]")

    return result


def run_threshold_analysis(args):
    """Run L2 threshold analysis across multiple models."""
    # Get list of models to analyze
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = MODELS_FOR_THRESHOLD_ANALYSIS

    # Get audio files
    audio_files = []
    if args.audio_dir:
        audio_files = glob.glob(os.path.join(args.audio_dir, "*.wav"))
        audio_files += glob.glob(os.path.join(args.audio_dir, "*.mp3"))
        audio_files += glob.glob(os.path.join(args.audio_dir, "*.m4a"))
    elif args.audio:
        audio_files = [args.audio]

    if not audio_files:
        print("Error: No audio files found. Use --audio or --audio-dir")
        return

    print(f"Analyzing {len(models)} models with {len(audio_files)} audio files")
    print("=" * 70)

    all_results = []
    for model_name in models:
        result = analyze_threshold_distribution(model_name, audio_files)
        if result:
            all_results.append(result)

    # Print summary across all models
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY: L2 Score Statistics Across Models")
        print("=" * 70)

        all_fixed_means = []
        all_non_fixed_means = []

        for r in all_results:
            if "fixed_head_scores" in r:
                all_fixed_means.append(r["fixed_head_scores"]["mean"])
            if "non_fixed_scores" in r:
                all_non_fixed_means.append(r["non_fixed_scores"]["mean"])

        if all_fixed_means and all_non_fixed_means:
            print(f"\nFixed Alignment Heads (across models):")
            print(f"  Mean L2: {statistics.mean(all_fixed_means):.2f}")
            print(f"  Range: [{min(all_fixed_means):.2f}, {max(all_fixed_means):.2f}]")

            print(f"\nNon-Fixed Heads (across models):")
            print(f"  Mean L2: {statistics.mean(all_non_fixed_means):.2f}")
            print(f"  Range: [{min(all_non_fixed_means):.2f}, {max(all_non_fixed_means):.2f}]")

            # Suggest threshold
            suggested_threshold = (
                min(all_fixed_means) + max(all_non_fixed_means)
            ) / 2
            print(f"\nSuggested L2 Threshold: {suggested_threshold:.2f}")
            print("  (midpoint between min fixed and max non-fixed)")


def run_single_analysis(args):
    """Analyze a single model with one audio file."""
    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Model dimensions: {model.dims}")

    # Check for fixed alignment heads
    fixed_heads = get_fixed_alignment_heads(model)
    if fixed_heads:
        print(f"Model has pre-defined alignment heads: {fixed_heads}")

    # Run transcription and capture attention
    print(f"\nTranscribing: {args.audio}")
    attention_matrices, tokens, text = compute_cross_attention_weights(model, args.audio)

    print(f"\nTranscription: {text}")
    print(f"Number of tokens: {len(tokens)}")

    # Analyze heads
    print(f"\nAnalyzing {len(attention_matrices)} attention heads...")
    top_heads = analyze_attention_heads(attention_matrices, args.top_k, sort_by="l2")

    print(f"\nTop {args.top_k} heads by L2 score:")
    print("-" * 80)
    print(f"{'Layer':>6} {'Head':>6} {'L2 Score':>12} {'Monotonic':>12} {'Diagonal':>12} {'Fixed':>8}")
    print("-" * 80)

    fixed_set = set(fixed_heads) if fixed_heads else set()

    for (layer, head), l2, mono, diag, combined in top_heads:
        is_fixed = "YES" if (layer, head) in fixed_set else ""
        print(f"{layer:>6} {head:>6} {l2:>12.2f} {mono:>12.3f} {diag:>12.3f} {is_fixed:>8}")

    # Compare with fixed heads if requested
    if args.compare_fixed and fixed_heads:
        print("\n" + "=" * 80)
        print("Comparison: Dynamic L2 Selection vs Fixed Heads")
        print("=" * 80)

        # Get L2 scores for all heads
        all_heads = analyze_attention_heads(
            attention_matrices, len(attention_matrices), sort_by="l2"
        )
        l2_by_head = {(l, h): score for (l, h), score, _, _, _ in all_heads}

        # Ranks of fixed heads
        print("\nFixed head ranks in L2 ordering:")
        for i, ((layer, head), l2, _, _, _) in enumerate(all_heads):
            if (layer, head) in fixed_set:
                print(f"  ({layer}, {head}): rank {i + 1}, L2 = {l2:.2f}")

        # Top N by L2 that aren't fixed
        n_fixed = len(fixed_heads)
        top_l2_heads = [(l, h) for (l, h), _, _, _, _ in top_heads[:n_fixed]]
        overlap = set(top_l2_heads) & fixed_set

        print(f"\nOverlap between top-{n_fixed} L2 and {n_fixed} fixed heads: {len(overlap)}")
        print(f"  Top-{n_fixed} by L2: {top_l2_heads}")
        print(f"  Fixed heads: {fixed_heads}")

    # Suggest heads based on L2 threshold
    print("\n" + "=" * 80)
    print("Suggested Alignment Heads (L2 > median):")
    print("=" * 80)

    all_l2 = [l2 for _, l2, _, _, _ in analyze_attention_heads(
        attention_matrices, len(attention_matrices)
    )]
    median_l2 = statistics.median(all_l2)

    good_heads = [(l, h) for (l, h), l2, _, _, _ in top_heads if l2 > median_l2]
    if len(good_heads) < 4:
        good_heads = [(l, h) for (l, h), _, _, _, _ in top_heads[:6]]

    model_name = args.model.replace("-", "_").replace(".", "_")
    print(f'"{args.model}": {good_heads},')
    print(f"\n(Median L2 threshold: {median_l2:.2f})")


def main():
    args = get_args()

    if args.analyze_threshold:
        run_threshold_analysis(args)
    elif args.model and args.audio:
        run_single_analysis(args)
    else:
        print("Usage:")
        print("  Analyze single model: --model <name> --audio <file>")
        print("  Analyze threshold:    --analyze-threshold --audio-dir <dir>")
        print("  Compare with fixed:   --model <name> --audio <file> --compare-fixed")


if __name__ == "__main__":
    main()
