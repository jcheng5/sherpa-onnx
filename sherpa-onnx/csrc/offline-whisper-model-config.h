// sherpa-onnx/csrc/offline-whisper-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineWhisperModelConfig {
  std::string encoder;
  std::string decoder;

  // Available languages can be found at
  // https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
  //
  // Note: For non-multilingual models, it supports only "en"
  //
  // If empty, we will infer it from the input audio file when
  // the model is multilingual.
  std::string language;

  // Valid values are transcribe and translate
  //
  // Note: For non-multilingual models, it supports only "transcribe"
  std::string task = "transcribe";

  // Number of tail padding frames.
  //
  // Since we remove the 30-second constraint, we need to add some paddings
  // at the end.
  //
  // Recommended values:
  //   - 50 for English models
  //   - 300 for multilingual models
  int32_t tail_paddings = -1;

  // If true, use cross-attention weights and DTW to compute token-level
  // timestamps. This requires ONNX models exported with attention outputs.
  bool enable_timestamps = false;

  // If true, use Whisper's native timestamp token mode to produce segment-level
  // timestamps. The decoder outputs timestamp tokens like <|0.00|> interleaved
  // with text, creating segments with start/end times. Does not require
  // attention outputs. Can be combined with enable_timestamps for both
  // segment-level and token-level timestamps.
  bool enable_segment_timestamps = false;

  // If true, use character-level tokenization for more accurate word timestamps.
  // This implements the approach from "Whisper Has an Internal Word Aligner"
  // (arxiv 2509.09987):
  // 1. Decode audio normally to get transcription
  // 2. Re-tokenize text as individual characters
  // 3. Run teacher-forced forward pass with character tokens
  // 4. Apply DTW on character-level attention for timestamps
  // 5. Map character timestamps back to original subword tokens
  //
  // Requires enable_timestamps=true and models with attention outputs.
  // Results in more accurate token timestamps but requires two decoder passes.
  bool enable_character_alignment = false;

  OfflineWhisperModelConfig() = default;
  OfflineWhisperModelConfig(const std::string &encoder,
                            const std::string &decoder,
                            const std::string &language,
                            const std::string &task, int32_t tail_paddings,
                            bool enable_timestamps = false,
                            bool enable_segment_timestamps = false,
                            bool enable_character_alignment = false)
      : encoder(encoder),
        decoder(decoder),
        language(language),
        task(task),
        tail_paddings(tail_paddings),
        enable_timestamps(enable_timestamps),
        enable_segment_timestamps(enable_segment_timestamps),
        enable_character_alignment(enable_character_alignment) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_CONFIG_H_
