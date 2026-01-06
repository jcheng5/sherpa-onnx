// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-whisper-decoder.h"
#include "sherpa-onnx/csrc/offline-whisper-model.h"

namespace sherpa_onnx {

class OfflineWhisperGreedySearchDecoder : public OfflineWhisperDecoder {
 public:
  OfflineWhisperGreedySearchDecoder(const OfflineWhisperModelConfig &config,
                                    OfflineWhisperModel *model)
      : config_(config), model_(model) {}

  std::vector<OfflineWhisperDecoderResult> Decode(
      Ort::Value cross_k, Ort::Value cross_v,
      int32_t num_feature_frames) override;

  void SetConfig(const OfflineWhisperModelConfig &config) override;

  /** Run teacher-forced forward pass with given tokens.
   *
   * This runs a single forward pass through the decoder with ALL tokens
   * at once (no autoregressive generation). Used for character-level
   * alignment where we know the text in advance.
   *
   * @param tokens The full token sequence including SOT, no_timestamps,
   *               text tokens (characters), and EOT
   * @param cross_k Encoder output cross-attention keys (will be copied)
   * @param cross_v Encoder output cross-attention values (will be copied)
   * @param num_feature_frames Number of feature frames from the encoder
   *
   * @return TeacherForcedResult containing attention weights for all tokens
   */
  TeacherForcedResult RunTeacherForced(
      const std::vector<int64_t>& tokens,
      Ort::Value cross_k,
      Ort::Value cross_v,
      int32_t num_feature_frames);

 private:
  OfflineWhisperModelConfig config_;
  OfflineWhisperModel *model_;  // not owned
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_GREEDY_SEARCH_DECODER_H_
