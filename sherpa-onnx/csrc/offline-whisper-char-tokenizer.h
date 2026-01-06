// sherpa-onnx/csrc/offline-whisper-char-tokenizer.h
//
// Copyright (c)  2025  Posit Software, PBC

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_CHAR_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_CHAR_TOKENIZER_H_

#include <functional>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// Result of character tokenization
struct CharTokenizationResult {
  std::vector<int32_t> tokens;  // Character token IDs (including space tokens)
  std::vector<std::string> words;  // Original words (after normalization)
  // word_char_boundaries[i] = index of first char token for word[i]
  // The characters for word[i] are tokens[word_char_boundaries[i]] to
  // tokens[word_char_boundaries[i+1]-1] (or tokens.size()-1 for last word)
  std::vector<int32_t> word_char_boundaries;
};

// Mapping from subword tokens to character token ranges
struct SubwordToCharMapping {
  // For each subword token, the range of character tokens it maps to
  // char_start[i] = index of first char token for subword[i]
  // char_end[i] = index past last char token for subword[i] (exclusive)
  // If char_start[i] == char_end[i], the subword has no character mapping
  // (e.g., punctuation-only tokens)
  std::vector<int32_t> char_start;
  std::vector<int32_t> char_end;
};

// Remove punctuation from text for alignment purposes.
// Keeps apostrophes (') since they're common in contractions.
// Returns lowercase text with punctuation removed.
std::string NormalizeTextForAlignment(const std::string& text);

// Tokenize text as individual characters.
// Each character is encoded separately using the symbol table.
// Space tokens are inserted between words.
//
// Args:
//   text: The text to tokenize (will be normalized internally)
//   symbol_table: The Whisper token vocabulary
//   space_token_id: Token ID for the space character (" ")
//
// Returns:
//   CharTokenizationResult containing character tokens and word boundaries
CharTokenizationResult CharacterTokenize(
    const std::string& text,
    const SymbolTable& symbol_table,
    int32_t space_token_id);

// Build a mapping from original subword tokens to character token ranges.
// This allows converting character-level timestamps back to subword timestamps.
//
// Args:
//   subword_tokens: Original BPE subword token IDs from normal decoding
//   char_result: Result from CharacterTokenize on the same text
//   symbol_table: The Whisper token vocabulary
//
// Returns:
//   SubwordToCharMapping where char_start[i]/char_end[i] give the character
//   token range for subword_tokens[i]. For punctuation-only tokens (not in
//   normalized text), char_start[i] == char_end[i].
SubwordToCharMapping BuildSubwordToCharMapping(
    const std::vector<int32_t>& subword_tokens,
    const CharTokenizationResult& char_result,
    const SymbolTable& symbol_table);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_CHAR_TOKENIZER_H_
