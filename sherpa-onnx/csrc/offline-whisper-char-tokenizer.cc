// sherpa-onnx/csrc/offline-whisper-char-tokenizer.cc
//
// Copyright (c)  2025  Posit Software, PBC

#include "sherpa-onnx/csrc/offline-whisper-char-tokenizer.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Split a UTF-8 string into individual characters (without merging)
// Unlike SplitUtf8(), this does NOT merge ASCII characters into words.
std::vector<std::string> SplitUtf8ToChars(const std::string &text) {
  const uint8_t *begin = reinterpret_cast<const uint8_t *>(text.c_str());
  const uint8_t *end = begin + text.size();

  std::vector<std::string> ans;

  auto start = begin;
  while (start < end) {
    uint8_t c = *start;
    uint8_t i = 0x80;
    int32_t num_bytes = 0;

    // See https://en.wikipedia.org/wiki/UTF-8
    for (; c & i; i >>= 1) {
      ++num_bytes;
    }

    if (num_bytes == 0) {
      // This is an ASCII character
      ans.emplace_back(reinterpret_cast<const char *>(start), 1);
      ++start;
    } else if (2 <= num_bytes && num_bytes <= 4) {
      ans.emplace_back(reinterpret_cast<const char *>(start), num_bytes);
      start += num_bytes;
    } else {
      // Invalid UTF-8 byte, skip it
      ++start;
    }
  }

  return ans;
}

// Check if a character is punctuation (except apostrophe)
bool IsPunctuation(char32_t c) {
  // Keep apostrophe for contractions (e.g., "don't", "it's")
  if (c == '\'') {
    return false;
  }
  // ASCII punctuation
  if ((c >= 0x21 && c <= 0x2F) ||  // !"#$%&'()*+,-./
      (c >= 0x3A && c <= 0x40) ||  // :;<=>?@
      (c >= 0x5B && c <= 0x60) ||  // [\]^_`
      (c >= 0x7B && c <= 0x7E)) {  // {|}~
    return true;
  }
  // Common Unicode punctuation ranges
  // General punctuation
  if (c >= 0x2000 && c <= 0x206F) {
    return true;
  }
  // Supplemental punctuation
  if (c >= 0x2E00 && c <= 0x2E7F) {
    return true;
  }
  return false;
}

// Decode a single token ID to its text representation
std::string DecodeToken(int32_t token_id, const SymbolTable& symbol_table) {
  if (!symbol_table.Contains(token_id)) {
    return "";
  }
  return symbol_table[token_id];
}

}  // namespace

std::string NormalizeTextForAlignment(const std::string& text) {
  // Convert to UTF-32 for easier character manipulation
  std::u32string u32text = Utf8ToUtf32(text);

  std::u32string result;
  result.reserve(u32text.size());

  for (char32_t c : u32text) {
    // Skip punctuation (except apostrophe)
    if (IsPunctuation(c)) {
      continue;
    }
    // Convert to lowercase
    if (c >= 'A' && c <= 'Z') {
      c = c - 'A' + 'a';
    }
    result.push_back(c);
  }

  // Convert back to UTF-8
  std::string utf8_result = Utf32ToUtf8(result);

  // Normalize multiple spaces to single space
  std::string normalized;
  normalized.reserve(utf8_result.size());
  bool prev_space = true;  // Start true to skip leading spaces
  for (char c : utf8_result) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      if (!prev_space) {
        normalized.push_back(' ');
        prev_space = true;
      }
    } else {
      normalized.push_back(c);
      prev_space = false;
    }
  }

  // Remove trailing space
  if (!normalized.empty() && normalized.back() == ' ') {
    normalized.pop_back();
  }

  return normalized;
}

CharTokenizationResult CharacterTokenize(
    const std::string& text,
    const SymbolTable& symbol_table,
    int32_t space_token_id) {
  CharTokenizationResult result;

  // Normalize text first
  std::string normalized = NormalizeTextForAlignment(text);

  if (normalized.empty()) {
    return result;
  }

  // Split into words
  std::vector<std::string> words;
  SplitStringToVector(normalized, " ", true, &words);

  if (words.empty()) {
    return result;
  }

  result.words = words;
  result.word_char_boundaries.reserve(words.size());

  // Tokenize each word character by character
  for (size_t word_idx = 0; word_idx < words.size(); ++word_idx) {
    const std::string& word = words[word_idx];

    // Record start of this word's characters
    result.word_char_boundaries.push_back(
        static_cast<int32_t>(result.tokens.size()));

    // Split word into individual UTF-8 characters
    std::vector<std::string> chars = SplitUtf8ToChars(word);

    for (const std::string& ch : chars) {
      // Try to find this character in the symbol table
      if (symbol_table.Contains(ch)) {
        result.tokens.push_back(symbol_table[ch]);
      } else {
        // Character not found - this can happen with special characters
        // Try lowercase version
        std::string lower_ch = ToLowerCase(ch);
        if (symbol_table.Contains(lower_ch)) {
          result.tokens.push_back(symbol_table[lower_ch]);
        } else {
          // Skip unknown characters
          SHERPA_ONNX_LOGE(
              "CharacterTokenize: Unknown character '%s' skipped",
              ch.c_str());
        }
      }
    }

    // Add space token between words (not after the last word)
    if (word_idx < words.size() - 1) {
      result.tokens.push_back(space_token_id);
    }
  }

  return result;
}

SubwordToCharMapping BuildSubwordToCharMapping(
    const std::vector<int32_t>& subword_tokens,
    const CharTokenizationResult& char_result,
    const SymbolTable& symbol_table) {
  SubwordToCharMapping mapping;
  mapping.char_start.resize(subword_tokens.size(), 0);
  mapping.char_end.resize(subword_tokens.size(), 0);

  if (subword_tokens.empty() || char_result.tokens.empty()) {
    return mapping;
  }

  // Build the full text from subword tokens (after normalization)
  std::string subword_text;
  for (int32_t token_id : subword_tokens) {
    std::string token_text = DecodeToken(token_id, symbol_table);
    subword_text += token_text;
  }

  // Normalize the subword text the same way as character tokenization
  std::string normalized_subword_text = NormalizeTextForAlignment(subword_text);

  // Build the full text from character tokens
  std::string char_text;
  for (int32_t token_id : char_result.tokens) {
    std::string token_text = DecodeToken(token_id, symbol_table);
    char_text += token_text;
  }

  // The two texts should match (after normalization)
  // Now we need to map each subword to its character range

  // Track position in character text
  int32_t char_idx = 0;
  int32_t char_text_pos = 0;

  for (size_t subword_idx = 0; subword_idx < subword_tokens.size();
       ++subword_idx) {
    std::string token_text = DecodeToken(subword_tokens[subword_idx],
                                         symbol_table);
    std::string normalized_token = NormalizeTextForAlignment(token_text);

    if (normalized_token.empty()) {
      // This subword has no characters after normalization (e.g., punctuation)
      // Set empty range at current position
      mapping.char_start[subword_idx] = char_idx;
      mapping.char_end[subword_idx] = char_idx;
      continue;
    }

    // Find where this normalized token appears in the character text
    int32_t start_char_idx = char_idx;

    // Count how many character tokens make up this normalized token
    size_t normalized_pos = 0;
    while (normalized_pos < normalized_token.size() &&
           char_idx < static_cast<int32_t>(char_result.tokens.size())) {
      std::string char_token_text = DecodeToken(char_result.tokens[char_idx],
                                                symbol_table);

      // Check if this character token matches the next part of normalized_token
      if (normalized_token.substr(normalized_pos, char_token_text.size()) ==
          char_token_text) {
        normalized_pos += char_token_text.size();
        char_idx++;
      } else if (char_token_text == " ") {
        // Space token - skip if we're at a word boundary
        if (normalized_pos == 0 ||
            normalized_pos == normalized_token.size()) {
          char_idx++;
        } else {
          // Space in the middle - might be part of the token
          if (normalized_token[normalized_pos] == ' ') {
            normalized_pos++;
            char_idx++;
          } else {
            break;
          }
        }
      } else {
        // Mismatch - this shouldn't happen if normalization is consistent
        SHERPA_ONNX_LOGE(
            "BuildSubwordToCharMapping: Mismatch at subword %zu, "
            "expected '%s' at pos %zu, got char token '%s'",
            subword_idx, normalized_token.c_str(), normalized_pos,
            char_token_text.c_str());
        break;
      }
    }

    mapping.char_start[subword_idx] = start_char_idx;
    mapping.char_end[subword_idx] = char_idx;
  }

  return mapping;
}

}  // namespace sherpa_onnx
