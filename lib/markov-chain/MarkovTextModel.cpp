#include "MarkovTextModel.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace ptm {

MarkovTextModel::MarkovTextModel(TokenLevel level) : level_(level) {
}

void MarkovTextModel::TrainFromText(const std::string& text) {
  auto tokens = Tokenize(text);
  chain_.Train(tokens);
}

std::string MarkovTextModel::GenerateText(std::size_t num_tokens,
                                          std::mt19937& rng,
                                          const std::string& start_token) const {
  if (num_tokens == 0) {
    return "";
  }
  std::string start = start_token;
  auto states = chain_.States();
  if (states.empty()) {
    return "";
  }
  if (start.empty() || std::ranges::find(states, start) == states.end()) {
    start = states[0];
  }
  auto generated = chain_.Generate(start, num_tokens, rng);
  return Detokenize(generated);
}

const MarkovChain& MarkovTextModel::Chain() const noexcept {
  return chain_;
}

std::vector<std::string> MarkovTextModel::Tokenize(const std::string& text) const {
  std::vector<std::string> tokens;
  if (level_ == TokenLevel::Character) {
    for (char c : text) {
      tokens.emplace_back(1, c);
    }
  } else if (level_ == TokenLevel::Word) {
    std::string current_word;
    for (char c : text) {
      if (std::isalnum(c) || c == '\'') {
        current_word += c;
      } else {
        if (!current_word.empty()) {
          tokens.push_back(current_word);
          current_word.clear();
        }
        if (!std::isspace(c)) {
          tokens.emplace_back(1, c);
        }
      }
    }
    if (!current_word.empty()) {
      tokens.push_back(current_word);
    }
  } else {
    // std::unreachable(); TODO: add c++23
  }
  return tokens;
}

std::string MarkovTextModel::Detokenize(const std::vector<std::string>& tokens) const {
  if (tokens.empty()) {
    return "";
  }
  if (level_ == TokenLevel::Character) {
    std::string result;
    for (const auto& t : tokens) {
      result += t;
    }
    return result;
  } else if (level_ == TokenLevel::Word) {
    std::string result = tokens[0];
    for (size_t i = 1; i < tokens.size(); ++i) {
      bool is_punctuation = (tokens[i].size() == 1 && !std::isalnum(tokens[i][0]) && tokens[i][0] != '\'');
      if (is_punctuation) {
        result += tokens[i];
      } else {
        result += " " + tokens[i];
      }
    }
    return result;
  } else {
    // std::unreachable(); TODO: add c++23
  }
}

} // namespace ptm
