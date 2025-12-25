#include <sstream>

#include "MarkovTextModel.hpp"

namespace ptm {

MarkovTextModel::MarkovTextModel(TokenLevel level) {
  chain_ = MarkovChain();
  level_ = level;
}

void MarkovTextModel::TrainFromText(const std::string& text) {
  chain_.Train(Tokenize(text));
}

std::string MarkovTextModel::GenerateText(std::size_t num_tokens, std::mt19937& rng, const std::string& start_token) const {
  std::string start = start_token;
  if (!chain_.HasState(start))
    start = chain_.States()[0];

  return Detokenize(chain_.Generate(start, num_tokens, rng));
}

const MarkovChain& MarkovTextModel::Chain() const noexcept {
  return chain_;
}

std::vector<std::string> MarkovTextModel::Tokenize(const std::string& text) const {
  if (level_ == TokenLevel::Word) {
    std::stringstream textStream(text);
    std::string s;

    std::vector<std::string> ans;
    while (getline(textStream, s, ' '))
      ans.push_back(s);
    return ans;
  }

  if(level_ == TokenLevel::Character) {
    std::vector<std::string> ans(text.size());
    for (int i = 0; i < text.size(); ++i)
      ans[i] = std::string(1, text[i]);
    return ans;
  }
}


std::string MarkovTextModel::Detokenize(const std::vector<std::string>& tokens) const {
  if (level_ == TokenLevel::Word) {
    std::string ans;
    for (int i = 0; i < tokens.size(); ++i) {
      ans += tokens[i];
      if (i != tokens.size() - 1)
        ans += " ";
    }
    return ans;
  }

  if (level_ == TokenLevel::Character) {
    std::string ans;
    for (int i = 0; i < tokens.size(); ++i) {
      ans += tokens[i];
    }
    return ans;
  }
}

}
