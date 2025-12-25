#include "MarkovChain.hpp"

namespace ptm {

void MarkovChain::Train(const std::vector<State>& sequence) {

  for (size_t i = 0; i < sequence.size(); ++i) {
    if (!state_to_index_.contains(sequence[i])) {
      state_to_index_[sequence[i]] = index_to_state_.size();
      index_to_state_.push_back(sequence[i]);

      for (auto vec: counts_)
        vec.push_back(0);
      counts_.push_back(std::vector<size_t>(index_to_state_.size(), 0));
      row_sums_.push_back(0);
    }

    if (i > 0) {
      size_t from = state_to_index_[sequence[i - 1]];
      size_t to = state_to_index_[sequence[i]];

      counts_[from][to]++;
      row_sums_[from]++;
    }
  }
}

[[nodiscard]] std::unordered_map<MarkovChain::State, double> MarkovChain::NextDistribution(const State& current) const {
  std::unordered_map<State, double> ans;  
  size_t from = state_to_index_.at(current);

  if (row_sums_[from] == 0)
    return ans;
  
  for (size_t to = 0; to < index_to_state_.size(); ++to) {
    ans[index_to_state_[to]] = static_cast<double>(counts_[from][to]) / static_cast<double>(row_sums_[from]);
  }
  return ans;
}

double MarkovChain::TransitionProbability(const State& from, const State& to) const {
  size_t fromI = state_to_index_.at(from);
  size_t toI = state_to_index_.at(to);
  return static_cast<double>(counts_[fromI][toI]) / static_cast<double>(row_sums_[fromI]);
}

std::optional<MarkovChain::State> MarkovChain::SampleNext(const State& current, std::mt19937& rng) const {
  double r = static_cast<double>(rng() - rng.min()) / static_cast<double>(rng.max() - rng.min());
  auto distribution = NextDistribution(current);

  if (distribution.empty())
    return {};
  
  for (auto x: distribution) {
    r -= x.second;
    if (r <= 0)
      return {x.first};
  }
}


std::vector<MarkovChain::State> MarkovChain::Generate(const State& start, size_t length, std::mt19937& rng) const {
  std::vector<MarkovChain::State> ans;
  ans.reserve(length);

  State current = start;
  for (size_t i = 0; i < length; ++i) {
    ans.push_back(current);

    std::optional<State> next = SampleNext(current, rng);
    if (!next.has_value())
      return ans;
    current = *next;
  }
  return ans;
}

std::vector<MarkovChain::State> MarkovChain::States() const {
  return index_to_state_;
}

bool MarkovChain::HasState(const State& state) const {
  return state_to_index_.contains(state);
}

}
