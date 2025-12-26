#include "MarkovChain.hpp"

namespace ptm {

size_t MarkovChain::ensureState(const State& s) {
  auto it = state_to_index_.find(s);
  if (it != state_to_index_.end()) {
    return it->second;
  }
  size_t index = index_to_state_.size();
  state_to_index_[s] = index;
  index_to_state_.push_back(s);
  counts_.emplace_back(index_to_state_.size(), 0);
  row_sums_.push_back(0);

  for (auto& row : counts_) {
    row.resize(index_to_state_.size(), 0);
  }
  return index;
}

void MarkovChain::Train(const std::vector<State>& sequence) {
  for (const auto& s : sequence) {
    ensureState(s);
  }
  for (size_t i = 0; i + 1 < sequence.size(); ++i) {
    size_t from_idx = ensureState(sequence[i]);
    size_t to_idx = ensureState(sequence[i + 1]);
    counts_[from_idx][to_idx]++;
    row_sums_[from_idx]++;
  }
}

std::unordered_map<MarkovChain::State, double> MarkovChain::NextDistribution(const State& current) const {
  std::unordered_map<State, double> dist;
  auto it = state_to_index_.find(current);
  if (it == state_to_index_.end() || row_sums_[it->second] == 0) {
    return dist;
  }
  size_t idx = it->second;
  for (size_t j = 0; j < index_to_state_.size(); ++j) {
    if (counts_[idx][j] > 0) {
      dist[index_to_state_[j]] = static_cast<double>(counts_[idx][j]) / static_cast<double>(row_sums_[idx]);
    }
  }
  return dist;
}

double MarkovChain::TransitionProbability(const State& from, const State& to) const {
  auto from_it = state_to_index_.find(from);
  auto to_it = state_to_index_.find(to);
  if (from_it == state_to_index_.end() || to_it == state_to_index_.end()) {
    return 0.0;
  }
  size_t from_idx = from_it->second;
  size_t to_idx = to_it->second;
  if (row_sums_[from_idx] == 0) {
    return 0.0;
  }
  return static_cast<double>(counts_[from_idx][to_idx]) / static_cast<double>(row_sums_[from_idx]);
}

std::optional<MarkovChain::State> MarkovChain::SampleNext(const State& current, std::mt19937& rng) const {
  auto it = state_to_index_.find(current);
  if (it == state_to_index_.end() || row_sums_[it->second] == 0) {
    return std::nullopt;
  }
  size_t idx = it->second;
  std::vector<double> weights;
  for (size_t j = 0; j < index_to_state_.size(); ++j) {
    weights.push_back(static_cast<double>(counts_[idx][j]));
  }
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  size_t next_idx = dist(rng);
  return index_to_state_[next_idx];
}

std::vector<MarkovChain::State> MarkovChain::Generate(const State& start, size_t length, std::mt19937& rng) const {
  std::vector<State> result;
  if (length == 0) {
    return result;
  }
  result.push_back(start);
  for (size_t i = 1; i < length; ++i) {
    auto next = SampleNext(result.back(), rng);
    if (!next) {
      break;
    }
    result.push_back(*next);
  }
  return result;
}

std::vector<MarkovChain::State> MarkovChain::States() const {
  return index_to_state_;
}

} // namespace ptm
