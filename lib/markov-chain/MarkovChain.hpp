#ifndef PTM_MARKOVCHAIN_HPP_
#define PTM_MARKOVCHAIN_HPP_

#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace ptm {

class MarkovChain {
public:
  using State = std::string;

  MarkovChain() = default;

  // Обучение на одной последовательности (инкрементально)
  void Train(const std::vector<State>& sequence);

  // Получить распределение P(next | current) как map state -> prob
  [[nodiscard]] std::unordered_map<State, double> NextDistribution(const State& current) const;

  // Вероятность конкретного перехода P(to | from). 0, если переход или состояние не встречались
  double TransitionProbability(const State& from, const State& to) const;

  // Сгенерировать следующий токен из распределения P(next | current)
  // Если у current нет исходящих переходов, возвращает std::nullopt
  std::optional<State> SampleNext(const State& current, std::mt19937& rng) const;

  // Сгенерировать последовательность длины length, начиная с start
  std::vector<State> Generate(const State& start, size_t length, std::mt19937& rng) const;

  // Все известные состояния
  std::vector<State> States() const;

  // Знает ли цепь об этом состоянии
  bool HasState(const State& state) const;

private:
  // отображение state -> index
  std::unordered_map<State, size_t> state_to_index_;
  std::vector<State> index_to_state_;

  // counts_[i][j] = c_ij, row_sums_[i] = sum_j c_ij
  std::vector<std::vector<size_t>> counts_;
  std::vector<size_t> row_sums_;

  size_t ensureState(const State& s);
};

} // namespace ptm

#endif // PTM_MARKOVCHAIN_HPP_
