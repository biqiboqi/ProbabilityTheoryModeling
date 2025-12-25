#ifndef PTM_LAWOFLARGENUMBERSSIMULATOR_HPP_
#define PTM_LAWOFLARGENUMBERSSIMULATOR_HPP_

#include <memory>
#include <random>

#include "LLNPathResult.hpp"
#include "distributions/Distribution.hpp"

namespace ptm {
class Distribution;

class LawOfLargeNumbersSimulator {
public:
  explicit LawOfLargeNumbersSimulator(std::shared_ptr<Distribution> dist);

  // Смоделировать одну траекторию по LLN:
  //
  // - max_n: максимальное N
  // - step: шаг, через который будем сохранять статистику (например, 100, 1000,...)
  //
  // Алгоритм:
  // 1) генерируем X_1, ..., X_max_n
  // 2) считаем префиксные суммы и выборочные средние
  // 3) для n кратных step сохраняем (n, mean_n, |mean_n - mu|)
  LLNPathResult Simulate(std::mt19937& rng, size_t max_n, size_t step) const;

  // Доступ к распределению
  [[nodiscard]] std::shared_ptr<Distribution> GetDistribution() const noexcept;

private:
  std::shared_ptr<Distribution> dist_;
};

} // namespace ptm

#endif // PTM_LAWOFLARGENUMBERSSIMULATOR_HPP_
