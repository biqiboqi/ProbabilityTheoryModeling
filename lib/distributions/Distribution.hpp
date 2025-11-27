#ifndef PTM_DISTRIBUTION_HPP_
#define PTM_DISTRIBUTION_HPP_

#include <random>

namespace ptm {

// Базовый класс для распределения
class Distribution { // NOLINT(cppcoreguidelines-special-member-functions)
public:
  virtual ~Distribution() = default;

  // Для дискретных распределений pdf(x) трактуем как P(X = x)
  [[nodiscard]] virtual double Pdf(double x) const = 0;

  // F(x) = P(X <= x)
  [[nodiscard]] virtual double Cdf(double x) const = 0;

  // Генерация выборочного значения
  virtual double Sample(std::mt19937& rng) const = 0;

  // Теоретическое матожидание и дисперсия (если определены).
  // Для распределений, где это не определено - можно вернуть NaN.
  [[nodiscard]] virtual double TheoreticalMean() const = 0;
  [[nodiscard]] virtual double TheoreticalVariance() const = 0;
};

} // namespace ptm

#endif // PTM_DISTRIBUTION_HPP_
