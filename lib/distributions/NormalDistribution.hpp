#ifndef PTM_NORMALDISTRIBUTION_HPP_
#define PTM_NORMALDISTRIBUTION_HPP_

#include <random>

#include "Distribution.hpp"

namespace ptm {

// Нормальное N(mu, sigma^2)
class NormalDistribution : public Distribution {
public:
  NormalDistribution(double mean, double stddev);

  [[nodiscard]] double Pdf(double x) const override;
  [[nodiscard]] double Cdf(double x) const override;
  double Sample(std::mt19937& rng) const override;

  [[nodiscard]] double TheoreticalMean() const override;
  [[nodiscard]] double TheoreticalVariance() const override;

  [[nodiscard]] double GetMean() const;
  [[nodiscard]] double GetStddev() const;

private:
  double mean_;
  double stddev_;
};

} // namespace ptm

#endif // PTM_NORMALDISTRIBUTION_HPP_
