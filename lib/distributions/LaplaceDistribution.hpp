#ifndef PTM_LAPLACEDISTRIBUTION_HPP_
#define PTM_LAPLACEDISTRIBUTION_HPP_

#include <random>

#include "Distribution.hpp"

namespace ptm {

// Распределение Лапласа Laplace(mu, b)
class LaplaceDistribution : public Distribution {
public:
  LaplaceDistribution(double mu, double b);

  [[nodiscard]] double Pdf(double x) const override;
  [[nodiscard]] double Cdf(double x) const override;
  double Sample(std::mt19937& rng) const override;

  [[nodiscard]] double TheoreticalMean() const override;
  [[nodiscard]] double TheoreticalVariance() const override;

private:
  double mu_;
  double b_;
};

} // namespace ptm

#endif // PTM_LAPLACEDISTRIBUTION_HPP_
