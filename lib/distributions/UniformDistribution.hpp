#ifndef PTM_UNIFORMDISTRIBUTION_HPP_
#define PTM_UNIFORMDISTRIBUTION_HPP_

#include <random>

#include "Distribution.hpp"

namespace ptm {

// Равномерное U(a, b)
class UniformDistribution : public Distribution {
public:
  UniformDistribution(double a, double b);

  [[nodiscard]] double Pdf(double x) const override;
  [[nodiscard]] double Cdf(double x) const override;
  double Sample(std::mt19937& rng) const override;

  [[nodiscard]] double TheoreticalMean() const override;
  [[nodiscard]] double TheoreticalVariance() const override;

private:
  double a_;
  double b_;
};

} // namespace ptm

#endif // PTM_UNIFORMDISTRIBUTION_HPP_
