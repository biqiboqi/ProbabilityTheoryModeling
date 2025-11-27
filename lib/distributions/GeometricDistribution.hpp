#ifndef PTM_GEOMETRICDISTRIBUTION_HPP_
#define PTM_GEOMETRICDISTRIBUTION_HPP_

#include <random>

#include "Distribution.hpp"

namespace ptm {

// Геометрическое Geom(p) на {1, 2, 3, ...}
class GeometricDistribution : public Distribution {
public:
  explicit GeometricDistribution(double p);

  [[nodiscard]] double Pdf(double x) const override;
  [[nodiscard]] double Cdf(double x) const override;
  double Sample(std::mt19937& rng) const override;

  [[nodiscard]] double TheoreticalMean() const override;
  [[nodiscard]] double TheoreticalVariance() const override;

private:
  double p_;
};

} // namespace ptm

#endif // PTM_GEOMETRICDISTRIBUTION_HPP_
