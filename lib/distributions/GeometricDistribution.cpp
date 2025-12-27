#include <cmath>
#include "GeometricDistribution.hpp"

constexpr double exponent = 1e-9;

namespace ptm {

GeometricDistribution::GeometricDistribution(double p) : p_(p) {
}

double GeometricDistribution::Pdf(double x) const {
  int k = static_cast<int>(std::round(x));
  if (k < 1 || std::abs(x - k) > exponent)
    return 0.0;
  return std::pow(1.0 - p_, k - 1) * p_;
}

double GeometricDistribution::Cdf(double x) const {
  if (x < 1.0)
    return 0.0;
  return 1.0 - std::pow(1.0 - p_, std::floor(x));
}

double GeometricDistribution::Sample(std::mt19937& rng) const {
  std::geometric_distribution<int> dist(p_);
  return static_cast<double>(dist(rng) + 1);
}

double GeometricDistribution::TheoreticalMean() const {
  return 1.0 / p_;
}
double GeometricDistribution::TheoreticalVariance() const {
  return (1.0 - p_) / (p_ * p_);
}

} // namespace ptm
