#include <cmath>
#include "ExponentialDistribution.hpp"

namespace ptm {

ExponentialDistribution::ExponentialDistribution(double lambda) : lambda_(lambda) {
}

double ExponentialDistribution::Pdf(double x) const {
  return (x < 0.0) ? 0.0 : lambda_ * std::exp(-lambda_ * x);
}

double ExponentialDistribution::Cdf(double x) const {
  return (x < 0.0) ? 0.0 : 1.0 - std::exp(-lambda_ * x);
}

double ExponentialDistribution::Sample(std::mt19937& rng) const {
  std::exponential_distribution<double> dist(lambda_);
  return dist(rng);
}

double ExponentialDistribution::TheoreticalMean() const {
  return 1.0 / lambda_;
}
double ExponentialDistribution::TheoreticalVariance() const {
  return 1.0 / (lambda_ * lambda_);
}

} // namespace ptm
