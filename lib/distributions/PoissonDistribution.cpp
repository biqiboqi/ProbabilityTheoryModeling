#include <cmath>
#include "PoissonDistribution.hpp"

constexpr double exponent = 1e-9;

namespace ptm {

PoissonDistribution::PoissonDistribution(double lambda) : lambda_(lambda) {
}

double PoissonDistribution::Pdf(double x) const {
  int k = static_cast<int>(std::round(x));
  if (k < 0 || std::abs(x - k) > exponent)
    return 0.0;
  return (std::pow(lambda_, k) * std::exp(-lambda_)) / std::tgamma(k + 1);
}

double PoissonDistribution::Cdf(double x) const {
  if (x < 0)
    return 0.0;
  double sum = 0;
  for (int k = 0; k <= static_cast<int>(x); ++k)
    sum += Pdf(static_cast<double>(k));
  return sum;
}

double PoissonDistribution::Sample(std::mt19937& rng) const {
  std::poisson_distribution<int> dist(lambda_);
  return static_cast<double>(dist(rng));
}

double PoissonDistribution::TheoreticalMean() const {
  return lambda_;
}
double PoissonDistribution::TheoreticalVariance() const {
  return lambda_;
}

} // namespace ptm
