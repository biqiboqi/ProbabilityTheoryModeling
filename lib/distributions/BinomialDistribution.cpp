#include <cmath>
#include "BinomialDistribution.hpp"

constexpr double exponent = 1e-9;

namespace ptm {

BinomialDistribution::BinomialDistribution(unsigned int n, double p) : n_(n), p_(p) {
}

double BinomialDistribution::Pdf(double x) const {
  int k = static_cast<int>(std::round(x));
  if (k < 0 || k > static_cast<int>(n_) || std::abs(x - k) > exponent)
    return 0.0;
  double log_coeff = std::lgamma(n_ + 1) - std::lgamma(k + 1) - std::lgamma(n_ - k + 1);
  return std::exp(log_coeff + k * std::log(p_) + (n_ - k) * std::log(1.0 - p_));
}

double BinomialDistribution::Cdf(double x) const {
  if (x < 0.0)
    return 0.0;
  if (x >= n_)
    return 1.0;
  double sum = 0;
  for (int k = 0; k <= static_cast<int>(x); ++k)
    sum += Pdf(static_cast<double>(k));
  return sum;
}

double BinomialDistribution::Sample(std::mt19937& rng) const {
  std::binomial_distribution<int> dist((int) n_, p_);
  return static_cast<double>(dist(rng));
}

double BinomialDistribution::TheoreticalMean() const {
  return n_ * p_;
}
double BinomialDistribution::TheoreticalVariance() const {
  return n_ * p_ * (1.0 - p_);
}

} // namespace ptm
