#include <cmath>
#include "LaplaceDistribution.hpp"

constexpr double half = 0.5;
constexpr double two = 2;

namespace ptm {

LaplaceDistribution::LaplaceDistribution(double mu, double b) : mu_(mu), b_(b) {
}

double LaplaceDistribution::Pdf(double x) const {
  return (1.0 / (two * b_)) * std::exp(-std::abs(x - mu_) / b_);
}

double LaplaceDistribution::Cdf(double x) const {
  if (x < mu_)
    return half * std::exp((x - mu_) / b_);
  return 1.0 - half * std::exp(-(x - mu_) / b_);
}

double LaplaceDistribution::Sample(std::mt19937& rng) const {
  std::uniform_real_distribution<double> dist(-half, half);
  double u = dist(rng);
  return mu_ - b_ * (u < 0 ? -1.0 : 1.0) * std::log(1.0 - two * std::abs(u));
}

double LaplaceDistribution::TheoreticalMean() const {
  return mu_;
}

double LaplaceDistribution::TheoreticalVariance() const {
  return two * b_ * b_;
}

} // namespace ptm
