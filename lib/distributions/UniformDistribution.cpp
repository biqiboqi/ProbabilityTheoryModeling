#include "UniformDistribution.hpp"

constexpr double two = 2;
constexpr double twelve = 12;

namespace ptm {

UniformDistribution::UniformDistribution(double a, double b) : a_(a), b_(b) {
}

double UniformDistribution::Pdf(double x) const {
  if (x >= a_ && x <= b_)
    return 1.0 / (b_ - a_);
  return 0.0;
}

double UniformDistribution::Cdf(double x) const {
  if (x < a_)
    return 0.0;
  if (x > b_)
    return 1.0;
  return (x - a_) / (b_ - a_);
}

double UniformDistribution::Sample(std::mt19937& rng) const {
  std::uniform_real_distribution<double> dist(a_, b_);
  return dist(rng);
}

double UniformDistribution::TheoreticalMean() const {
  return (a_ + b_) / two;
}

double UniformDistribution::TheoreticalVariance() const {
  return std::pow(b_ - a_, 2) / twelve;
}

} // namespace ptm
