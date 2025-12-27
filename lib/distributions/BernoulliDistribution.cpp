#include <cmath>
#include "BernoulliDistribution.hpp"

namespace ptm {

BernoulliDistribution::BernoulliDistribution(double p) : p_(p) {
}

double BernoulliDistribution::Pdf(double x) const {
  if (x == 1.0)
    return p_;
  if (x == 0.0)
    return 1.0 - p_;
  return 0.0;
}

double BernoulliDistribution::Cdf(double x) const {
  if (x < 0.0)
    return 0.0;
  if (x < 1.0)
    return 1.0 - p_;
  return 1.0;
}

double BernoulliDistribution::Sample(std::mt19937& rng) const {
  std::bernoulli_distribution dist(p_);
  return dist(rng) ? 1.0 : 0.0;
}

double BernoulliDistribution::TheoreticalMean() const {
  return p_;
}
double BernoulliDistribution::TheoreticalVariance() const {
  return p_ * (1.0 - p_);
}

} // namespace ptm
