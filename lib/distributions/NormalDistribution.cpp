#include <cmath>
#include "NormalDistribution.hpp"

constexpr double half = 0.5;
constexpr double two = 2;

namespace ptm {

NormalDistribution::NormalDistribution(double mean, double stddev) : mean_(mean), stddev_(stddev) {
}

double NormalDistribution::Pdf(double x) const {
  static const double inv_sqrt_2pi = 1.0 / std::sqrt(two * M_PI);
  double z = (x - mean_) / stddev_;
  return (inv_sqrt_2pi / stddev_) * std::exp(-half * z * z);
}

double NormalDistribution::Cdf(double x) const {
  return half * (1.0 + std::erf((x - mean_) / (stddev_ * std::sqrt(two))));
}

double NormalDistribution::Sample(std::mt19937& rng) const {
  std::normal_distribution<double> dist(mean_, stddev_);
  return dist(rng);
}

double NormalDistribution::TheoreticalMean() const {
  return mean_;
}
double NormalDistribution::TheoreticalVariance() const {
  return stddev_ * stddev_;
}
double NormalDistribution::GetMean() const {
  return mean_;
}
double NormalDistribution::GetStddev() const {
  return stddev_;
}

} // namespace ptm
