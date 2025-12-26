#include <cmath>
#include <limits>
#include "CauchyDistribution.hpp"

namespace ptm {

CauchyDistribution::CauchyDistribution(double x0, double gamma) : x0_(x0), gamma_(gamma) {
}

double CauchyDistribution::Pdf(double x) const {
  return 1.0 / (M_PI * gamma_ * (1.0 + std::pow((x - x0_) / gamma_, 2)));
}

double CauchyDistribution::Cdf(double x) const {
  const double half = 0.5;
  return std::atan((x - x0_) / gamma_) / M_PI + half;
}

double CauchyDistribution::Sample(std::mt19937& rng) const {
  std::cauchy_distribution<double> dist(x0_, gamma_);
  return dist(rng);
}

double CauchyDistribution::TheoreticalMean() const {
  return std::numeric_limits<double>::quiet_NaN();
}
double CauchyDistribution::TheoreticalVariance() const {
  return std::numeric_limits<double>::quiet_NaN();
}

} // namespace ptm
