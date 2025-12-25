#include <numbers>

#include "CauchyDistribution.hpp"

namespace ptm {

CauchyDistribution::CauchyDistribution(double x0, double gamma)
    : x0_(x0)
    , gamma_(gamma)
{}

double CauchyDistribution::Pdf(double x) const {
    return 1 / std::numbers::pi / gamma_ / (1 + std::pow((x - x0_) / gamma_, 2));
}

double CauchyDistribution::Cdf(double x) const {
    return 0.5 + std::atan((x - x0_) / gamma_) / std::numbers::pi;
}

double CauchyDistribution::Sample(std::mt19937& rng) const {
    std::normal_distribution<double> distribution(0, 1);

    return x0_ + gamma_ * distribution(rng) / distribution(rng);
}

double CauchyDistribution::TheoreticalMean() const {
    return std::nan("");
}

double CauchyDistribution::TheoreticalVariance() const {
    return std::nan("");
}

} // namespace ptm