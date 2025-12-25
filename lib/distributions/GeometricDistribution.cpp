#include "GeometricDistribution.hpp"

namespace ptm {

GeometricDistribution::GeometricDistribution(double p)
    : p_(p)
{}

double GeometricDistribution::Pdf(double x) const {
    if (x < 1 || std::round(x) != x)
        return 0;

    return std::pow(1- p_, x - 1) * p_;
}

double GeometricDistribution::Cdf(double x) const {
    if (x < 1)
        return 0;

    return 1 - std::pow(1 - p_, std::floor(x));
}

double GeometricDistribution::Sample(std::mt19937& rng) const {
    std::geometric_distribution<std::uint32_t> distribution(p_);

    return distribution(rng);
}

double GeometricDistribution::TheoreticalMean() const {
    return 1 / p_;
}

double GeometricDistribution::TheoreticalVariance() const {
    return (1 - p_) / std::pow(p_, 2);
}

} // namespace ptm