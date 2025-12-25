#include "ExponentialDistribution.hpp"

namespace ptm {

ExponentialDistribution::ExponentialDistribution(double lambda)
    : lambda_(lambda)
{}

double ExponentialDistribution::Pdf(double x) const {
    if (x < 0)
        return 0;

    return lambda_ * std::exp(-lambda_ * x);
}

double ExponentialDistribution::Cdf(double x) const {
    return 1 - std::exp(-lambda_ * x);
}

double ExponentialDistribution::Sample(std::mt19937& rng) const {
    std::exponential_distribution distribution(lambda_);

    return distribution(rng);
}

double ExponentialDistribution::TheoreticalMean() const {
    return 1 / lambda_;
}

double ExponentialDistribution::TheoreticalVariance() const {
    return 1 / std::pow(lambda_, 2);
}

} // namespace ptm