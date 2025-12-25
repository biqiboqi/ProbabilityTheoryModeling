#include "LaplaceDistribution.hpp"

namespace ptm {

LaplaceDistribution::LaplaceDistribution(double mu, double b)
    : mu_(mu)
    , b_(b)
{}

double LaplaceDistribution::Pdf(double x) const {
    return std::exp(-std::abs(x - mu_) / b_) / 2 / b_;
}

double LaplaceDistribution::Cdf(double x) const {
    if (x < mu_)
        return 0.5 * std::exp((x - mu_) / b_);
    else
        return 1 - 0.5 * std::exp(-(x - mu_) / b_);
}

double LaplaceDistribution::Sample(std::mt19937& rng) const {
    std::exponential_distribution distribution(1 / b_);
    
    return mu_ + distribution(rng) * (rng() % 2 ? -1 : 1);
}

double LaplaceDistribution::TheoreticalMean() const {
    return mu_;
}

double LaplaceDistribution::TheoreticalVariance() const {
    return 2 * std::pow(b_, 2);
}

} // namespace ptm