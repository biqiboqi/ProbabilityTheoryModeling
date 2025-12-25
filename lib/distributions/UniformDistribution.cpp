#include "UniformDistribution.hpp"

namespace ptm {

UniformDistribution::UniformDistribution(double a, double b)
    : a_(a)
    , b_(b)
{}

double UniformDistribution::Pdf(double x) const {
    if (x < a_ || x > b_)
        return 0;

    return 1 / (b_ - a_);
}

double UniformDistribution::Cdf(double x) const {
    if (x < a_)
        return 0;
    if (x > b_)
        return 1;
    
    return (x - a_) / (b_ - a_);
}

double UniformDistribution::Sample(std::mt19937& rng) const {
    std::uniform_real_distribution distribution(a_, b_);

    return distribution(rng);
}

double UniformDistribution::TheoreticalMean() const {
    return (a_ + b_) / 2;
}

double UniformDistribution::TheoreticalVariance() const {
    return std::pow(b_ - a_, 2) / 12;
}

} // namespace ptm