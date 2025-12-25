#include "BernoulliDistribution.hpp"

namespace ptm {

BernoulliDistribution::BernoulliDistribution(double p)
    : p_(p)
{}

double BernoulliDistribution::Pdf(double x) const {
    if (x != 0 && x != 1)
        return 0;

    return x ? p_ : 1 - p_;
}

double BernoulliDistribution::Cdf(double x) const {
    if (x < 0)
        return 0;
    else if (x < 1)
        return 1 - p_;
    else
        return 1;
}

double BernoulliDistribution::Sample(std::mt19937& rng) const {
    std::bernoulli_distribution distribution(p_);

    return distribution(rng);
}

double BernoulliDistribution::TheoreticalMean() const {
    return p_;
}

double BernoulliDistribution::TheoreticalVariance() const {
    return p_ * (1 - p_);
}

} // namespace ptm