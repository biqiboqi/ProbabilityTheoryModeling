#include "PoissonDistribution.hpp"

namespace ptm {

PoissonDistribution::PoissonDistribution(double lambda)
    : lambda_(lambda)
{}

double PoissonDistribution::Pdf(double x) const {
    if (x < 0 || x != std::round(x))
        return 0;

    std::uint32_t k = std::lround(x);
    std::uint32_t k_fact = 1;

    for (int i = k; i > 1; --i) {
        k_fact *= i;
    }

    return std::pow(lambda_, k) / k_fact * std::exp(-lambda_);
}

double PoissonDistribution::Cdf(double x) const {
    if (x < 0)
        return 0;

    double res = 0;
    
    for (int i = 0; i <= x; ++i) {
      res += Pdf(i);
    }

    return res;
}

double PoissonDistribution::Sample(std::mt19937& rng) const {
    std::poisson_distribution distribution(lambda_);

    return distribution(rng);
}

double PoissonDistribution::TheoreticalMean() const {
    return lambda_;
}

double PoissonDistribution::TheoreticalVariance() const {
    return lambda_;
}

} // namespace ptm