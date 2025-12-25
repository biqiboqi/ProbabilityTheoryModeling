#include <numbers>

#include "BinomialDistribution.hpp"

namespace ptm {

BinomialDistribution::BinomialDistribution(unsigned int n, double p)
    : n_(n)
    , p_(p)
{}

double BinomialDistribution::Pdf(double x) const {
    const std::uint32_t moivre_laplace_formula_n_border = 100;
    const double moivre_laplace_formula_p_border = 0.1;

    if (std::round(x) != x || x < 0 || x > n_)
        return 0;

    std::uint32_t k = static_cast<std::uint32_t>(x);

    if (n_ < moivre_laplace_formula_n_border)
        return BernoulliFormula(k);
    else if (p_ < moivre_laplace_formula_p_border)
        return PoissonFormula(k);
    else
        return MoivreLaplaceFormula(k);
}

double BinomialDistribution::Cdf(double x) const {
    if (x < 0)
        return 0;

    if (x >= n_)
        return 1;

    double border = (x - n_ * p_) / std::sqrt(n_ * p_ * (1 - p_));

    return std::erf(border / std::sqrt(2)) / 2;
}

double BinomialDistribution::Sample(std::mt19937& rng) const {
    std::binomial_distribution<std::uint32_t> distribution(n_, p_);

    return distribution(rng);
}

double BinomialDistribution::TheoreticalMean() const {
    return n_ * p_;
}

double BinomialDistribution::TheoreticalVariance() const {
    return n_ * p_ * (1 - p_);
}

double BinomialDistribution::BernoulliFormula(std::uint32_t k) const {
    double result = 1;
    
    for (std::uint32_t i = 1; i <= k; ++i) {
        result *= n_ + 1 - i;
        result /= i;
    }

    result *= std::pow(p_, k);
    result *= std::pow(1 - p_, n_ - k);

    return result;
}

double BinomialDistribution::PoissonFormula(std::uint32_t k) const {
    double lambda = n_ * (1 - n_);
    double k_fact = 1;

    for (int i = k; i > 1; --i) {
        k_fact *= i;
    }

    return std::pow(lambda, k) * std::exp(-lambda) / k_fact;
}

double BinomialDistribution::MoivreLaplaceFormula(std::uint32_t k) const {
    double x = (k - p_ * n_) / std::sqrt(n_ * p_ * (1 - p_));

    return std::exp(- x * x / 2) / std::sqrt(2 * std::numbers::pi * n_ * p_ * (1 - p_));
}

} // namespace ptm