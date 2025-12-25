#include <numbers>

#include "NormalDistribution.hpp"

namespace ptm {

NormalDistribution::NormalDistribution(double mean, double stddev)
    : mean_(mean)
    , stddev_(stddev)
{}

double NormalDistribution::Pdf(double x) const {
    return std::exp(-std::pow(x - mean_, 2) / 2 / std::pow(stddev_, 2)) / stddev_ / std::sqrt(2 * std::numbers::pi);
}

double NormalDistribution::Cdf(double x) const {
    return 0.5 * (1 + std::erf((x - mean_) / stddev_/ std::sqrt(2)));
}

double NormalDistribution::Sample(std::mt19937& rng) const {
    std::normal_distribution distribution(mean_, stddev_);

    return distribution(rng);
}

double NormalDistribution::TheoreticalMean() const {
    return mean_;
}

double NormalDistribution::TheoreticalVariance() const {
    return std::pow(stddev_, 2);
}

double NormalDistribution::GetMean() const {
    return mean_;
}

double NormalDistribution::GetStddev() const {
    return stddev_;
}

} // namespace ptm