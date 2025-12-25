#include <numeric>

#include "DistributionExperiment.hpp"

namespace ptm {

DistributionExperiment::DistributionExperiment(std::shared_ptr<Distribution> dist, size_t sample_size)
    : dist_(dist)
    , sample_size_(sample_size)
{}

ExperimentStats DistributionExperiment::Run(std::mt19937& rng) {
    std::vector<double> values;

    for (std::size_t i = 0; i < sample_size_; ++i) {
        values.push_back(dist_->Sample(rng));
    }

    ExperimentStats result;

    result.empirical_mean = std::accumulate(values.begin(), values.end(), 0.0) / sample_size_;

    for (double value : values) {
        result.empirical_variance += std::pow(result.empirical_mean - value, 2);
    }

    result.empirical_variance /= sample_size_;
    result.mean_error = result.empirical_mean - dist_->TheoreticalMean();
    result.variance_error = result.empirical_variance - dist_->TheoreticalVariance();

    return result;
}

std::vector<double> DistributionExperiment::EmpiricalCdf(const std::vector<double>& grid, std::mt19937& rng, std::size_t sample_size) {
    std::vector<double> values;

    for (std::size_t i = 0; i < sample_size; ++i) {
        values.push_back(dist_->Sample(rng));
    }

    std::sort(values.begin(), values.end());

    std::size_t current_value_index = 0;
    std::vector<double> result; 

    for (double point : grid) {
        while (current_value_index < sample_size && point > values[current_value_index]) {
            ++current_value_index;
        }

        result.push_back(static_cast<double>(current_value_index) / sample_size);
    }

    return result;
}

double DistributionExperiment::KolmogorovDistance(const std::vector<double>& grid, const std::vector<double>& empirical_cdf) const {
    double distance = 0;

    for (std::size_t i = 0; i < grid.size(); ++i) {
        distance = std::max(distance, std::abs(empirical_cdf[i] - dist_->Cdf(grid[i])));
    }

    return distance;
}

} // namespace ptm