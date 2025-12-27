#include <algorithm>
#include <cmath>
#include "DistributionExperiment.hpp"

namespace ptm {

DistributionExperiment::DistributionExperiment(std::shared_ptr<Distribution> dist, size_t sample_size) :
    dist_(std::move(dist)), sample_size_(sample_size) {
}

ExperimentStats DistributionExperiment::Run(std::mt19937& rng) {
  std::vector<double> samples;
  samples.reserve(sample_size_);

  double sum = 0.0;
  for (size_t i = 0; i < sample_size_; ++i) {
    double s = dist_->Sample(rng);
    samples.push_back(s);
    sum += s;
  }

  double emp_mean = sum / (double) sample_size_;

  double var_sum = 0.0;
  for (double s : samples) {
    var_sum += std::pow(s - emp_mean, 2);
  }
  double emp_var = var_sum / (double) (sample_size_ - 1);

  ExperimentStats stats;
  stats.empirical_mean = emp_mean;
  stats.empirical_variance = emp_var;

  double t_mean = dist_->TheoreticalMean();
  double t_var = dist_->TheoreticalVariance();

  if (!std::isnan(t_mean))
    stats.mean_error = std::abs(emp_mean - t_mean);
  if (!std::isnan(t_var))
    stats.variance_error = std::abs(emp_var - t_var);

  return stats;
}

std::vector<double> DistributionExperiment::EmpiricalCdf(const std::vector<double>& grid,
                                                         std::mt19937& rng,
                                                         std::size_t sample_size) {
  std::vector<double> samples(sample_size);
  for (size_t i = 0; i < sample_size; ++i) {
    samples[i] = dist_->Sample(rng);
  }
  std::sort(samples.begin(), samples.end());

  std::vector<double> ecdf;
  ecdf.reserve(grid.size());

  for (double x : grid) {
    auto it = std::upper_bound(samples.begin(), samples.end(), x);
    auto count = (double) std::distance(samples.begin(), it);
    ecdf.push_back(count / (double) sample_size);
  }
  return ecdf;
}

double DistributionExperiment::KolmogorovDistance(const std::vector<double>& grid,
                                                  const std::vector<double>& empirical_cdf) const {
  double max_dist = 0.0;
  for (size_t i = 0; i < grid.size(); ++i) {
    double t_cdf = dist_->Cdf(grid[i]);
    double dist = std::abs(empirical_cdf[i] - t_cdf);
    if (dist > max_dist)
      max_dist = dist;
  }
  return max_dist;
}

} // namespace ptm
