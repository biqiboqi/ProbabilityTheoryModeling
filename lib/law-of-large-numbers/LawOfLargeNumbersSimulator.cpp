#include "LawOfLargeNumbersSimulator.hpp"

ptm::LLNPathResult ptm::LawOfLargeNumbersSimulator::Simulate(std::mt19937& rng, size_t max_n, size_t step) const {
  LLNPathResult result;

  double theoretical_mean = dist_->TheoreticalMean();

  double sum = 0.0;

  for (size_t n = 1; n <= max_n; ++n) {
    double x = dist_->Sample(rng);
    sum += x;

    if (n % step == 0) {
      double sample_mean = sum / n;
      double diff = std::abs(sample_mean - theoretical_mean);

      result.entries.emplace_back(n, sample_mean, diff);
    }
  }

  if (max_n % step != 0) {
    double sample_mean = sum / max_n;
    double diff = std::abs(sample_mean - theoretical_mean);

    result.entries.emplace_back(max_n, sample_mean, diff);
  }

  return result;
}
