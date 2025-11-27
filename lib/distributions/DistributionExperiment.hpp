#ifndef PTM_DISTRIBUTIONEXPERIMENT_HPP_
#define PTM_DISTRIBUTIONEXPERIMENT_HPP_

#include <memory>
#include <random>

#include "Distribution.hpp"
#include "ExperimentStats.hpp"

namespace ptm {

// Класс для массовых экспериментов по моделированию распределений
class DistributionExperiment {
public:
  DistributionExperiment(std::shared_ptr<Distribution> dist, size_t sample_size);

  ExperimentStats Run(std::mt19937& rng);

  // Эмпирическая CDF на сетке точек
  std::vector<double> EmpiricalCdf(const std::vector<double>& grid, std::mt19937& rng, std::size_t sample_size);

  // Оценка статистики Колмогорова между эмпирической и теоретической CDF
  [[nodiscard]] double KolmogorovDistance(const std::vector<double>& grid,
                                          const std::vector<double>& empirical_cdf) const;

private:
  std::shared_ptr<Distribution> dist_;
  std::size_t sample_size_;
};

} // namespace ptm

#endif // PTM_DISTRIBUTIONEXPERIMENT_HPP_
