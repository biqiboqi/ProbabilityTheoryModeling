#ifndef PTM_EXPERIMENTSTATS_HPP_
#define PTM_EXPERIMENTSTATS_HPP_

struct ExperimentStats {
  double empirical_mean = 0.0;
  double empirical_variance = 0.0;
  double mean_error = 0.0;
  double variance_error = 0.0;
};

#endif // PTM_EXPERIMENTSTATS_HPP_
