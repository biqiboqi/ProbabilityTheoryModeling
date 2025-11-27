#ifndef PTM_BERNOULLIDISTRIBUTION_HPP_
#define PTM_BERNOULLIDISTRIBUTION_HPP_

#include <random>

#include "Distribution.hpp"

namespace ptm {

// Бернулли Bernoulli(p)
class BernoulliDistribution : public Distribution {
public:
  explicit BernoulliDistribution(double p);

  [[nodiscard]] double Pdf(double x) const override;
  [[nodiscard]] double Cdf(double x) const override;
  double Sample(std::mt19937& rng) const override;

  [[nodiscard]] double TheoreticalMean() const override;
  [[nodiscard]] double TheoreticalVariance() const override;

private:
  double p_;
};

} // namespace ptm

#endif // PTM_BERNOULLIDISTRIBUTION_HPP_
