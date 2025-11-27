#ifndef PTM_BINOMIALDISTRIBUTION_HPP_
#define PTM_BINOMIALDISTRIBUTION_HPP_

#include <random>

#include "Distribution.hpp"

namespace ptm {

// Биномиальное Binomial(n, p)
class BinomialDistribution : public Distribution {
public:
  BinomialDistribution(unsigned int n, double p);

  [[nodiscard]] double Pdf(double x) const override;
  [[nodiscard]] double Cdf(double x) const override;
  double Sample(std::mt19937& rng) const override;

  [[nodiscard]] double TheoreticalMean() const override;
  [[nodiscard]] double TheoreticalVariance() const override;

private:
  unsigned int n_;
  double p_;
};

} // namespace ptm

#endif // PTM_BINOMIALDISTRIBUTION_HPP_
