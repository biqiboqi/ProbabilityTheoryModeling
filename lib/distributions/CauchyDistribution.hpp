#ifndef PTM_CAUCHYDISTRIBUTION_HPP_
#define PTM_CAUCHYDISTRIBUTION_HPP_

#include "Distribution.hpp"

namespace ptm {

// Распределение Коши (x0, gamma)
class CauchyDistribution : public Distribution {
public:
  CauchyDistribution(double x0, double gamma);

  [[nodiscard]] double Pdf(double x) const override;
  [[nodiscard]] double Cdf(double x) const override;
  double Sample(std::mt19937& rng) const override;

  [[nodiscard]] double TheoreticalMean() const override;
  [[nodiscard]] double TheoreticalVariance() const override;

private:
  double x0_;
  double gamma_;
};

} // namespace ptm

#endif // PTM_CAUCHYDISTRIBUTION_HPP_
