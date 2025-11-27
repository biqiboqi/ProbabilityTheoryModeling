#ifndef PTM_PROBABILITYMEASURE_HPP_
#define PTM_PROBABILITYMEASURE_HPP_

#include <vector>

#include "Event.hpp"
#include "OutcomeSpace.hpp"

namespace ptm {

class ProbabilityMeasure {
public:
  explicit ProbabilityMeasure(const OutcomeSpace& omega);

  // Задать P({ω_i}) = p_i
  void SetAtomicProbability(OutcomeSpace::OutcomeId id, double p);
  double GetAtomicProbability(OutcomeSpace::OutcomeId id) const;

  bool IsValid(double eps = 1e-9) const;

  double Probability(const Event& event) const;

private:
  const OutcomeSpace& omega_;
  std::vector<double> atom_probs_;
};

}; // namespace ptm

#endif // PTM_PROBABILITYMEASURE_HPP_
