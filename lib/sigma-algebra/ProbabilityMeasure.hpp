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
  [[nodiscard]] double GetAtomicProbability(OutcomeSpace::OutcomeId id) const;

  [[nodiscard]] bool IsValid(double eps) const;

  [[nodiscard]] double Probability(const Event& event) const;

private:
  const OutcomeSpace& omega_;
  std::vector<double> atom_probs_;
};

}; // namespace ptm

#endif // PTM_PROBABILITYMEASURE_HPP_
