#include <cmath>
#include "ProbabilityMeasure.hpp"

namespace ptm {

ProbabilityMeasure::ProbabilityMeasure(const OutcomeSpace& omega) : omega_(omega), atom_probs_(omega.GetSize(), 0.0) {
}

void ProbabilityMeasure::SetAtomicProbability(OutcomeSpace::OutcomeId id, double p) {
  if (id < atom_probs_.size()) {
    atom_probs_[id] = p;
  }
}

double ProbabilityMeasure::GetAtomicProbability(OutcomeSpace::OutcomeId id) const {
  return (id < atom_probs_.size()) ? atom_probs_[id] : 0.0;
}

bool ProbabilityMeasure::IsValid(double eps) const {
  double sum = 0.0;
  for (double p : atom_probs_) {
    if (p < 0.0)
      return false;
    sum += p;
  }
  return std::abs(sum - 1.0) < eps;
}

double ProbabilityMeasure::Probability(const Event& event) const {
  double p_sum = 0.0;
  const auto& mask = event.GetMask();
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) {
      p_sum += atom_probs_[i];
    }
  }
  return p_sum;
}

} // namespace ptm
