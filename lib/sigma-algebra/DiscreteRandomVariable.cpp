#include "DiscreteRandomVariable.hpp"

namespace ptm {

DiscreteRandomVariable::DiscreteRandomVariable(const OutcomeSpace& omega,
                                               const ProbabilityMeasure& P,
                                               std::vector<double> values) :
    omega_(omega), P_(P), values_(std::move(values)) {
}

std::optional<double> DiscreteRandomVariable::Value(OutcomeSpace::OutcomeId id) const {
  if (id < values_.size()) {
    return values_[id];
  }
  return std::nullopt;
}

double DiscreteRandomVariable::ExpectedValue() const {
  double ev = 0.0;
  size_t n = omega_.GetSize();
  for (size_t i = 0; i < n; ++i) {
    ev += values_[i] * P_.GetAtomicProbability(i);
  }
  return ev;
}

} // namespace ptm
