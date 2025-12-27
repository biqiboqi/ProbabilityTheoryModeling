#include <stdexcept>
#include "OutcomeSpace.hpp"

namespace ptm {

OutcomeSpace::OutcomeId OutcomeSpace::AddOutcome(const std::string& name) {
  names_.push_back(name);
  return names_.size() - 1;
}

size_t OutcomeSpace::GetSize() const noexcept {
  return names_.size();
}

const std::string& OutcomeSpace::GetName(OutcomeId id) const {
  if (id >= names_.size()) {
    throw std::out_of_range("OutcomeId out of range");
  }
  return names_[id];
}

} // namespace ptm
