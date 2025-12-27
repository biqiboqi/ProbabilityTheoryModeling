#include <algorithm>
#include "Event.hpp"

namespace ptm {

Event::Event(std::vector<bool> mask) : mask_(std::move(mask)) {
}

size_t Event::GetSize() const noexcept {
  return mask_.size();
}

bool Event::Contains(OutcomeSpace::OutcomeId id) const {
  return id < mask_.size() && mask_[id];
}

const std::vector<bool>& Event::GetMask() const noexcept {
  return mask_;
}

Event Event::Empty(std::size_t n) {
  return Event(std::vector<bool>(n, false));
}

Event Event::Full(std::size_t n) {
  return Event(std::vector<bool>(n, true));
}

Event Event::Complement(const Event& e) {
  std::vector<bool> new_mask = e.mask_;
  new_mask.flip();
  return Event(new_mask);
}

Event Event::Unite(const Event& a, const Event& b) {
  size_t n = a.mask_.size();
  std::vector<bool> result(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = a.mask_[i] || b.mask_[i];
  }
  return Event(result);
}

Event Event::Intersect(const Event& a, const Event& b) {
  size_t n = a.mask_.size();
  std::vector<bool> result(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = a.mask_[i] && b.mask_[i];
  }
  return Event(result);
}

} // namespace ptm
