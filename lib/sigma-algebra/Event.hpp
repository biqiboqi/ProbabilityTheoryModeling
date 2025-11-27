#ifndef PTM_EVENT_HPP_
#define PTM_EVENT_HPP_

#include <vector>

#include "OutcomeSpace.hpp"

namespace ptm {

class Event {
public:
  Event() = default;
  explicit Event(std::vector<bool> mask);

  size_t GetSize() const noexcept;
  bool Contains(OutcomeSpace::OutcomeId id) const;

  const std::vector<bool>& GetMask() const noexcept;

  static Event Empty(std::size_t n);
  static Event Full(std::size_t n);
  static Event Complement(const Event& e);
  static Event Unite(const Event& a, const Event& b);
  static Event Intersect(const Event& a, const Event& b);

private:
  std::vector<bool> mask_;
};

} // namespace ptm

#endif // PTM_EVENT_HPP_
