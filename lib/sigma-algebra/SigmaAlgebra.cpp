#include <set>
#include "SigmaAlgebra.hpp"

namespace ptm {

SigmaAlgebra::SigmaAlgebra(const OutcomeSpace& omega, std::vector<Event> events) :
    omega_(omega), events_(std::move(events)) {
}

const OutcomeSpace& SigmaAlgebra::GetOutcomeSpace() const noexcept {
  return omega_;
}

const std::vector<Event>& SigmaAlgebra::GetEvents() const noexcept {
  return events_;
}

bool SigmaAlgebra::IsSigmaAlgebra() const {
  size_t n = omega_.GetSize();
  std::set<std::vector<bool>> storage;
  for (const auto& e : events_)
    storage.insert(e.GetMask());

  if (storage.find(Event::Empty(n).GetMask()) == storage.end())
    return false;
  if (storage.find(Event::Full(n).GetMask()) == storage.end())
    return false;

  for (const auto& mask : storage) {
    Event comp = Event::Complement(Event(mask));
    if (storage.find(comp.GetMask()) == storage.end())
      return false;

    for (const auto& mask2 : storage) {
      Event united = Event::Unite(Event(mask), Event(mask2));
      if (storage.find(united.GetMask()) == storage.end())
        return false;
    }
  }
  return true;
}

SigmaAlgebra SigmaAlgebra::Generate(const OutcomeSpace& omega, const std::vector<Event>& generators) {
  size_t n = omega.GetSize();
  std::set<std::vector<bool>> current_sets;

  current_sets.insert(Event::Empty(n).GetMask());
  current_sets.insert(Event::Full(n).GetMask());
  for (const auto& g : generators) {
    current_sets.insert(g.GetMask());
  }

  bool changed = true;
  while (changed) {
    changed = false;
    std::set<std::vector<bool>> next_step = current_sets;

    for (const auto& mask : current_sets) {
      auto comp = Event::Complement(Event(mask)).GetMask();
      if (next_step.insert(comp).second)
        changed = true;

      for (const auto& mask2 : current_sets) {
        auto uni = Event::Unite(Event(mask), Event(mask2)).GetMask();
        if (next_step.insert(uni).second)
          changed = true;
      }
    }
    current_sets = std::move(next_step);
  }

  std::vector<Event> final_events;
  final_events.reserve(current_sets.size());
  for (const auto& mask : current_sets) {
    final_events.emplace_back(mask);
  }
  return {omega, final_events};
}

} // namespace ptm
