#include "LawOfLargeNumbersSimulator.hpp"


namespace ptm {

LawOfLargeNumbersSimulator::LawOfLargeNumbersSimulator(std::shared_ptr<Distribution> dist) : dist_(dist) {}

LLNPathResult LawOfLargeNumbersSimulator::Simulate(std::mt19937& rng, size_t max_n, size_t step) const {
    if (step == 0) return {};

    LLNPathResult result;
    result.entries.reserve(max_n / step);
    double sum = 0;

    for (size_t i = 0; i < max_n; ++i) {

        if (i % step != 0) {
            continue;
        }
        double sample = dist_->Sample(rng);
        sum += sample;

        LLNPathEntry entry = LLNPathEntry{
                                    .n = i,
                                    .sample_mean = sum / i,
                                    .abs_error = std::abs(sample - sum / i)
                                    };
        result.entries.push_back(entry);
    }
    return result;
}

[[nodiscard]] std::shared_ptr<Distribution> LawOfLargeNumbersSimulator::GetDistribution() const noexcept {
    return dist_;
}


}