#include "LawOfLargeNumbersSimulator.hpp"


namespace ptm {

LawOfLargeNumbersSimulator::LawOfLargeNumbersSimulator(std::shared_ptr<Distribution> dist) 
    : dist_(dist)
{}

LLNPathResult LawOfLargeNumbersSimulator::Simulate(std::mt19937& rng, std::size_t max_n, std::size_t step) const {
    std::vector<double> values;
    LLNPathResult result;
    double sum = 0;

    for (std::size_t i = 1; i <= max_n; ++i) {
        values.push_back(dist_->Sample(rng));
        sum += values.back();

        if (i % step == 0) {
            LLNPathEntry entry{};
            entry.n = i;
            entry.sample_mean = sum / i;
            entry.abs_error = std::abs(entry.sample_mean - dist_->TheoreticalMean());

            result.entries.push_back(entry);
        }
    }

    return result;
}

std::shared_ptr<Distribution> LawOfLargeNumbersSimulator::GetDistribution() const noexcept {
    return dist_;
}

} // namespace ptm
