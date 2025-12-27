#include <gtest/gtest.h>
#include <random>

#include "lib/distributions/BernoulliDistribution.hpp"
#include "lib/distributions/BinomialDistribution.hpp"
#include "lib/distributions/CauchyDistribution.hpp"
#include "lib/distributions/ExponentialDistribution.hpp"
#include "lib/distributions/LaplaceDistribution.hpp"
#include "lib/distributions/UniformDistribution.hpp"
#include "lib/law-of-large-numbers/LawOfLargeNumbersSimulator.hpp"

TEST(LawOfLargeNumbersTest, BernoulliMeanConverges) {
  using namespace ptm;

  std::mt19937 rng(123);

  auto dist = std::make_shared<BernoulliDistribution>(0.3);
  LawOfLargeNumbersSimulator sim(dist);

  size_t max_n = 100000;
  size_t step = 5000;

  LLNPathResult result = sim.Simulate(rng, max_n, step);

  ASSERT_FALSE(result.entries.empty());

  const double theoretical_mean = dist->TheoreticalMean();

  double first_error = result.entries.front().abs_error;
  double last_error = result.entries.back().abs_error;
  double last_mean = result.entries.back().sample_mean;

  EXPECT_GT(first_error, last_error);

  EXPECT_NEAR(last_mean, theoretical_mean, 0.05);

  EXPECT_LT(last_error, 0.05);

  for (std::size_t i = 1; i < result.entries.size(); ++i) {
    EXPECT_EQ(result.entries[i].n, result.entries[i - 1].n + step);
  }
}

TEST(LawOfLargeNumbersTest, UniformMeanConverges) {
    using namespace ptm;

    std::mt19937 rng(456);

    auto dist = std::make_shared<UniformDistribution>(2.0, 5.0);
    LawOfLargeNumbersSimulator sim(dist);

    size_t max_n = 50000;
    size_t step = 1000;

    LLNPathResult result = sim.Simulate(rng, max_n, step);

    ASSERT_FALSE(result.entries.empty());

    const double theoretical_mean = dist->TheoreticalMean();

    double first_error = result.entries.front().abs_error;
    double last_error = result.entries.back().abs_error;
    double last_mean = result.entries.back().sample_mean;

    // Проверяем сходимость
    EXPECT_GT(first_error, last_error);
    EXPECT_NEAR(last_mean, theoretical_mean, 0.1);
    EXPECT_LT(last_error, 0.1);

    // Проверяем, что все ошибки неотрицательные
    for (const auto& record : result.entries) {
        EXPECT_GE(record.abs_error, 0.0);
    }
}

TEST(LawOfLargeNumbersTest, ExponentialMeanConverges) {
    using namespace ptm;

    std::mt19937 rng(789);

    auto dist = std::make_shared<ExponentialDistribution>(2.0);
    LawOfLargeNumbersSimulator sim(dist);

    size_t max_n = 80000;
    size_t step = 2000;

    LLNPathResult result = sim.Simulate(rng, max_n, step);

    ASSERT_FALSE(result.entries.empty());

    const double theoretical_mean = dist->TheoreticalMean();

    double last_error = result.entries.back().abs_error;
    double last_mean = result.entries.back().sample_mean;

    // Проверяем сходимость
    EXPECT_NEAR(last_mean, theoretical_mean, 0.15);
    EXPECT_LT(last_error, 0.15);

    // Проверяем, что размер выборки увеличивается
    EXPECT_EQ(result.entries.back().n, max_n);
}

TEST(LawOfLargeNumbersTest, LaplaceMeanConverges) {
    using namespace ptm;

    std::mt19937 rng(101112);

    auto dist = std::make_shared<LaplaceDistribution>(3.0, 1.5);
    LawOfLargeNumbersSimulator sim(dist);

    size_t max_n = 70000;
    size_t step = 1500;

    LLNPathResult result = sim.Simulate(rng, max_n, step);

    ASSERT_FALSE(result.entries.empty());

    const double theoretical_mean = dist->TheoreticalMean();

    double last_error = result.entries.back().abs_error;
    double last_mean = result.entries.back().sample_mean;

    // Проверяем сходимость
    EXPECT_NEAR(last_mean, theoretical_mean, 0.1);
    EXPECT_LT(last_error, 0.1);

    // Проверяем монотонность n
    for (size_t i = 1; i < result.entries.size(); ++i) {
        EXPECT_LT(result.entries[i - 1].n, result.entries[i].n);
    }
}

TEST(LawOfLargeNumbersTest, CauchyNoMean) {
  using namespace ptm;

  std::mt19937 rng(131415);

  auto dist = std::make_shared<CauchyDistribution>(0.0, 1.0);
  LawOfLargeNumbersSimulator sim(dist);

  size_t max_n = 30000;
  size_t step = 1000;

  auto result = sim.Simulate(rng, max_n, step);

  ASSERT_FALSE(result.entries.empty());

  EXPECT_TRUE(std::isnan(dist->TheoreticalMean()));

  for (const auto& entry : result.entries) {
    EXPECT_FALSE(std::isinf(entry.sample_mean));
  }
}