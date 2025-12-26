#include <gtest/gtest.h>

#include <cmath>

#include "lib/distributions/BernoulliDistribution.hpp"
#include "lib/distributions/BinomialDistribution.hpp"
#include "lib/distributions/CauchyDistribution.hpp"
#include "lib/distributions/DistributionExperiment.hpp"
#include "lib/distributions/ExponentialDistribution.hpp"
#include "lib/distributions/GeometricDistribution.hpp"
#include "lib/distributions/LaplaceDistribution.hpp"
#include "lib/distributions/NormalDistribution.hpp"
#include "lib/distributions/PoissonDistribution.hpp"
#include "lib/distributions/UniformDistribution.hpp"

TEST(DistributionTest, NormalDistributionBasicProperties) {
  using namespace ptm;

  NormalDistribution nd(0.0, 1.0);
  double pdf0 = nd.Pdf(0.0);
  double cdf0 = nd.Cdf(0.0);

  EXPECT_NEAR(pdf0, 0.3989, 1e-3);

  EXPECT_NEAR(cdf0, 0.5, 1e-3);
}

TEST(DistributionExperimentTest, EmpiricalMeanCloseToTheoretical) {
  using namespace ptm;

  std::mt19937 rng(123);

  auto dist = std::make_shared<NormalDistribution>(5.0, 2.0);
  DistributionExperiment experiment(dist, 20000);

  auto stats = experiment.Run(rng);

  EXPECT_NEAR(stats.empirical_mean, dist->TheoreticalMean(), 0.1);
  EXPECT_NEAR(stats.empirical_variance, dist->TheoreticalVariance(), 0.3);
}

TEST(DistributionTest, UniformDistributionBasicProperties) {
  using namespace ptm;

  UniformDistribution ud(0.0, 2.0);
  EXPECT_NEAR(ud.Pdf(1.0), 0.5, 1e-9);
  EXPECT_NEAR(ud.Cdf(0.0), 0.0, 1e-9);
  EXPECT_NEAR(ud.Cdf(2.0), 1.0, 1e-9);

  EXPECT_NEAR(ud.TheoreticalMean(), 1.0, 1e-9);
  EXPECT_NEAR(ud.TheoreticalVariance(), 1.0 / 3.0, 1e-9);
}

TEST(DistributionTest, BernoulliDistributionBasic) {
  using namespace ptm;

  BernoulliDistribution bd(0.3);
  EXPECT_NEAR(bd.Pdf(0.0), 0.7, 1e-9);
  EXPECT_NEAR(bd.Pdf(1.0), 0.3, 1e-9);
  EXPECT_NEAR(bd.Cdf(0.5), 0.7, 1e-9);
  EXPECT_NEAR(bd.TheoreticalMean(), 0.3, 1e-9);
  EXPECT_NEAR(bd.TheoreticalVariance(), 0.21, 1e-9);
}

TEST(DistributionTest, BinomialDistributionBasic) {
  using namespace ptm;

  BinomialDistribution bd(10, 0.5);
  double p5 = bd.Pdf(5.0);
  EXPECT_NEAR(p5, 0.246, 1e-2);

  EXPECT_NEAR(bd.TheoreticalMean(), 5.0, 1e-9);
  EXPECT_NEAR(bd.TheoreticalVariance(), 2.5, 1e-9);
}

TEST(DistributionTest, GeometricDistributionBasic) {
  using namespace ptm;

  double p = 0.4;
  GeometricDistribution gd(p);

  EXPECT_NEAR(gd.Pdf(1.0), p, 1e-9);
  EXPECT_NEAR(gd.Cdf(3.0), 1.0 - std::pow(1.0 - p, 3), 1e-9);

  EXPECT_NEAR(gd.TheoreticalMean(), 1.0 / p, 1e-9);
  EXPECT_NEAR(gd.TheoreticalVariance(), (1.0 - p) / (p * p), 1e-9);
}

TEST(DistributionTest, PoissonDistributionBasic) {
  using namespace ptm;

  double lambda = 3.0;
  PoissonDistribution pd(lambda);

  EXPECT_NEAR(pd.Pdf(0.0), std::exp(-lambda), 1e-9);

  EXPECT_NEAR(pd.TheoreticalMean(), lambda, 1e-9);
  EXPECT_NEAR(pd.TheoreticalVariance(), lambda, 1e-9);

  double cdf1 = pd.Cdf(1.0);
  double p0 = pd.Pdf(0.0);
  double p1 = pd.Pdf(1.0);
  EXPECT_NEAR(cdf1, p0 + p1, 1e-6);
}

TEST(DistributionTest, CauchyDistributionBasic) {
  using namespace ptm;

  CauchyDistribution cd(0.0, 1.0);
  double pi = 3.14159265358979323846;
  EXPECT_NEAR(cd.Pdf(0.0), 1.0 / (pi * 1.0), 1e-9);
  EXPECT_NEAR(cd.Cdf(0.0), 0.5, 1e-9);
}

TEST(DistributionTest, LaplaceDistributionBasic) {
  using namespace ptm;

  LaplaceDistribution ld(0.0, 1.0);
  EXPECT_NEAR(ld.Pdf(0.0), 0.5, 1e-9);
  EXPECT_NEAR(ld.Cdf(0.0), 0.5, 1e-9);

  EXPECT_NEAR(ld.TheoreticalMean(), 0.0, 1e-9);
  EXPECT_NEAR(ld.TheoreticalVariance(), 2.0, 1e-9);
}

TEST(DistributionExperimentTest, BinomialEmpiricalMean) {
  using namespace ptm;

  std::mt19937 rng(777);
  auto dist = std::make_shared<BinomialDistribution>(20, 0.3);
  DistributionExperiment experiment(dist, 50000);

  auto stats = experiment.Run(rng);
  EXPECT_NEAR(stats.empirical_mean, dist->TheoreticalMean(), 0.2);
  EXPECT_NEAR(stats.empirical_variance, dist->TheoreticalVariance(), 0.5);
}

using namespace ptm;

static std::mt19937 g_rng(42);

TEST(DistributionTest, BernoulliBasic) {
  BernoulliDistribution dist(0.4);
  EXPECT_DOUBLE_EQ(dist.TheoreticalMean(), 0.4);
  EXPECT_DOUBLE_EQ(dist.TheoreticalVariance(), 0.4 * 0.6);
  EXPECT_DOUBLE_EQ(dist.Pdf(1.0), 0.4);
  EXPECT_DOUBLE_EQ(dist.Pdf(0.0), 0.6);
  EXPECT_DOUBLE_EQ(dist.Cdf(0.5), 0.6);
}

TEST(DistributionTest, BinomialBasic) {
  BinomialDistribution dist(10, 0.5);
  EXPECT_DOUBLE_EQ(dist.TheoreticalMean(), 5.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalVariance(), 2.5);

  EXPECT_NEAR(dist.Pdf(0), std::pow(0.5, 10), 1e-7);
  EXPECT_DOUBLE_EQ(dist.Cdf(-1), 0.0);
  EXPECT_DOUBLE_EQ(dist.Cdf(11), 1.0);
}

TEST(DistributionTest, GeometricBasic) {
  GeometricDistribution dist(0.5);
  EXPECT_DOUBLE_EQ(dist.TheoreticalMean(), 2.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalVariance(), 2.0);

  EXPECT_DOUBLE_EQ(dist.Pdf(1), 0.5);
  EXPECT_DOUBLE_EQ(dist.Pdf(2), 0.25);
  EXPECT_DOUBLE_EQ(dist.Cdf(2), 0.75);
}

TEST(DistributionTest, NormalBasic) {
  NormalDistribution dist(10.0, 2.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalMean(), 10.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalVariance(), 4.0);

  double pi = 3.14159265358979323846;
  EXPECT_NEAR(dist.Pdf(10.0), 1.0 / (2.0 * std::sqrt(2.0 * pi)), 1e-7);
  EXPECT_NEAR(dist.Cdf(10.0), 0.5, 1e-7);
}

TEST(DistributionTest, UniformBasic) {
  UniformDistribution dist(0.0, 10.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalMean(), 5.0);
  EXPECT_NEAR(dist.TheoreticalVariance(), 100.0 / 12.0, 1e-7);
  EXPECT_DOUBLE_EQ(dist.Pdf(5.0), 0.1);
  EXPECT_DOUBLE_EQ(dist.Pdf(15.0), 0.0);
  EXPECT_DOUBLE_EQ(dist.Cdf(2.0), 0.2);
}

TEST(DistributionTest, ExponentialBasic) {
  ExponentialDistribution dist(2.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalMean(), 0.5);
  EXPECT_DOUBLE_EQ(dist.TheoreticalVariance(), 0.25);
  EXPECT_NEAR(dist.Pdf(0.0), 2.0, 1e-7);
  EXPECT_NEAR(dist.Cdf(0.5), 1.0 - std::exp(-1.0), 1e-7);
}

TEST(DistributionTest, LaplaceBasic) {
  LaplaceDistribution dist(0.0, 1.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalMean(), 0.0);
  EXPECT_DOUBLE_EQ(dist.TheoreticalVariance(), 2.0);
  EXPECT_DOUBLE_EQ(dist.Pdf(0.0), 0.5);
  EXPECT_DOUBLE_EQ(dist.Cdf(0.0), 0.5);
}

TEST(DistributionTest, CauchyMoments) {
  CauchyDistribution dist(0.0, 1.0);
  EXPECT_TRUE(std::isnan(dist.TheoreticalMean()));
  EXPECT_TRUE(std::isnan(dist.TheoreticalVariance()));
  EXPECT_NEAR(dist.Cdf(0.0), 0.5, 1e-7);
}

TEST(DistributionExperimentTest, RunAndStats) {
  auto dist = std::make_shared<NormalDistribution>(0.0, 1.0);
  size_t N = 10000;
  DistributionExperiment exp(dist, N);

  ExperimentStats stats = exp.Run(g_rng);

  EXPECT_NEAR(stats.empirical_mean, 0.0, 0.05);
  EXPECT_NEAR(stats.empirical_variance, 1.0, 0.05);
  EXPECT_GT(stats.mean_error, 0.0);
}

TEST(DistributionExperimentTest, KolmogorovAndEcdf) {
  auto dist = std::make_shared<UniformDistribution>(0.0, 1.0);
  size_t N = 1000;
  DistributionExperiment exp(dist, N);

  std::vector<double> grid = {0.1, 0.5, 0.9};
  auto ecdf = exp.EmpiricalCdf(grid, g_rng, N);

  ASSERT_EQ(ecdf.size(), grid.size());
  EXPECT_NEAR(ecdf[1], 0.5, 0.1);

  double ks_dist = exp.KolmogorovDistance(grid, ecdf);
  EXPECT_LT(ks_dist, 0.1);
  EXPECT_GE(ks_dist, 0.0);
}

TEST(DistributionExperimentTest, CauchySimulation) {
  auto dist = std::make_shared<CauchyDistribution>(0.0, 1.0);
  DistributionExperiment exp(dist, 1000);

  ExperimentStats stats = exp.Run(g_rng);
  EXPECT_TRUE(std::isnan(dist->TheoreticalMean()));
  EXPECT_FALSE(std::isnan(stats.empirical_mean));
}
