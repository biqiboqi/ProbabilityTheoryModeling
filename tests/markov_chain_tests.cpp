#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>

#include "lib/markov-chain/MarkovChain.hpp"
#include "lib/markov-chain/MarkovTextModel.hpp"

TEST(MarkovChainTest, SimpleCountsAndProbabilities) {
  using namespace ptm;

  MarkovChain chain;
  std::vector<std::string> seq = {"A", "B", "A", "B", "A"};
  chain.Train(seq);

  EXPECT_NEAR(chain.TransitionProbability("A", "B"), 1.0, 1e-9);
  EXPECT_NEAR(chain.TransitionProbability("B", "A"), 1.0, 1e-9);
  EXPECT_NEAR(chain.TransitionProbability("A", "A"), 0.0, 1e-9);
}

TEST(MarkovChainTest, IncrementalTraining) {
  using namespace ptm;

  MarkovChain chain;
  chain.Train({"A", "B"});
  double p_ab1 = chain.TransitionProbability("A", "B");

  chain.Train({"A", "C"});
  double p_ab2 = chain.TransitionProbability("A", "B");
  double p_ac2 = chain.TransitionProbability("A", "C");

  EXPECT_NEAR(p_ab2, 0.5, 1e-9);
  EXPECT_NEAR(p_ac2, 0.5, 1e-9);
  EXPECT_NEAR(p_ab1, 1.0, 1e-9);
}

TEST(MarkovTextModelTest, WordLevelGeneration) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  std::string text = "hello world hello world hello";
  model.TrainFromText(text);

  std::mt19937 rng(123);

  auto& chain = model.Chain();
  double p = chain.TransitionProbability("hello", "world");
  EXPECT_NEAR(p, 1.0, 1e-9);

  std::string generated = model.GenerateText(5, rng, "hello");
  EXPECT_FALSE(generated.empty());
}

TEST(MarkovTextModelTest, CharacterLevelGeneration) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Character);
  std::string text = "ababa";
  model.TrainFromText(text);

  std::mt19937 rng(321);

  double p_ab = model.Chain().TransitionProbability("a", "b");
  double p_ba = model.Chain().TransitionProbability("b", "a");

  EXPECT_NEAR(p_ab, 1.0, 1e-9);
  EXPECT_NEAR(p_ba, 1.0, 1e-9);

  std::string generated = model.GenerateText(4, rng, "a");
  EXPECT_EQ(generated.size(), 4u);
}

TEST(MarkovTextModelTest, TrainOnWarAndPeaceWordLevel) {
  using namespace ptm;

  std::ifstream in("../../tests/war_and_peace.txt");
  ASSERT_TRUE(in.good()) << "Не удалось открыть файл ../../tests/war_and_peace.txt";

  std::stringstream buffer;
  buffer << in.rdbuf();
  std::string text = buffer.str();
  ASSERT_FALSE(text.empty()) << "Файл войны и мира пустой";

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText(text);

  const auto& chain = model.Chain();
  auto states = chain.States();

  EXPECT_GT(states.size(), 5000u) << "Слишком маленький словарь, похоже, текст обрезан";

  auto has_token = [&](const std::string& token) { return std::ranges::find(states, token) != states.end(); };

  EXPECT_TRUE(has_token("and")) << "Слово \"and\" не найдено в словаре";
  EXPECT_TRUE(has_token("in")) << "Слово \"in\" не найдено в словаре";
  EXPECT_TRUE(has_token("on")) << "Слово \"on\" не найдено в словаре";

  std::mt19937 rng(123);

  std::string generated = model.GenerateText(50, rng, "and");
  EXPECT_FALSE(generated.empty()) << "Сгенерированный текст пустой";

  std::size_t space_count = std::ranges::count(generated, ' ');
  EXPECT_GT(space_count, 5u);
}

// Add your tests...
