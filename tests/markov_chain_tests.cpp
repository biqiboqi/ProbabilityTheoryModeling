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

  std::ifstream in("../tests/war_and_peace.txt");
  ASSERT_TRUE(in.good()) << "Не удалось открыть файл ../tests/war_and_peace.txt";

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

TEST(MarkovTextModelTest, WordLevelWithPunctuationTraining) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  std::string text = "Hello, world! Hello, there!";
  model.TrainFromText(text);

  auto& chain = model.Chain();

  EXPECT_GT(chain.TransitionProbability("Hello", ","), 0.0) << "Пунктуация не токенизирована отдельно";
  EXPECT_GT(chain.TransitionProbability(",", "world"), 0.0) << "Переход от запятой не найден";
  EXPECT_GT(chain.TransitionProbability("world", "!"), 0.0) << "Переход к восклицательному знаку не найден";
}

TEST(MarkovTextModelTest, WordLevelWithPunctuationGeneration) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  std::string text = "Hello, world! How are you?";
  model.TrainFromText(text);

  std::mt19937 rng(456);

  std::string generated = model.GenerateText(10, rng, "Hello");
  EXPECT_FALSE(generated.empty()) << "Сгенерированный текст пустой";

  EXPECT_EQ(generated.substr(0, 5), "Hello") << "Текст не начинается с 'Hello'";
  EXPECT_TRUE(generated.find(',') != std::string::npos || generated.find('!') != std::string::npos ||
              generated.find('?') != std::string::npos) << "Пунктуация не найдена в сгенерированном тексте";
}

TEST(MarkovTextModelTest, CharacterLevelWithPunctuation) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Character);
  std::string text = "a!b?a.";
  model.TrainFromText(text);

  std::mt19937 rng(789);

  std::string generated = model.GenerateText(5, rng, "a");
  EXPECT_EQ(generated.size(), 5u) << "Размер сгенерированного текста не равен 5";
  EXPECT_TRUE(generated.find('!') != std::string::npos || generated.find('?') != std::string::npos ||
              generated.find('.') != std::string::npos) << "Пунктуация не найдена в сгенерированном тексте";
}

TEST(MarkovTextModelTest, EmptyStartTokenDefaultsToFirst) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText("some text here");
  std::mt19937 rng(123);
  std::string generated = model.GenerateText(3, rng, "");
  EXPECT_FALSE(generated.empty()) << "Сгенерированный текст пустой";
  EXPECT_EQ(generated.substr(0, 4), "some") << "Текст не начинается с первого токена";
}

TEST(MarkovTextModelTest, UnknownStartTokenDefaultsToFirst) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText("known text");
  std::mt19937 rng(123);
  std::string generated = model.GenerateText(2, rng, "unknown");
  EXPECT_FALSE(generated.empty()) << "Сгенерированный текст пустой";
  EXPECT_EQ(generated.substr(0, 5), "known") << "Текст не начинается с первого известного токена";
}

TEST(MarkovTextModelTest, SingleTokenGeneration) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText("word");
  std::mt19937 rng(123);
  std::string generated = model.GenerateText(1, rng, "word");
  EXPECT_EQ(generated, "word") << "Сгенерированный текст не равен 'word'";
}

TEST(MarkovTextModelTest, ZeroTokensGeneration) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  std::mt19937 rng(123);
  std::string generated = model.GenerateText(0, rng, "start");
  EXPECT_EQ(generated, "") << "Сгенерированный текст не пустой";
}

TEST(MarkovTextModelTest, EmptyTextTraining) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText("");
  auto states = model.Chain().States();
  EXPECT_TRUE(states.empty()) << "Состояния не пустые после тренировки на пустом тексте";
  std::mt19937 rng(123);
  std::string generated = model.GenerateText(1, rng);
  EXPECT_TRUE(generated.empty()) << "Сгенерированный текст не пустой для пустой модели";
}

TEST(MarkovTextModelTest, PunctuationOnlyText) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText("!!! ???");
  auto states = model.Chain().States();
  EXPECT_TRUE(std::ranges::find(states, "!") != states.end()) << "Восклицательный знак не найден в состояниях";
  EXPECT_TRUE(std::ranges::find(states, "?") != states.end()) << "Вопросительный знак не найден в состояниях";
  std::mt19937 rng(123);
  std::string generated = model.GenerateText(3, rng, "!");
  EXPECT_FALSE(generated.empty()) << "Генерация не удалась для текста с только пунктуацией";
}

TEST(MarkovTextModelTest, ApostropheHandling) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  std::string text = "Don't worry, it's okay.";
  model.TrainFromText(text);

  auto& chain = model.Chain();
  EXPECT_GT(chain.TransitionProbability("Don't", "worry"), 0.0) << "'Don't' не токенизирован как единое слово";
  EXPECT_GT(chain.TransitionProbability("worry", ","), 0.0) << "Запятая не отделена";
}

TEST(MarkovTextModelTest, GenerationStopsAtDeadEnd) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText("start end");
  std::mt19937 rng(123);
  std::string generated = model.GenerateText(5, rng, "start");
  EXPECT_EQ(generated, "start end") << "Генерация не остановилась на мертвом конце";
}

TEST(MarkovTextModelTest, LargeTextTraining) {
  using namespace ptm;

  std::string text;
  for (int i = 0; i < 100; ++i) {
    text += "word1 word2, word3! ";
  }
  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText(text);

  auto states = model.Chain().States();
  EXPECT_TRUE(std::ranges::find(states, "word1") != states.end()) << "'word1' не найден в состояниях";
  EXPECT_TRUE(std::ranges::find(states, ",") != states.end()) << "Запятая не найдена в состояниях";
  EXPECT_TRUE(std::ranges::find(states, "!") != states.end()) << "Восклицательный знак не найден в состояниях";

  std::mt19937 rng(123);
  std::string generated = model.GenerateText(10, rng, "word1");
  EXPECT_FALSE(generated.empty()) << "Генерация не удалась для большого текста";
}