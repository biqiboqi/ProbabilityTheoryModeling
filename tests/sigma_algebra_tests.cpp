#include <sstream>

#include <gtest/gtest.h>
#include "lib/sigma-algebra/DiscreteRandomVariable.hpp"
#include "lib/sigma-algebra/Event.hpp"
#include "lib/sigma-algebra/OutcomeSpace.hpp"
#include "lib/sigma-algebra/ProbabilityMeasure.hpp"
#include "lib/sigma-algebra/SigmaAlgebra.hpp"

TEST(SigmaAlgebraTest, ProbabilityMeasureAndExpectation) {
  using namespace ptm;

  OutcomeSpace omega;
  auto w0 = omega.AddOutcome("1");
  auto w1 = omega.AddOutcome("2");
  auto w2 = omega.AddOutcome("3");

  ProbabilityMeasure P(omega);
  P.SetAtomicProbability(w0, 0.2);
  P.SetAtomicProbability(w1, 0.3);
  P.SetAtomicProbability(w2, 0.5);

  EXPECT_TRUE(P.IsValid(1e-9));

  // событие A = {1,3}
  std::vector<bool> mask(omega.GetSize(), false);
  mask[w0] = true;
  mask[w2] = true;
  Event A(mask);

  double pA = P.Probability(A);
  EXPECT_NEAR(pA, 0.7, 1e-9);

  // X(1)=1, X(2)=2, X(3)=3
  std::vector<double> X_values = {1.0, 2.0, 3.0};
  DiscreteRandomVariable X(omega, P, X_values);

  double EX = X.ExpectedValue();
  // E[X] = 1*0.2 + 2*0.3 + 3*0.5 = 2.3
  EXPECT_NEAR(EX, 2.3, 1e-9);
}

TEST(SigmaAlgebraTest, EventOperations) {
  using namespace ptm;

  OutcomeSpace omega;
  auto a = omega.AddOutcome("a");
  auto b = omega.AddOutcome("b");
  auto c = omega.AddOutcome("c");

  auto E1 = ptm::Event::Empty(omega.GetSize());
  auto E2 = ptm::Event::Full(omega.GetSize());

  EXPECT_FALSE(E1.Contains(a));
  EXPECT_TRUE(E2.Contains(a));
  EXPECT_TRUE(E2.Contains(b));
  EXPECT_TRUE(E2.Contains(c));

  auto E3 = Event::Complement(E2);
  EXPECT_FALSE(E3.Contains(a));
  EXPECT_FALSE(E3.Contains(b));
  EXPECT_FALSE(E3.Contains(c));
}

TEST(SigmaAlgebraTest, BasicAccessorsAndTrivialCase) {
  using namespace ptm;

  OutcomeSpace omega;
  omega.AddOutcome("heads");
  omega.AddOutcome("tails");
  size_t n = omega.GetSize();

  std::vector<Event> events = {Event::Empty(n), Event::Full(n)};

  SigmaAlgebra sa(omega, events);

  EXPECT_EQ(&sa.GetOutcomeSpace(), &omega);
  EXPECT_EQ(sa.GetEvents().size(), 2);

  EXPECT_TRUE(sa.IsSigmaAlgebra());
}

TEST(SigmaAlgebraTest, InvalidSigmaAlgebra) {
  using namespace ptm;

  OutcomeSpace omega;
  omega.AddOutcome("1");
  omega.AddOutcome("2");
  omega.AddOutcome("3");
  size_t n = omega.GetSize();

  Event onlyA({true, false, false});
  SigmaAlgebra sa1(omega, {onlyA});
  EXPECT_FALSE(sa1.IsSigmaAlgebra());

  SigmaAlgebra sa2(omega, {Event::Empty(n), onlyA, Event::Full(n)});
  EXPECT_FALSE(sa2.IsSigmaAlgebra());

  Event onlyB({false, true, false});
  Event notA({false, true, true});
  Event notB({true, false, true});
  SigmaAlgebra sa3(omega, {Event::Empty(n), onlyA, onlyB, notA, notB, Event::Full(n)});
  EXPECT_FALSE(sa3.IsSigmaAlgebra());
}

TEST(SigmaAlgebraTest, GenerateFromSingleEvent) {
  using namespace ptm;

  OutcomeSpace omega;
  omega.AddOutcome("A");
  omega.AddOutcome("B");
  size_t n = omega.GetSize();

  Event eventA({true, false});
  SigmaAlgebra generated = SigmaAlgebra::Generate(omega, {eventA});

  const auto& events = generated.GetEvents();
  EXPECT_EQ(events.size(), 4);
  EXPECT_TRUE(generated.IsSigmaAlgebra());

  bool foundComplement = false;
  for (const auto& e : events) {
    if (!e.Contains(0) && e.Contains(1))
      foundComplement = true;
  }
  EXPECT_TRUE(foundComplement);
}

TEST(SigmaAlgebraTest, GenerateComplex) {
  using namespace ptm;

  OutcomeSpace omega;
  omega.AddOutcome("1");
  omega.AddOutcome("2");
  omega.AddOutcome("3");
  omega.AddOutcome("4");
  size_t n = omega.GetSize();

  Event g1({true, true, false, false});
  Event g2({false, true, true, false});

  SigmaAlgebra sa = SigmaAlgebra::Generate(omega, {g1, g2});

  EXPECT_TRUE(sa.IsSigmaAlgebra());
  EXPECT_EQ(sa.GetEvents().size(), 16);
}

TEST(SigmaAlgebraTest, GenerateFromEmpty) {
  using namespace ptm;

  OutcomeSpace omega;
  omega.AddOutcome("1");
  omega.AddOutcome("2");

  SigmaAlgebra sa = SigmaAlgebra::Generate(omega, {});

  EXPECT_EQ(sa.GetEvents().size(), 2);
  EXPECT_TRUE(sa.IsSigmaAlgebra());
}
