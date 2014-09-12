// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include <fstream>

#include "base/base_head.h"
#include "gtest/gtest.h"

TEST(StrUtil, SplitStrTEST) {
  Str str("this,is,a,test");
  VStr vec;
  SplitStr(str, ',', &vec);
  EXPECT_EQ(4, vec.size());
  EXPECT_EQ("this", vec[0]);
  EXPECT_EQ("is", vec[1]);
  EXPECT_EQ("a", vec[2]);
  EXPECT_EQ("test", vec[3]);
  vec.clear();
  SplitStr(str, ",", &vec);
  EXPECT_EQ(4, vec.size());
  EXPECT_EQ("this", vec[0]);
  EXPECT_EQ("is", vec[1]);
  EXPECT_EQ("a", vec[2]);
  EXPECT_EQ("test", vec[3]);
}

TEST(StrUtil, TrimTEST) {
  Str str("  this\t");
  Str str2;
  TrimStr(str, " \t", &str2);
  EXPECT_EQ("this", str2);
  EXPECT_EQ("this", TrimStr(str));
}

TEST(StrUtil, StartWithTEST) {
  Str str("this");
  EXPECT_TRUE(StartWith(str, "th"));
  EXPECT_FALSE(StartWith(str, "ax"));
}

TEST(StrUtil, EndWithTEST) {
  Str str("this");
  EXPECT_TRUE(EndWith(str, "is"));
  EXPECT_FALSE(EndWith(str, "  "));
}

TEST(StrUtil, LowerTEST) {
  Str str("tHiS");
  EXPECT_EQ("this", Lower(str));
}

TEST(Join, MapToStr) {
  MIntInt m;
  m[1] = 1;
  m[2] = 2;
  EXPECT_EQ("1 1 \n2 2 \n", MapToStr(m));
}

TEST(Join, JoinTEST) {
  VStr v;
  v.push_back("hello");
  v.push_back("world");
  EXPECT_EQ("hello world ", Join(v, " "));
  VVStr v2;
  v2.push_back(v);
  v2.push_back(v);
  EXPECT_EQ("hello world \nhello world \n", Join(v2, " ", "\n"));
  int len1 = 2;
  int len2 = 3;
  double**a = NewArray(len1, len2);
  Init(len1, len2, 1, a);
  EXPECT_EQ("1 1 1 \n1 1 1 \n", Join(a, len1, len2));
  DelArray(a, len1);
}

TEST(Join, IntTEST) {
  VInt data;
  data.push_back(1);
  data.push_back(2);
  EXPECT_EQ("1 2 ", Join(data, " "));
  VVInt data2;
  data2.push_back(data);
  data2.push_back(data);
  EXPECT_EQ("1 2 \n1 2 \n", Join(data2, " ", "\n"));
}

TEST(Stat, LogSumTest) {
  EXPECT_LT(std::abs(1.69315 - LogSum(1, 1)), 0.0001);
  LOG(INFO) << LogSum(1, 2);
}

TEST(Stat, LogParitionTest) {
  VReal e;
  e.push_back(4);
  e.push_back(6);
  EXPECT_DOUBLE_EQ(log(exp(6) + exp(4)), LogPartition(e));
  VInt v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(1);
  e.clear();
  e.push_back(0);
  e.push_back(0);
  e.push_back(0);
  EXPECT_DOUBLE_EQ(4, exp(LogPartition(v, e)));
}

TEST(Stat, TriGammaTest) {
  EXPECT_LT(std::abs(0.105166 - TriGamma(10)), 0.0001);
}

TEST(Stat, DigGmmaTest) {
  EXPECT_LT(std::abs(2.25175 - DiGamma(10)), 0.0001);
}

TEST(Stat, LoggammaTest) {
  EXPECT_LT(std::abs(-94.0718 - LogGamma(0.1)), 0.0001);
}

TEST(Stat, MaxTest) {
  double a[] = {1, 2, 40.0, 10, -1};
  EXPECT_LT(std::abs(40 - Max(a, 5)), 0.0001);
}

TEST(Base, RangeTest) {
  VInt v;
  Range(1, 5, 1, &v);
  EXPECT_EQ(Join(v, " "), "1 2 3 4 ");
  VReal v2;
  Range(0.1, 0.5, 0.1, &v2);
  EXPECT_EQ(Join(v2, " "), "0.1 0.2 0.3 0.4 ");
}

TEST(Base, RandomTest) {
  VReal tmp;
  tmp.push_back(20);
  tmp.push_back(10);
  tmp.push_back(10);
  int num = 100000;
  MIntInt m;
  for (int i = 0; i < num; i++) {
    m[Random(tmp)]++;
  }
  int num2 = 25000;
  int num3 = 50000;
  EXPECT_LT(std::abs(num3 - m[0]), 100);
  EXPECT_LT(std::abs(num2 - m[1]), 100);
  EXPECT_LT(std::abs(num2 - m[2]), 100);
}

TEST(Base, RandomTest2) {
  int k = 4;
  int num = 100000;
  MIntInt m;
  for (int i = 0; i < num; i++) {
    m[Random(k)]++;
  }
  VReal r(k);
  for (int i = 0; i < k; i++) {
    r[i] = static_cast<double>(m[i]) / num;
  }
  double precision = 0.01;
  double p = 0.25;
  EXPECT_LT(std::abs(p - r[0]), precision);
  EXPECT_LT(std::abs(p - r[1]), precision);
  EXPECT_LT(std::abs(p - r[2]), precision);
  EXPECT_LT(std::abs(p - r[3]), precision);
}

TEST(Join, InitTEST) {
  VReal tmp;
  Init(2, 0.0, &tmp);
  EXPECT_EQ("0 0 ", Join(tmp, " "));
  VVReal tmp2;
  Init(2, 2, 0.0, &tmp2);
  EXPECT_EQ("0 0 \n0 0 \n", Join(tmp2, " ", "\n"));

  VReal tmp3;
  Init(2, 1, &tmp3);
  EXPECT_EQ("1 1 ", Join(tmp3, " "));
  VVInt tmp4;
  Init(2, 2, 1, &tmp4);
  EXPECT_EQ("1 1 \n1 1 \n", Join(tmp4, " ", "\n"));
}

TEST(Random, SigmoidTest) {
  double a = 0;
  MIntInt m;
  for (int i = 0; i < 1000; i++) {
    m[SigmoidSample(a)]++;
  }
  EXPECT_LT(500 - m[0], 50);
}

TEST(Random, Sample1Test) {
  double a = 0;
  MIntInt m;
  for (int i = 0; i < 1000; i++) {
    m[Sample1(a)]++;
  }
  EXPECT_LT(500 - m[0], 50);
}


TEST(Base, SquareTEST) {
  double a = 2.0;
  EXPECT_LT(std::abs(Square(a) - 4), 0.001);
}

TEST(Stat, ProbabilityTEST) {
  VReal v(4, 1.0);
  VReal v2;
  Probability(v, &v2);
  EXPECT_LT(std::abs(v2.at(0) - 0.25), 0.001);
}

TEST(Stat, ExpectTEST) {
  VReal v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  VReal v2;
  Probability(v, &v2);
  EXPECT_LT(std::abs(Expect(v2) - 3), 0.000001);
}

TEST(Base, SumTEST) {
  VReal v;
  v.push_back(1);
  v.push_back(2);
  VVReal v2;
  v2.push_back(v);
  v2.push_back(v);
  VReal v3;
  Sum(v2, &v3);
  EXPECT_EQ(3, v3[0]);
  EXPECT_EQ(3, v3[1]);
}

TEST(StlUtil, ToSetTEST) {
  VInt v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(2);
  v.push_back(3);
  SInt s;
  ToSet(v, &s);
  EXPECT_EQ(3, s.size());
  EXPECT_EQ("1 2 3 ", Join(s, " "));
}

TEST(StlUtil, MaxTEST) {
  VInt v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  EXPECT_EQ(3, Max(v));
}

TEST(StlUtil, DiffNumTEST) {
  VInt v;
  v.push_back(1);
  v.push_back(2);
  VInt v2;
  v2.push_back(1);
  v2.push_back(2);
  EXPECT_EQ(0, DiffNum(v, v2));
  v.push_back(2);
  v2.push_back(1);
  EXPECT_EQ(1, DiffNum(v, v2));
}

TEST(Random, UniformSampleTEST) {
  VInt v(3);
  int len = 3000;
  UniformSample(len, &v);
  EXPECT_LT(std::abs(v[0] - 1000), 100);
  EXPECT_LT(std::abs(v[1] - 1000), 100);
  EXPECT_LT(std::abs(v[2] - 1000), 100);
}

TEST(StlUtil, MultiplyTEST) {
  VReal v(2);
  v[0] = 1;
  v[1] = 2;
  VVReal v2(2, v);
  VReal v3(v.size());
  double m = 2;
  Multiply(v, m, &v);
  EXPECT_DOUBLE_EQ(v[0], 2);
  EXPECT_DOUBLE_EQ(v[1], 4);
  Multiply(v2, m, &v2);
}

TEST(Probability, NextMutiSeqTEST) {
  VInt v(4, 0);
  v[0] = 10;
  while (NextMultiSeq(&v)) {
    EXPECT_EQ(Sum(v), 10);
  }
}

TEST(Probability, NextBinarySeqTEST) {
  VInt v(4, 0);
  int count = 1;
  while (NextBinarySeq(&v)) {
    count++;
  }
  EXPECT_EQ(16, count);
}

TEST(IOUtilTest, FIOTEST) {
  VInt a(2);
  a[0] = 1;
  a[1] = 2;
  Str file("data/test");
  WriteFile(file, a);

  VInt v2(2);
  ReadFile(file, &v2);
  EXPECT_EQ(v2[0], 1);
  EXPECT_EQ(v2[1], 2);
}

TEST(StlUtil, PushTEST) {
  VInt v;
  int a = 1;
  Push(2, a, &v);
  EXPECT_EQ(2, v.size());
  EXPECT_EQ(1, v[0]);
  EXPECT_EQ(1, v[1]);
}

TEST(MathUtilTest, FactorialTEST) {
  EXPECT_DOUBLE_EQ(120, Factorial(5));
  EXPECT_DOUBLE_EQ(720, Factorial(6));
}

TEST(MathUtilTest, MultiNumTEST) {
  VInt v;
  v.push_back(2);
  v.push_back(2);
  EXPECT_DOUBLE_EQ(6, MultiNum(4, v));
  v.push_back(2);
  EXPECT_DOUBLE_EQ(90, MultiNum(6, v));
}

TEST(Stat, QuadraticTest) {
  VReal lhs(2);
  lhs[0] = 1;
  lhs[1] = 2;
  VReal rhs(3);
  rhs[0] = 3;
  rhs[1] = 4;
  rhs[2] = 5;
  VVReal w;
  Init(lhs.size(), rhs.size(), 0, &w);
  w[0][0] = 1;
  w[0][1] = 2;
  w[0][2] = 3;
  w[1][0] = 4;
  w[1][1] = 5;
  w[1][2] = 6;
  EXPECT_DOUBLE_EQ(150, Quadratic(lhs, rhs, w));
}

TEST(Stat, InnerProdTest) {
  VReal lhs(2);
  lhs[0] = 1;
  lhs[1] = 2;
  VReal rhs(2);
  rhs[0] = 3;
  rhs[1] = 4;
  EXPECT_DOUBLE_EQ(11, InnerProd(lhs, rhs));
}

TEST(IOUtil, IsFileTest) {
  EXPECT_TRUE(IsFile("io_util.h"));
  EXPECT_FALSE(IsFile("xx"));
}

TEST(Base, TransTest) {
  VVReal v;
  Init(2, 3, 0, &v);
  v[0][0] = 1;
  v[0][1] = 2;
  v[0][2] = 3;
  v[1][0] = 4;
  v[1][1] = 5;
  v[1][2] = 6;
  VVReal v2;
  Trans(v, &v2);
  EXPECT_EQ("1 4 \n2 5 \n3 6 \n", Join(v2, " ", "\n"));
}
int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
