// copyright 2014 lijiankou. all rights reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/eigen_util.h"
#include "ml/matrix_factorization.h"
#include "ml/util.h"
#include "ml/kmean.h"
#include "gtest/gtest.h"

namespace ml {
TEST(Eigen, DemoTest) {
  TripleVec vec;
  vec.push_back(Triple(5, 0, 1));
  vec.push_back(Triple(4, 0, 2));
  vec.push_back(Triple(2, 1, 3));
  vec.push_back(Triple(3, 2, 4));
  vec.push_back(Triple(2, 1, 4));
  EMat m3(2, 3);
  SpMat m(6, 3);
  m.setFromTriplets(vec.begin(), vec.end());
  EVec v(2);
  v[1] = 1;
  v[0] = 1;
  m3.col(1) = v; 
  m3.col(0) = v; 
  m3.col(2) = v; 
  LOG(INFO) << m;
  LOG(INFO) << m3;
  LOG(INFO) << m.size();
  LOG(INFO) << m.innerSize();
  LOG(INFO) << m.rows();
}

TEST(Eigen, StrTest) {
  std::string str = "1 2 3 4 ";
  VReal v1;
  StrToVReal(str, &v1);
  EVec m(v1.size());
  ToEVec(v1, &m);
  str = "1 2 3 4 \n1 2 3 4 \n";
  VVReal v2;
  StrToVVReal(str, &v2);
  EMat m2(v2.size(), v2[0].size());
  ToEMat(v2, &m2);
  EXPECT_EQ(EMatToStr(m2), str);
  EXPECT_EQ(EVecToStr(m), "1 2 3 4 ");
}

TEST(Eigen, ReadDataTest) {
  Str path("rbm/tmp/train_g20.txt");
  TripleVec vec;
  ReadData(path, &vec);
  EXPECT_EQ(79951, vec.size());
  std::pair<int, int> p = ::Max(vec);
  EXPECT_EQ(1682, p.first);
  EXPECT_EQ(943, p.second);
  SpMat mat;
  ReadData(path, &mat);
  EXPECT_EQ(943, mat.cols());
}

TEST(Eigen, NormalRandomTest) {
  EMat t(2, 3);
  NormalRandom(&t);
  LOG(INFO) << t.transpose();
}

//show results by different parameters
//eta:0.0005
TEST(MatrixFactorization, GradientDescent) {
  Str path1("../data/movielen_train.txt");
  Str path2("../data/movielen_test.txt");
  SpMat train;
  SpMat test;
  std::pair<int, int> p = ReadData(path1, &train);
  ReadData(path2, &test);
  double eta = 0.0005;
  int it_num = 100000;
  double lambda = 0.12;
  int k = 20;
  EMat v(k, p.first);
  EMat u(k, p.second);
  v.setRandom();
  u.setRandom();
  // BGD(it_num, eta, lambda, train, test, &u, &v);
  //SGD(it_num, eta, lambda, train, test, &u, &v);
}

TEST(MatrixFactorization, GradientDescent2) {
  Str path1("../data/movielen_train.txt");
  Str path2("../data/movielen_test.txt");
  SpMat train;
  SpMat test;
  std::pair<int, int> p = ReadData(path1, &train);
  ReadData(path2, &test);
  double eta = 0.0005;
  int it_num = 100000;
  double lambda = 0.12;
  int k = 20;
  MF mf;
  RandomInit(p.second, p.first, k, &mf);
  LOG(INFO) << k;
  SGD(it_num, eta, lambda, train, test, &mf);
}

TEST(Eigen, SpMat) {
  SpVec vec;
  vec.insert(1) = 1;
  vec.insert(2) = 2;
  SpMat m;
  m.resize(3, 3);
  m.col(1) = vec;
  m.col(2) = vec;
  LOG(INFO) << m.transpose();
}

TEST(KMean, KMean) {
  EMat a(2, 4); 
  a(0, 0) = 1;
  a(1, 0) = 1;
  a(0, 1) = 1.1;
  a(1, 1) = 1;
  a(0, 2) = 10;
  a(1, 2) = 10;
  a(0, 3) = 10;
  a(1, 3) = 10.1;
  VVInt com;
  int k = 2;
  int iter = 10;
  KMean(a, k, iter, &com);
  EXPECT_EQ("0 1 \n2 3 \n", Join(com, " ", "\n"));
}

TEST(Eigen, Join) {
  EMat m(2, 2);
  m.setZero();
  m(1, 1) = 1;
  m(1, 0) = 0;
  LOG(INFO) << Join(m);
}

TEST(Eigen, CreateAdjTest) {
  SpMat u(3, 4);
  TripleVec vec;
  vec.push_back(Triple(0, 1, 3));
  vec.push_back(Triple(1, 2, 2));
  vec.push_back(Triple(2, 3, 0));
  vec.push_back(Triple(1, 1, 0));
  u.setFromTriplets(vec.begin(), vec.end());
  EMat m(u.rows(), u.cols());
  CreateAdj(u, &m);
  LOG(INFO) << m;
  EMat uu(u);
  LOG(INFO) << uu;
}

TEST(Eigen, SampleTest) {
  EMat u(3, 3);
  u.setRandom();
  LOG(INFO) << u;
  ::Sample(&u);
  LOG(INFO) << u;
}

TEST(Eigen, BaseTest) {
  Vec v1(3);
  v1[0] = 1;
  v1[1] = 2;
  v1[2] = 0.1;
  LOG(INFO) << v1;
  LOG(INFO) << v1.sum();
  LOG(INFO) << v1.maxCoeff();
  //LOG(INFO) << m[0];
}

TEST(EigenUtil, LogSumTest) {
  Vec v1(3);
  v1.setZero();
  LOG(INFO) << v1;
  v1[0] = 2;
  v1[1] = 3;
  v1[2] = 4;
  EXPECT_DOUBLE_EQ(log(exp(2.0) + exp(3.0) + exp(4.0)), LogSum(v1));
  Mat m(3, 3);
  LOG(INFO) << m;
  m.col(0) = v1;
  m.col(1) = v1;
  m.col(2) = v1;
  LOG(INFO) << m;
}
} // namespace ml 

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
