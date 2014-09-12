// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/rbm/ais.h"
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"
#include "ml/rbm/repsoftmax.h"
#include "ml/rbm/roc.h"
#include "gtest/gtest.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>

DEFINE_double(eta, 0.1, "learning rate");
DEFINE_double(beta_beg, 0.5, "beta");
DEFINE_int32(ais_run, 2000, "ais sample time");
DEFINE_double(tbm_lambda, 0.1, "the regularizer para");

DEFINE_int32(bach_size, 100, "bach size");
DEFINE_int32(k, 5, "class size");
DEFINE_int32(m, 2000, "visual size");
DEFINE_int32(hidden, 100, "hidden feature size");
DEFINE_int32(it_num, 1000, "iter number");
DEFINE_int32(algorithm_type, 1, "iter number");

DEFINE_int32(feature_size, 1000, "feature number");

DEFINE_string(type, "eigen", "");
DEFINE_string(train_path, "tmp/train1", "");
DEFINE_string(test_path, "tmp/test1", "");
DEFINE_string(feature_path, "tmp/test1", "feature path");

DEFINE_string(path_w0, "./data/rbm_w0", "");
DEFINE_string(path_w, "./data/rbm_w", "");
DEFINE_string(path_bv, "./data/rbm_bv", "");
DEFINE_string(path_bh, "./data/rbm_bh", "");

DEFINE_bool(lr_flag, false, "flag of lr");
DEFINE_bool(svm_flag, false, "flag of svm");
DEFINE_bool(naive_flag, false, "flag of naive bayes");
DEFINE_string(svm_path, "./tmp/fengxing/data/libsvm-svr", "");
DEFINE_string(lr_path, "./tmp/fengxing/data/sigmoid", "");
DEFINE_string(naive_path, "./tmp/fengxing/data/naive", "");

DEFINE_int32(nCD, 1, "step size of CD");
namespace ml {
/*
TEST(Ais, UniformSampleTest) {
  Corpus c;
  Str dat = "../data/document_demo";
  c.LoadData(dat);
  VInt v(c.TermNum());
  UniformSample(c.docs[0], &v);
  LOG(INFO) << Join(v, " ");
}

TEST(Ais, AisTest) {
  Corpus corpus;
  Str dat = "../data/document_demo";
  corpus.LoadData(dat);
  int bach_size = 2;
  double eta = 0.0001;
  int k = 2;
  int it_num = 2000;
  RepSoftMax rep;
  rep.Init(k, corpus.num_terms, bach_size, 1, eta);
  RBMLearning(corpus, it_num, &rep);
  int run = 10;
  VReal beta(10);
  for (size_t i = 0; i < beta.size(); i++) {
    beta[i] = 0.1 * i;
  }
  LOG(INFO) << Likelihood(corpus.docs[0], run, beta, rep);
}

// test Partition = 2^F
TEST(Ais, LogPartitionTest) {
  RepSoftMax rep;
  int size_f = 1;
  int size_v = 2;
  ZeroRep(size_f, size_v, &rep);
  int doc_len = 2;
  int word_num = 2;
  double beta_a = 1;
  EXPECT_DOUBLE_EQ(2, exp(LogMultiPartition(doc_len, word_num, beta_a, rep)));
  size_f = 2;
  ZeroRep(size_f, size_v, &rep);
  EXPECT_DOUBLE_EQ(4, exp(LogMultiPartition(doc_len, word_num, beta_a, rep)));
}

TEST(Ais, WAisTest) {
  Corpus corpus;
  corpus.LoadData("test");
  RepSoftMax rep;
  int f_size = 1;
  int v_size = 2;
  double value = 0;
  InitRep(f_size, v_size, value, &rep);
  VReal beta;
  Range(0, 1, 0.01, &beta);
  int ais_run = 10;
  double wais = WAis(corpus.docs[0], ais_run, beta, rep);
  double z = wais * pow(2, rep.c.size()) * rep.b.size(); 
  double p = LogPartition(corpus.TLen(0), corpus.ULen(0), rep);
  EXPECT_DOUBLE_EQ(z, exp(p));
}
*/

TEST(ROC, AUCTest) {
  VReal r;
  r.push_back(1);
  r.push_back(2);
  r.push_back(1);
  r.push_back(2);
  r.push_back(2);
  VReal p;
  p.push_back(0.1);
  p.push_back(0.3);
  p.push_back(0.4);
  p.push_back(0.5);
  p.push_back(0.46);
  LOG(INFO) << AUC(r, p);
}
} // namespace ml

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS(); 
}
