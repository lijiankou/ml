// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/rbm/ais.h"
#include "ml/rbm/crbm.h"
#include "ml/rbm/tbm.h"
#include "ml/rbm/rbm.h"
#include "ml/rbm/rbm_gaussian.h"
#include "ml/rbm/rbm_bin.h"
#include "ml/rbm/rbm_util.h"
#include "ml/rbm/repsoftmax.h"
#include "ml/rbm/softmax_crbm.h"
#include "ml/util.h"
#include "ml/roc.h"
#include "ml/rbm/bm_matrix.h"
#include "ml/document.h"

DEFINE_double(eta, 0.1, "learning rate");
DEFINE_double(beta_beg, 0.5, "beta");
DEFINE_int32(ais_run, 2000, "ais sample time");
DEFINE_double(tbm_lambda, 0.1, "the regularizer para");
DEFINE_double(bmmatrix_lambda, 0.1, "the regularizer para");

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
void App3() {
  if (FLAGS_type != "softmax") {
    return;
  }
  Corpus corpus;
  corpus.LoadData(FLAGS_train_path);
  // corpus.RandomOrder();
  /*
  RepSoftMax rep;
  rep.Init(FLAGS_k, corpus.num_terms, FLAGS_bach_size, 1, FLAGS_eta);
  if (FLAGS_algorithm_type == 1) {
    RBMLearning(corpus, FLAGS_it_num, &rep);
  } else {
    RBMLearning2(corpus, FLAGS_it_num, &rep);
  }
  */
  RepSoftMax rep;
  // rep.Init(FLAGS_k, corpus.num_terms, FLAGS_bach_size, 1, FLAGS_eta);
  int size_v = 2;
  // InitRep(FLAGS_k, size_v, 0.1, &rep);
  InitRep(FLAGS_k, size_v, 0.01, &rep);
  VReal beta;
  ::Range(0, 1, FLAGS_beta_beg, &beta);
  double l = Likelihood(corpus.docs[0], FLAGS_ais_run, beta, rep);
  RepSoftMax tmp;
  Multiply(rep, beta[1], &tmp);
  // double p = LogPartition(corpus.TLen(0), corpus.ULen(0), tmp);
  double beta_a = 1;
  double p = LogMultiPartition(corpus.TLen(0), corpus.ULen(0), beta_a, rep);
  LOG(INFO) << p << " real:" << exp(p);
}

void AppRBM(const Str &train_path, const Str &test_path) {
  SpMat u_v;
  ReadData(FLAGS_train_path, 0, 0, &u_v);
  SpMat test_u_v;
  ReadData(FLAGS_test_path, u_v.rows(), u_v.cols(), &test_u_v);
  // SpMat test_v_u = test_u_v.transpose();
  RBM rbm(u_v.rows(), FLAGS_hidden, FLAGS_k);
  rbm.Train(u_v, test_u_v, 2000, FLAGS_eta, FLAGS_bach_size);
}

void CreateTestDic(const SpMat &mat, SInt* dic) {
  for (int i = 1; i < mat.cols(); i++) {
    int num = 0;
    for (SpVec::InnerIterator it(mat.col(i)); it; ++it){
      num++;
    }
    if (num == 0) {
      dic->insert(i);
    }
  }
}

void GaussianRBM(const Str &train_path, const Str &test_path) {
  if (FLAGS_type != "gaussian_rbm") {
    return;
  }
  SpMat u_v;
  ReadData(train_path, 0, 0, &u_v);
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);
  SpMat test_v_u = test_u_v.transpose();
  //read data over

  //RBMGaussian rbm(v_u.rows(), FLAGS_hidden);
  //rbm.Train(v_u, test_v_u, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);
  RBMGaussian rbm(u_v.rows(), FLAGS_hidden);
  //rbm.Train(u_v, test_u_v, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);
  rbm.Train2(u_v, test_u_v, FLAGS_it_num, FLAGS_eta);
}

void AppRBMTranspose(const Str &train_path, const Str &test_path) {
  SpMat u_v;
  ReadData(train_path, 0, 0, &u_v);
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);
  SpMat test_v_u = test_u_v.transpose();
  //read data over

  RBM rbm(v_u.rows(), FLAGS_hidden, FLAGS_k);
  rbm.Train(v_u, test_v_u, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);

  SInt test_dic;
  CreateTestDic(v_u, &test_dic);
  VVReal h;
  rbm.ExpectH(v_u, test_dic, &h);
  WriteStrToFile(Join(h, " ", "\n"), "../libsvm/data/hidden_feature");

  rbm.SaveModel(FLAGS_path_w0, FLAGS_path_w, FLAGS_path_bv, FLAGS_path_bh);
  rbm.Visual();
 
  if (FLAGS_svm_flag) {
    rbm.TestROC(FLAGS_svm_path, "roc_svr", v_u, test_v_u);
    rbm.ROC(FLAGS_svm_path, "roc_svm2", v_u, test_v_u);
    LOG(INFO) << rbm.Predict(FLAGS_svm_path, v_u, test_v_u);
  }
}

void App2() {
  if (FLAGS_type != "rating_rbm") {
    return;
  }
  // AppRBM(FLAGS_train_path, FLAGS_test_path);
  AppRBMTranspose(FLAGS_train_path, FLAGS_test_path);
}

void SoftmaxCRBMApp(const Str &train_path, const Str &test_path,
                                           const Str &feature_path) {
  if (FLAGS_type != "softmax_crbm") {
    return;
  }
  SpMat v_u;
  ReadData(train_path, 0, 0, &v_u);
  LOG(INFO) << "load train over";
  if (!CheckData(v_u, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat u_v = v_u.transpose();
  SpMat test_v_u;
  LOG(INFO) << FLAGS_test_path;
  ReadData(test_path, v_u.rows(), v_u.cols(), &test_v_u);
  LOG(INFO) << "load test over";
  SpMat test_u_v = test_v_u.transpose();
  //read data over

  SpMat feature;
  ReadData(feature_path, 0, 0, &feature);
 
  SoftmaxCRBM crbm(u_v.rows(), FLAGS_hidden, FLAGS_k, FLAGS_feature_size);
  crbm.Train(u_v, feature, test_u_v, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);
}

void TBMApp(const Str &train_path, const Str &test_path, const Str &feature_path) {
  if (FLAGS_type != "tbm") {
    return;
  }
  SpMat u_v;
  ReadData(train_path, 1683, 944, &u_v);
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }

  SpMat test_u_v;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);

  SpMat feature;
  ReadData(feature_path, 0, 0, &feature);
 
  TBM tbm(feature.rows(), FLAGS_tbm_lambda);
  tbm.LoadColdMovie(test_u_v);
  tbm.Train(u_v, feature, test_u_v, FLAGS_it_num, FLAGS_eta);
}

/*
void BinRBM(const Str &train_path, const Str &test_path) {
  if (FLAGS_type != "bin_rbm") {
    return;
  }
  SpMat u_v;
  ReadData(train_path, 0, 0, &u_v);
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);
  SpMat test_v_u = test_u_v.transpose();
  //read data over

  RBMBin rbm(v_u.rows(), FLAGS_hidden);
  //rbm.Train(v_u, test_v_u, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);
  rbm.Recommend(v_u, test_v_u, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);
}
*/

void BinRBM(const Str &train_path, const Str &test_path) {
  if (FLAGS_type != "bin_rbm") {
    return;
  }
  SpMat u_v;
  ReadData(train_path, 0, 0, &u_v);
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);
  SpMat test_v_u = test_u_v.transpose();
  SpMat recommend;
  Str path = "../../data/ali/ali_real_test";
  ReadData(path, u_v.rows(), u_v.cols(), &recommend);
  SpMat recommend2 = recommend.transpose();
  RBMBin rbm(u_v.rows(), FLAGS_hidden);
  rbm.Recommend(u_v, test_u_v, recommend, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);
}

void CRBMApp(const Str &train_path, const Str &test_path, const Str &f_path) {
  if (FLAGS_type != "crbm") {
    return;
  }
  SpMat u_v;
  ReadData(train_path, 0, 0, &u_v);
  LOG(INFO) << "load train over";
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  LOG(INFO) << FLAGS_test_path;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);
  LOG(INFO) << "load test over";
  SpMat test_v_u = test_u_v.transpose();
  //read data over

  SpMat feature;
  ReadData(f_path, 0, 0, &feature);
 
  CRBM crbm(FLAGS_feature_size, v_u.rows(), FLAGS_hidden);
  crbm.LoadColdMovie(test_v_u);
  //crbm.PreTrain(v_u, feature, test_v_u, it_num, FLAGS_eta, FLAGS_bach_size);
  //crbm.Visual();
  //crbm.VisualW();
  //crbm.InitFeature();
  crbm.Train(v_u, feature, test_v_u, FLAGS_it_num, FLAGS_eta, FLAGS_bach_size);
  crbm.Visual();
  //crbm.VisualW();
}

void BMMatrix(const Str &train_path, const Str &test_path) {
  if (FLAGS_type != "bmmatrix") {
    return;
  }
  SpMat u_v;
  ReadData(train_path, 0, 0, &u_v);
  LOG(INFO) << "load train over";
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat test_u_v;
  LOG(INFO) << FLAGS_test_path;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);
  LOG(INFO) << "load test over";
  SpMat test_v_u = test_u_v.transpose();
  //read data over

  BM bm(u_v.rows(), FLAGS_hidden, FLAGS_bmmatrix_lambda);
  bm.Train(u_v, test_u_v, FLAGS_it_num, FLAGS_eta);
  //crbm.VisualW();
}
} // namespace ml

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  ml::App3();
  ml::App2();
  ml::GaussianRBM(FLAGS_train_path, FLAGS_test_path);
  ml::BinRBM(FLAGS_train_path, FLAGS_test_path);
  ml::CRBMApp(FLAGS_train_path, FLAGS_test_path, FLAGS_feature_path);
  ml::SoftmaxCRBMApp(FLAGS_train_path, FLAGS_test_path, FLAGS_feature_path);
  ml::TBMApp(FLAGS_train_path, FLAGS_test_path, FLAGS_feature_path);
  ml::BMMatrix(FLAGS_train_path, FLAGS_test_path);
  return  0;
}
