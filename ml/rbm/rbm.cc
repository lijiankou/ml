// Copyright 2013 zhangwei, lijiankou. All Rights Reserved.
// Author: zhangw@ios.ac.cn  lijk_start@163.com
#include "ml/rbm/rbm.h"

#include "base/base_head.h"
#include "ml/rbm/rbm_util.h"
#include "ml/eigen.h"
#include "ml/kmean.h"
#include "ml/util.h"

DECLARE_string(path_w);
DECLARE_string(path_bv);
DECLARE_string(path_bh);
DECLARE_int32(hidden);
DECLARE_double(lambda_weight);

DECLARE_int32(nCD);

using std::string;
namespace ml {
RBM::RBM(int nv, int nh, int nsoftmax){
   W.resize(nsoftmax);
   dW.resize(nsoftmax);
   for (size_t i = 0; i < W.size(); ++i) {
     W[i].resize(nh, nv);
     dW[i].resize(nh, nv);
     NormalRandom(&W[i]);
   }

   bv.resize(nsoftmax, nv);
   dv.resize(nsoftmax, nv);
   NormalRandom(&bv);

   bh.resize(nh);
   bh.setZero();
   dh.resize(nh);

   v0.resize(nv);
   h0.resize(nh);
   vk.resize(nv);
   hk.resize(nh);
}

void RBM::ExpectV(const EVec &h, const SpVec &t, VVReal* des) {
  for (SpVec::InnerIterator it(t); it; ++it){
    VReal a(W.size());
    for(size_t k = 0; k < W.size(); ++k) {
      a[k] = bv(k, it.index()) +  W[k].col(it.index()).dot(h);
    }
    VReal b(W.size());
    ml::Softmax(a, &b);
    des->push_back(b);
  }
}

void RBM::ExpectRating(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VVReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = ml::ExpectRating(vec[i]);
  }
}

void RBM::SampleV(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VVReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = Sample(vec[i]);
  }
}

void RBM::ExpectH(const SpVec &v, EVec *h) {
  for (int j = 0; j < h->rows(); ++j) {
    double s = 0;
    for (SpVec::InnerIterator it(v); it; ++it) {
      s += W[it.value() - 1](j, it.index());
    }
    s += bh[j];
    (*h)[j] = Sigmoid(s);
  }
}

void RBM::SampleH(const SpVec &v, EVec *h) {
  ExpectH(v, h);
  ::Sample(h);
}

void RBM::PartGrad(const SpVec &v, const EVec &h, double coeff){
  for (SpVec::InnerIterator it(v); it; ++it) {
    dv(it.value() - 1, it.index()) += coeff;
    for (int j = 0; j < h.rows(); ++j) {
      dW[it.value() - 1](j, it.index()) += coeff * h[j];
    }
  }
  for (int j = 0; j < h.rows(); ++j) {
    dh(j) += coeff * h[j];
  }
}

void RBM::InitGradient(){
  for(size_t k = 0; k < dW.size(); ++k) {
    dW[k].setZero();
  }
  dh.setZero();
  dv.setZero();
}

void RBM::UpdateGradient(double alpha, int batch_size) {
  double r = alpha / batch_size;
  for(size_t k = 0; k < dW.size(); ++k) {
    //W[k] += r * dW[k];
    W[k] += r * dW[k] + r * W[k] * FLAGS_lambda_weight;
  }
  bh += r * dh;
  bv += r * dv;
}

void RBM::Gradient(const SpVec &x, int step) {
  SampleH(x, &h0);
  SampleV(h0, x, &vk);
  for (int k = 0; k < step - 1; ++k) {
    SampleH(vk, &hk);
    SampleV(hk, x, &vk);
  }
  ExpectH(vk, &hk);
  // SampleH(vk, &hk);
  PartGrad(x,  h0, 1);
  PartGrad(vk, hk, -1);
}

void RBM::Train(const SpMat &train, const SpMat &test, int niter, double alpha,
                                                       int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = FLAGS_nCD;
  for (int i = 0; i < niter; ++i) {
    Count.clear();
    if (i % 50 == 0) {
      nCD++;
    }
    for (int n = 0; n < train.cols(); n++) {
      Gradient(train.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    LOG(INFO) << i << " " << 
              Predict(train, train) << " " << Predict(train, test);
              // << "  " << PredictWithError(train, test)
              // << " " << PredictWithRandom(train, test);
  }
}

void RBM::ExpectH(const SpMat &train, const SInt &test_dic, VVReal* h) {
  LOG(INFO) << train.cols();
  h->resize(train.cols() - 1);
  for(int n = 1; n < train.cols(); n++) {
    if (test_dic.find(n) != test_dic.end()) {
      continue;
    }
    VReal tmp;
    ExpectH(train.col(n), &h0);
    for (int i = 0; i < h0.size(); i++) {
      tmp.push_back(h0[i]);
    }
    h->at(n - 1).swap(tmp);
  }
}

void ReadH(int h_num, const Str &path, std::vector<EVec>* h) {
  LOG(INFO) << path;
  VReal h2(h_num);
  Str str;
  LOG(INFO) << path;
  ReadFileToStr(path, &str);
  if (str.empty()) {
    LOG(INFO) << "error, maybe the file not exist";
  }
  VStr lines;
  SplitStr(str, "\n", &lines);
  for (size_t i = 0; i < lines.size(); i++) {
    if (TrimStr(lines[i]).empty()) {
      continue;
    }
    VStr l2;
    SplitStr(lines[i], " ", &l2);
    int doc_id = StrToInt(l2[0]);
    (*h)[doc_id].resize(h2.size());
    for (size_t j = 1; j < l2.size(); j++) {
      (*h)[doc_id][j - 1] = StrToReal(l2[j]);
    }
  }
  LOG(INFO) << "read";
}

void RBM::TestROC(const Str &hid_path, const Str &res, const SpMat &train,
                                                       const SpMat &test) {
  std::vector<EVec> h;
  h.resize(train.cols());
  ReadH(FLAGS_hidden, hid_path, &h);
  VVReal predict_pairs;
  for(int n = 0; n < train.cols(); n++) {
    VVReal predict;
    // ExpectRating(h[n], test.col(n), &v0);
    ExpectV(h[n], test.col(n), &predict);
    int i = 0;
    for (SpMat::InnerIterator it(test, n); it; ++it){
      if (predict.size() == 0) {
        break;
      }
      VReal predict_pair;
      predict_pair.push_back(predict[i][1]);
      i++;
      predict_pair.push_back(it.value());
      predict_pairs.push_back(predict_pair);
    }
  }
  WriteStrToFile(Join(predict_pairs, " ", "\n"), res);
}

double RBM::Predict(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    //ExpectH(train.col(n), &h0);
    SampleH(train.col(n), &h0);
    ExpectRating(h0, test.col(n), &v0);
    v0 -= test.col(n);
    rmse += v0.cwiseAbs2().sum();
  }
  return sqrt(rmse/test.nonZeros());
}

double RBM::PredictWithRandom(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    v0.setZero();
    for (SpVec::InnerIterator it(test.col(n)); it; ++it){
      v0.insert(it.index()) = Random1() + 1;
    }
    v0 -= test.col(n);
    rmse += v0.cwiseAbs2().sum();
  }
  return sqrt(rmse/test.nonZeros());
}

double RBM::PredictWithError(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    ExpectH(train.col(n), &h0);
    for (int i = 0; i < h0.size(); i++) {
      h0[i] += NormalSample() * 2;
    }
    // SampleH(train.col(n), &h0);
    ExpectRating(h0, test.col(n), &v0);
    v0 -= test.col(n);
    rmse += v0.cwiseAbs2().sum();
  }
  return sqrt(rmse/test.nonZeros());
}

void RBM::SaveModel(const Str &path0, const Str &path1,
                    const Str &path2, const Str &path3) const {
  WriteStrToFile(EMatToStr(W[0]), path0);
  WriteStrToFile(EMatToStr(W[1]), path1);
  WriteStrToFile(EMatToStr(bv), path2);
  WriteStrToFile(EVecToStr(bh), path3);
}

void RBM::LoadModel(const Str &path0, const Str &path1,
                    const Str &path2, const Str &path3) {
  W.resize(2);

  VVReal tmp_w0;
  StrToVVReal(ReadFileToStr(path0), &tmp_w0);
  W[0].resize(tmp_w0.size(), tmp_w0[0].size());
  ToEMat(tmp_w0, &W[0]);

  VVReal tmp_w;
  StrToVVReal(ReadFileToStr(path1), &tmp_w);
  W[1].resize(tmp_w.size(), tmp_w[0].size());
  ToEMat(tmp_w, &W[1]);

  VVReal tmp_bv;
  StrToVVReal(ReadFileToStr(path2), &tmp_bv);
  bv.resize(tmp_bv.size(), tmp_bv[0].size());
  ToEMat(tmp_bv, &bv);

  VReal tmp_bh;
  StrToVReal(ReadFileToStr(path3), &tmp_bh);
  bh.resize(tmp_bh.size());
  ToEVec(tmp_bh, &bh);
}

//create test from train, all id not in train are in test
void SpMatDic(const SpMat &train, VSInt* dic) {
  dic->resize(train.cols());
  for(int n = 0; n < train.cols(); n++) {
    for (SpMat::InnerIterator it(train, n); it; ++it){
      (*dic)[n].insert(it.index());
    }
  }
}
 
void SpMatDic2(const SpMat &train, VSInt* dic2) {
  VSInt dic;
  SpMatDic(train, &dic);
  dic2->resize(train.cols());
  for (int n = 0; n < train.cols(); n++) {
    for (int i = 0; i < train.rows(); i++) {
      if (dic[n].find(i) == dic[n].end()) {
        (*dic2)[n].insert(i);
      }
    }
  }
}
 
//create positive sample
void PositiveSample(const SpMat &test, VSInt* dic) {
  dic->resize(test.cols());
  for(int n = 0; n < test.cols(); n++) {
    for (SpMat::InnerIterator it(test, n); it; ++it){
      if (it.value() == 2) {
        (*dic)[n].insert(it.index());
      }
    }
  }
}

void CreateGlobal(const SpMat &mat, VVInt* vec) {
  VSInt dic2;
  SpMatDic2(mat, &dic2);
  vec->resize(mat.cols()); 
  for(int n = 0; n < mat.cols(); n++) {
    for (SInt::iterator it = dic2[n].begin(); it != dic2[n].end(); ++it) {
      (*vec)[n].push_back(*it);
    }
  }
}

void RBM::ExpectV(const EVec &h, VReal* des) {
  for (int i = 0; i < bv.cols(); i++) {
    VReal a(W.size());
    for(size_t k = 0; k < W.size(); ++k) {
      a[k] = bv(k, i) +  W[k].col(i).dot(h);
    }
    VReal b(W.size());
    ml::Softmax(a, &b);
    des->push_back(b[1]);
  }
}

double RBM::Predict(const Str &hid_path, const SpMat &train, const SpMat &test) {
  LOG(INFO) << hid_path;
  std::vector<EVec> h;
  h.resize(train.cols());
  ReadH(FLAGS_hidden, hid_path, &h);
  LOG(INFO) << h.size();
  double rmse = 0;
  int c = 0;
  LOG(INFO) << h.size();
  LOG(INFO) << test.cols();
  VVReal xx;
  for(int n = 1; n < test.cols(); n++) {
    if (h[n].size() != 0) {
      c++;
      v0.setZero(); // v0 must set zero
      ExpectRating(h[n], test.col(n), &v0);
      v0 -= test.col(n);
      rmse += v0.cwiseAbs2().sum();
      VReal tmp;
      tmp.push_back(sqrt(v0.cwiseAbs2().sum() / v0.nonZeros()));
      tmp.push_back(v0.nonZeros());
      xx.push_back(tmp);
    }
  }
  // LOG(INFO) << Join(xx, " ", "\n");
  LOG(INFO) << test.nonZeros();
  return sqrt(rmse/test.nonZeros());
}

void RBM::ROC(const Str &hid_path, const Str &res, const SpMat &train,
                                                   const SpMat &test) {
  std::vector<EVec> h;
  h.resize(train.cols());
  ReadH(FLAGS_hidden, hid_path, &h);
  
  VSInt positive;
  PositiveSample(test, &positive);
  VVReal predict_all;
  LOG(INFO) << test.cols();
  for(int n = 0; n < test.cols(); n++) {
    LOG_IF(INFO, n % 100 == 0) << n;
    if (h[n].size() == 0) {
      continue; //why h[n] is empty
    }
    VReal predict;
    ExpectV(h[n], &predict);
    for (size_t i = 0; i < predict.size(); i++) {
      VReal predict_pair;
      predict_pair.push_back(predict[i]);
      if (positive[n].find(i) != positive[n].end()) {
        predict_pair.push_back(2);
      } else {
        predict_pair.push_back(1);
      }
      predict_all.push_back(predict_pair);
    }
  }
  WriteStrToFile(Join(predict_all, " ", "\n"), res);
}

void RBM::Visual() {
  VVInt c;
  int k = 20;
  int iter = 1000;
  KMean(W[0], k, iter, &c);
  LOG(INFO) << Join(c, " ", "\n");
}
} // namespace ml
