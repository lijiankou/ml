// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#include "ml/rbm/rbm_bin.h"

#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/rbm/rbm_util.h"
#include "ml/roc.h"
#include "ml/util.h"

DECLARE_double(lambda_weight);

namespace ml {
//W.rows(), the hidden feature dimension
//W.cols(), the visual dimension,
RBMBin::RBMBin(int nv, int nh) {
   LOG(INFO) << "nv:" << nv << "nh:" << nh;
   W.resize(nh, nv);
   NormalRandom(&W);
   dW.resize(nh, nv);

   bv.resize(nv);
   NormalRandom(&bv);
   dv.resize(nv);

   bh.resize(nh);
   NormalRandom(&bh);
   dh.resize(nh);

   v0.resize(nv);
   h0.resize(nh);
   expect_h0.resize(nh);
   vk.resize(nv);
   hk.resize(nh);
}

void RBMBin::ExpectV(const EVec &h, const SpVec &t, VReal* des) {
  for (SpVec::InnerIterator it(t); it; ++it){
    des->push_back(Sigmoid(bv(it.index()) +  W.col(it.index()).dot(h)));
  }
}

void RBMBin::ExpectV(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = vec[i];
  }
}

//this function is not effieicy, vec should be defined before
void RBMBin::SampleV(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = Sample1(vec[i]);
  }
}

void RBMBin::ExpectH(const SpVec &v, EVec *h) {
  for (int f = 0; f < h->rows(); ++f) {
    double s = 0;
    for (SpVec::InnerIterator it(v); it; ++it) {
      s += W(f, it.index()) * it.value();
    }
    (*h)[f] = Sigmoid(s + bh[f]);
  }
}

void RBMBin::SampleH(const SpVec &v, EVec *h) {
  ExpectH(v, h);
  ::Sample(h);
}

void RBMBin::PartGrad(const SpVec &v, const EVec &h, double coeff){
  for (SpVec::InnerIterator it(v); it; ++it) {
    dv(it.index()) += coeff * it.value();
    for (int j = 0; j < h.rows(); ++j) {
      dW(j, it.index()) += coeff * h[j] * it.value();
    }
  }
  for (int j = 0; j < h.rows(); ++j) {
    dh(j) += coeff * h[j];
  }
}

void RBMBin::InitGradient(){
  dW.setZero();
  dh.setZero();
  dv.setZero();
}

void RBMBin::UpdateGradient(double alpha, int batch_size) {
  double r = alpha / batch_size;
  W += r * dW;
  //W += r * dW + r * W * FLAGS_lambda_weight;
  bh += r * dh;
  bv += r * dv;
}

/*
void RBMBin::Gradient(const SpVec &v0, int step) {
  SampleH(v0, &h0);
  SampleV(h0, v0, &vk);
  for (int k = 0; k < step - 1; ++k) {
    SampleH(vk, &hk);
    SampleV(hk, v0, &vk);
  }
  ExpectH(vk, &hk);
  PartGrad(v0, h0, 1); //positive phase
  PartGrad(vk, hk, -1); //negative phase
}
*/

void RBMBin::Gradient(const SpVec &v0, int step) {
  ExpectH(v0, &expect_h0);
  ::Sample(expect_h0, &h0);
  SampleV(h0, v0, &vk);
  for (int k = 0; k < step - 1; ++k) {
    SampleH(vk, &hk);
    SampleV(hk, v0, &vk);
  }
  ExpectH(vk, &hk);
  PartGrad(v0, expect_h0, 1); //positive phase
  PartGrad(vk, hk, -1); //negative phase
}

void RBMBin::Train(const SpMat &train, const SpMat &test, int niter,
                                       double alpha, int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = 1;
  for (int i = 0; i < niter; ++i) {
    for (int n = 1; n < train.cols(); n++) {
      Gradient(train.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    VVReal v(2);
    LOG(INFO) << i << " " << Predict(train, train) <<
                      " " << Predict(train, test);
    PrecisionRecall(train, test, &v);
    double p = DecisionPro(v[0], v[1]);
    LOG(INFO) << F1Score(v[0], v[1]) << " " << p;
  }
}

void RBMBin::Recommend(const SpMat &train, const SpMat &test,
                       const SpMat &recom, int niter, double alpha,
                       int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = 1;
  for (int i = 0; i < niter; ++i) {
    for (int n = 1; n < train.cols(); n++) {
      Gradient(train.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    VVReal v(2);
    LOG(INFO) << i << " " << Predict(train, train) <<
                      " " << Predict(train, test);
    PrecisionRecall(train, test, &v);
    PrecisionRecall(train, test, &v);
    double p = DecisionPro(v[0], v[1]);
    LOG(INFO) << F1Score(v[0], v[1]) << " " << p;
    Recommend(train, recom, p);
  }
}

//it's important using expect and not use sample
double RBMBin::Predict(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for(int n = 1; n < train.cols(); n++) {
    ExpectH(train.col(n), &h0);
    ExpectV(h0, test.col(n), &v0);
    v0 -= test.col(n);
    rmse += v0.cwiseAbs2().sum();
  }
  return sqrt(rmse/test.nonZeros());
}

void RBMBin::PrecisionRecall(const SpMat &train, const SpMat &test, VVReal* res) {
  res->resize(2);
  for(int n = 1; n < train.cols(); n++) {
    ExpectH(train.col(n), &h0);
    ExpectV(h0, test.col(n), &v0);
    SpMat::InnerIterator it2(test, n);
    for (SpVec::InnerIterator it(v0); it; ++it, ++it2) {
      res->at(0).push_back(it2.value() + 1);
      res->at(1).push_back(it.value());
    }
  }
}

void RBMBin::Recommend(const SpMat &train, const SpMat &test, double t) {
  VVReal res;
  for(int n = 1; n < train.cols(); n++) {
    VReal tmp;
    ExpectH(train.col(n), &h0);
    ExpectV(h0, test.col(n), &v0);
    for (SpVec::InnerIterator it(v0); it; ++it) {
      if (it.value() > t) {
        tmp.push_back(it.index());
      }
    }
    res.push_back(tmp);
  }
  WriteStrToFile(Join(res, " ", "\n"), "recommend");
}
} // namespace ml
