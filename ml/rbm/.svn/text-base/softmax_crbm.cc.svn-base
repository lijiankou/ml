// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#include "ml/rbm/softmax_crbm.h"

#include "base/base_head.h"
#include "ml/rbm/rbm_util.h"
#include "ml/eigen.h"
#include "ml/util.h"

DECLARE_int32(hidden);
DECLARE_double(lambda_weight);
DECLARE_double(lambda_feature);

DECLARE_int32(nCD);

using std::string;
namespace ml {
SoftmaxCRBM::SoftmaxCRBM(int nv, int nh, int nsoftmax, int n_feature) {
   w.resize(nsoftmax);
   dw.resize(nsoftmax);
   for (size_t i = 0; i < w.size(); ++i) {
     w[i].resize(nh, nv);
     dw[i].resize(nh, nv);
     NormalRandom(&w[i]);
   }

   u.resize(nh, n_feature);
   NormalRandom(&u);
   du.resize(nh, n_feature);

   bv.resize(nsoftmax, nv);
   dv.resize(nsoftmax, nv);
   NormalRandom(&bv);

   bh.resize(nh);
   bh.setZero();
   dh.resize(nh);

   v0.resize(nv);
   h0.resize(nh);
   expect_h0.resize(nh);
   vk.resize(nv);
   hk.resize(nh);
}

void SoftmaxCRBM::ExpectV(const EVec &h, const SpVec &t, VVReal* des) {
  for (SpVec::InnerIterator it(t); it; ++it){
    VReal a(w.size());
    for(size_t k = 0; k < w.size(); ++k) {
      a[k] = bv(k, it.index()) +  w[k].col(it.index()).dot(h);
    }
    VReal b(w.size());
    ml::Softmax(a, &b);
    des->push_back(b);
  }
}

void SoftmaxCRBM::ExpectRating(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VVReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = ml::ExpectRating(vec[i]);
  }
}

void SoftmaxCRBM::SampleV(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VVReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = Sample(vec[i]);
  }
}

void SoftmaxCRBM::ExpectH(const SpVec &v, const SpVec &feature, EVec *h) {
  for (int j = 0; j < h->rows(); ++j) {
    double s = 0;
    for (SpVec::InnerIterator it(v); it; ++it) {
      s += w[it.value() - 1](j, it.index());
    }
   // for (SpVec::InnerIterator it(feature); it; ++it) {
    //  s += u(j, it.index()) * it.value();
   // }
    (*h)[j] = Sigmoid(s + bh[j]);
  }
}

void SoftmaxCRBM::SampleH(const SpVec &v, const SpVec &f, EVec *h) {
  ExpectH(v, f, h);
  ::Sample(h);
}

void SoftmaxCRBM::PartGrad(const SpVec &v, const SpVec &f,
                           const EVec &h, double coeff) {
  for (SpVec::InnerIterator it(v); it; ++it) {
    dv(it.value() - 1, it.index()) += coeff;
    for (int j = 0; j < h.rows(); ++j) {
      dw[it.value() - 1](j, it.index()) += coeff * h[j];
    }
  }
  for (SpVec::InnerIterator it(f); it; ++it) {
    for (int j = 0; j < h.rows(); ++j) {
      du(j, it.index()) += coeff * h[j] * it.value();
    }
  }
  for (int j = 0; j < h.rows(); ++j) {
    dh(j) += coeff * h[j];
  }
}

void SoftmaxCRBM::InitGradient(){
  for(size_t k = 0; k < dw.size(); ++k) {
    dw[k].setZero();
  }
  dh.setZero();
  dv.setZero();
  du.setZero();
}

void SoftmaxCRBM::UpdateGradient(double alpha, int batch_size) {
  double r = alpha / batch_size;
  for(size_t k = 0; k < dw.size(); ++k) {
    w[k] += r * dw[k] + r * w[k] * FLAGS_lambda_weight;
  }
  u += r * (du + u * FLAGS_lambda_feature);
  bh += r * dh;
  bv += r * dv;
}

void SoftmaxCRBM::Gradient(const SpVec &x, const SpVec &f, int step) {
  ExpectH(v0, f, &expect_h0);
  ::Sample(expect_h0, &h0);
  SampleV(h0, x, &vk);
  for (int k = 0; k < step - 1; ++k) {
    SampleH(vk, f, &hk);
    SampleV(hk, x, &vk);
  }
  ExpectH(vk, f, &hk);
  PartGrad(x, f, expect_h0, 1);
  PartGrad(vk, f, hk, -1);
}

void SoftmaxCRBM::Train(const SpMat &train, const SpMat &f, const SpMat &test,
                        int niter, double alpha, int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = FLAGS_nCD;
  for (int i = 0; i < niter; ++i) {
    for (int n = 0; n < train.cols(); n++) {
      Gradient(train.col(n), f.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    LOG(INFO) << i << " " << Predict(train, f, train)
              << " " << Predict(train, f, test);
  }
}

double SoftmaxCRBM::Predict(const SpMat &train, const SpMat &f,
                                                const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    ExpectH(train.col(n), f.col(n), &h0);
    ExpectRating(h0, test.col(n), &v0);
    v0 -= test.col(n);
    rmse += v0.cwiseAbs2().sum();
  }
  return sqrt(rmse/test.nonZeros());
}
} // namespace ml
