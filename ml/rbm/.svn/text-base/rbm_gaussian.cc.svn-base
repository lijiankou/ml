// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#include "ml/rbm/rbm_gaussian.h"

#include "base/base_head.h"
#include "ml/rbm/rbm_util.h"
#include "ml/eigen.h"
#include "ml/util.h"

DEFINE_double(lambda_weight, 0.01, "regularize parameter");

namespace ml {
RBMGaussian::RBMGaussian(int nv, int nh) {
   W.resize(nh, nv);
   NormalRandom(&W);
   dW.resize(nh, nv);

   bv.resize(nv);
   NormalRandom(&bv);
   dv.resize(nv);

   bh.resize(nh);
   bh.setZero();
   dh.resize(nh);

   v0.resize(nv);
   h0.resize(nh);
   vk.resize(nv);
   hk.resize(nh);
   assert(h0.size() == W.rows());
}

void RBMGaussian::ExpectH(const SpVec &v, EVec *h) {
  for (int j = 0; j < h->rows(); ++j) {
    double s = 0;
    for (SpVec::InnerIterator it(v); it; ++it) {
      s += W(j, it.index()) * it.value();
    }
    (*h)[j] = Sigmoid(s + bh[j]);
  }
}

void RBMGaussian::SampleH(const SpVec &v, EVec *h) {
  ExpectH(v, h);
  ::Sample(h);
}

void RBMGaussian::PartGrad(const SpVec &v, const EVec &h, double coeff){
  for (SpVec::InnerIterator it(v); it; ++it) {
    //dv(it.index()) += coeff * (it.value() - bv(it.index()));
    dv(it.index()) += coeff * (bv(it.index()) - it.value());
    for (int j = 0; j < h.rows(); ++j) {
      dW(j, it.index()) += coeff * h[j] * it.value();
    }
  }
  for (int j = 0; j < h.rows(); ++j) {
    dh(j) += coeff * h[j];
  }
}

void RBMGaussian::InitGradient(){
  dW.setZero();
  dh.setZero();
  dv.setZero();
}

void RBMGaussian::UpdateGradient(double alpha, int batch_size) {
  double r = alpha / batch_size;
  W += r * dW + r * W * FLAGS_lambda_weight;
  bh += r * dh;
  bv += r * dv;
}

void RBMGaussian::Gradient(const SpVec &v0, int step) {
  //EVec expect_h0(h0.size());
  //ExpectH(v0, &expect_h0);
  SampleH(v0, &h0);
  ExpectV(v0, h0, &vk);
  for (int k = 0; k < step - 1; ++k) {
    SampleH(vk, &hk);
    ExpectV(v0, hk, &vk);
  }
  ExpectH(vk, &hk);
  PartGrad(v0, h0, 1); //positive phase
  PartGrad(vk, hk, -1); //negative phase
}

void RBMGaussian::Train(const SpMat &train, const SpMat &test, int niter,
                        double alpha, int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = 1;
  for (int i = 0; i < niter; ++i) {
    for (int n = 0; n < train.cols(); n++) {
      Gradient(train.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    //LOG(INFO) << i << ":" << Predict(train, train) << " " << Predict(train, test);
    LOG(INFO) << i << ":" << MAE(train, train) << " " << MAE(train, test);
  }
}

//must using expect instead of sample
double RBMGaussian::MAE(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    SampleH(train.col(n), &h0);
    ExpectV(test.col(n), h0, &vk);
    SpVec tmp;
    Range(vk, &tmp);
    tmp -= test.col(n);
    rmse += tmp.cwiseAbs().sum();
  }
  return rmse/test.nonZeros();
}

//must using sample intead of expect
double RBMGaussian::Predict(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    //ExpectH(train.col(n), &h0);
    SampleH(train.col(n), &h0);
    ExpectV(test.col(n), h0, &vk);
    SpVec tmp;
    Range(vk, &tmp);
    tmp -= test.col(n);
    rmse += tmp.cwiseAbs2().sum();
  }
  return sqrt(rmse/test.nonZeros());
}

void RBMGaussian::ExpectH(const SpMat &v, EMat* h) {
  for (int n = 1; n < v.cols(); n++) {
    EVec tmp(W.rows());
    ExpectH(v.col(n), &tmp);
    h->col(n) = tmp;
  }
}

void RBMGaussian::ExpectV(const SpVec &t, const EVec &h, SpVec *v) {
  v->setZero();
  for (SpVec::InnerIterator it(t); it; ++it) {
    v->insert(it.index()) = bv(it.index()) +  W.col(it.index()).dot(h);
  }
}

void RBMGaussian::ExpectV(const SpMat &mat, const EMat &h, SpMat* v) {
  for (int n = 1; n < mat.cols(); n++) {
    v0.setZero();
    ExpectV(mat.col(n), h.col(n), &v0);
    v->col(n) = v0;
  }
}

void RBMGaussian::Gradient(const SpMat &v0, const EMat &h0, const SpMat &v1) {
  for (int n = 1; n < v0.cols(); n++) {
    PartGrad(v0.col(n), h0.col(n), 1); //positive phase
    ExpectH(v1.col(n), &(this->h0));
    PartGrad(v1.col(n), this->h0, -1); //negative phase
  }
}

void RBMGaussian::Train2(const SpMat &train, const SpMat &test, int niter, double alpha) {
  InitGradient();
  for (int i = 0; i < niter; ++i) {
    EMat tmp(W.rows(), W.cols());
    ExpectH(train, &tmp);
    ::Sample(&tmp);
    SpMat v1_u(train.rows(), train.cols());
    ExpectV(train, tmp, &v1_u);
    Gradient(train, tmp, v1_u);
    UpdateGradient(alpha, train.cols());
    InitGradient();
    LOG(INFO) << i << " " << Predict(train, train) << " " << Predict(train, test);
  }
}
} // namespace ml
