// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#ifndef ML_RBM_RBMGAUSSIAN_
#define ML_RBM_RBMGAUSSIAN_
#include "base/base_head.h"
#include "ml/eigen.h"

//W: the first dimension is the rating dimension, 
//W[0].cols(), the visual dimension,
//W[0].rows(), the hidden feature dimension
namespace ml {
class RBMGaussian {
 public:
  RBMGaussian(int nv, int nh);
  void Train(const SpMat &train, const SpMat &test, int niter,
                                 double alpha, int batch_size);
  double Predict(const SpMat &train, const SpMat &test);

  void ExpectV(const SpVec &t, const EVec &h, SpVec *v);
  void ExpectV(const SpMat &mat, const EMat &h, SpMat* v);

  void SampleH(const SpVec &v, EVec *h);
  void ExpectH(const SpMat &t, EMat* h);

  double MAE(const SpMat &train, const SpMat &test);
  void Train2(const SpMat &train, const SpMat &test, int niter, double alpha);
  void UpdateGradient(double alpha, int batch_size);
  void InitGradient();
  void Gradient(const SpMat &v0, const EMat &h0, const SpMat &v1);
 public:
  SpVec v0, vk;
  EVec h0, hk;

 private:
  EMat W, dW;
  EVec bv, dv;
  EVec bh, dh;

  void ExpectH(const SpVec &v, EVec *h);
  void ExpectV(const EVec &h, const SpVec &t, VReal* des);
  void ExpectV(const EVec &h, VReal* des);

  void PartGrad(const SpVec &v, const EVec &h, double coeff);
  void Gradient(const SpVec &x, int step);
};
} // namespace ml
#endif // ML_RBM_RBMGAUSSIAN_
