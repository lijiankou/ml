// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#ifndef ML_RBM_RBM_BIN_
#define ML_RBM_RBM_BIN_
#include "base/base_head.h"
#include "ml/eigen.h"

//W.rows(), the hidden feature dimension
//W.cols(), the visual dimension,
namespace ml {
class RBMBin {
 public:
  RBMBin(int nv, int nh);
  void Train(const SpMat &train, const SpMat &test, int niter,
                                 double alpha, int batch_size);
  void ExpectV(const EVec &h, const SpVec &t, SpVec *v);
  double Predict(const SpMat &train, const SpMat &test);
  void Recommend(const SpMat &train, const SpMat &test, double t);
  void PrecisionRecall(const SpMat &train, const SpMat &test, VVReal* res);
  void Recommend(const SpMat &train, const SpMat &test,
                 const SpMat &recom, int niter, double alpha,
                 int batch_size);
 
 public:
  SpVec v0, vk;
  EVec h0, hk;
  EVec expect_h0;

 private:
  EMat W, dW;
  EVec bv, dv;
  EVec bh, dh;

  void InitGradient();
  void UpdateGradient(double alpha, int batch_size);

  void ExpectH(const SpVec &v, EVec *h);
  void SampleH(const SpVec &v, EVec *h);

  void ExpectV(const EVec &h, const SpVec &t, VReal* des);
  void SampleV(const EVec &h, const SpVec &t, SpVec *v);

  void ExpectV(const EVec &h, VReal* des);

  void PartGrad(const SpVec &v, const EVec &h, double coeff);
  void Gradient(const SpVec &x, int step);
};
} // namespace ml
#endif // ML_RBM_RBM_BIN_
