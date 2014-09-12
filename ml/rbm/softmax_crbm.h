// Copyright 2013 zhangwei, lijiankou. All Rights Reserved.
// Author: zhangw@ios.ac.cn  lijk_start@163.com
#ifndef ML_RBM_SOFTMAX_CRBM_H_
#define ML_RBM_SOFTMAX_CRBM_H_
#include "base/base_head.h"
#include "ml/eigen.h"

//W: the first dimension is the rating dimension, 
//W[0].cols(), the visual dimension,
//W[0].rows(), the hidden feature dimension
namespace ml {
class SoftmaxCRBM {
 public:
  SoftmaxCRBM(int nv, int nh, int nsoftmax, int n_feature);
  double Predict(const SpMat &train, const SpMat &feature, const SpMat &test);
  void ExpectH(const SpMat &train, const SInt &test_dic, VVReal* h);
  void Train(const SpMat &train, const SpMat &test,
                        const SpMat &f, int niter,
                        double alpha, int batch_size);
 private:
  void InitGradient();
  void UpdateGradient(double alpha, int batch_size);

  void ExpectH(const SpVec &v, const SpVec &feature, EVec *h);
  void SampleH(const SpVec &v, const SpVec &f, EVec *h);

  void ExpectRating(const EVec &h, const SpVec &t, SpVec *v);
  void ExpectV(const EVec &h, const SpVec &t, VVReal* des);
  void SampleV(const EVec &h, const SpVec &t, SpVec *v);

  void PartGrad(const SpVec &v, const SpVec &f, const EVec &h, double coeff);
  void Gradient(const SpVec &x, const SpVec &f, int step);

  SpVec v0, vk;
  EVec h0, hk;
  EVec expect_h0;

  std::vector<EMat> w, dw;
  EMat bv, dv;
  EVec bh, dh;
  EMat u, du; //u.rows(), hidden dimension, u.cols(), visual feature dimension
};
} // namespace ml
#endif // ML_RBM_SOFTMAX_CRBM_H_
