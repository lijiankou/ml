// Copyright 2013 zhangwei, lijiankou. All Rights Reserved.
// Author: zhangw@ios.ac.cn  lijk_start@163.com
#ifndef ML_RBM_RBM_
#define ML_RBM_RBM_
#include "base/base_head.h"
#include "ml/eigen.h"

//W: the first dimension is the rating dimension, 
//W[0].cols(), the visual dimension,
//W[0].rows(), the hidden feature dimension
namespace ml {
class RBM {
 public:
  RBM(int nv, int nh, int nsoftmax);
  void Train(const SpMat &train, const SpMat &test, int niter,
                                 double alpha, int batch_size);
  double Predict(const SpMat &train, const SpMat &test);
  double Predict(const Str &hid_path, const SpMat &train, const SpMat &test);
  double PredictWithError(const SpMat &train, const SpMat &test);
  double PredictWithRandom(const SpMat &train, const SpMat &test);

  void TestROC(const Str &hid_path, const Str &res, const SpMat &train,
                                                    const SpMat &test);
  void ExpectH(const SpMat &train, const SInt &test_dic, VVReal* h);
  void SaveModel(const Str &path0, const Str &path1,
                    const Str &path2, const Str &path3) const;
  void LoadModel(const Str &path0, const Str &path1,
                    const Str &path2, const Str &path3);

  void ROC(const Str &hid_path, const Str &res, const SpMat &train,
                                                const SpMat &test);
  void Visual();
 public:
  SpVec v0, vk;
  EVec h0, hk;

 private:
  std::vector<EMat> W, dW;
  EMat bv, dv;
  EVec bh, dh;

  void InitGradient();
  void UpdateGradient(double alpha, int batch_size);
  void ExpectH(const SpVec &v, EVec *h);
  void SampleH(const SpVec &v, EVec *h);
  void ExpectRating(const EVec &h, const SpVec &t, SpVec *v);
  void ExpectV(const EVec &h, const SpVec &t, VVReal* des);
  void SampleV(const EVec &h, const SpVec &t, SpVec *v);

  void ExpectV(const EVec &h, VReal* des);

  void PartGrad(const SpVec &v, const EVec &h, double coeff);
  void Gradient(const SpVec &x, int step);
};
} // namespace ml
#endif // ML_RBM_RBM_
