// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#ifndef ML_RBM_CRBM_H_
#define ML_RBM_CRBM_H_
#include "base/base_head.h"
#include "ml/eigen.h"

namespace ml {
class CRBM {
 public:
  CRBM(int n_feature, int nv, int nh);
  void InitW();
  void InitFeature();
  void Train(const SpMat &train, const SpMat &feature, const SpMat &test,
             int niter, double alpha, int batch_size);

  void Train2(const SpMat &train, const SpMat &feature, const SpMat &t_f,
              const SpMat &test, int niter, double alpha, int batch_size);

  double Predict(const SpMat &train, const SpMat &f, const SpMat &test);

  void PreGradient(const SpVec &v0, const SpVec &f0, int step);
  void PreTrain(const SpMat &train, const SpMat &feature, const SpMat &test,
                int niter, double alpha, int batch_size);
  void PreUpdateGradient(double alpha, int batch_size);
  void Visual();
  void VisualW();

  double Task3GROC(const SpMat &train, const SpMat &f, const SpMat &test,
                     VVReal* res);
  void Task3IROC(const SpMat &train, const SpMat &f, const SpMat &test);

  double Task2GROC(const SpMat &train, const SpMat &f, const SpMat &test,
                                                       VVReal *res);
  void Task2IROC(const SpMat &train, const SpMat &f, const SpMat &test,
                                                     const Str &path);

  void LoadColdMovie(const SpMat &test);

  void PrecisionRecall(const SpMat &train, const SpMat &f, const SpMat &test,
                             VVReal* res);
  void Recommend(const SpMat &train, const SpMat &test, double t);
 
 private:
  void InitGradient();
  void UpdateGradient(double alpha, int batch_size);

  void ExpectH(const SpVec &v, const SpVec &f, EVec *h);
  void SampleH(const SpVec &v, const SpVec &f, EVec *h);

  void ExpectV(const EVec &h, const SpVec &t, VReal* des);
  void SampleV(const EVec &h, const SpVec &t, SpVec *v);
  void ExpectV(const EVec &h, VReal* des);
  void ExpectV(const EVec &h, const SpVec &t, SpVec *v);

  void PartGrad(const SpVec &v, const SpVec &f, const EVec &h, double coe);
  void Gradient(const SpVec &v0, const SpVec &f, int step);

  SpVec v0, vk;
  SpVec f0, fk;
  EVec h0, hk;
  EVec expect_h0;

  EMat w, dw; //w.rows(), hidden dimension, w.cols(), visual dimension
  EMat u, du; //u.rows(), hidden dimension, u.cols(), visual feature dimension
  EVec bv, dv;
  EVec bh, dh;
  EVec bf, df;

  SInt cold_dic;
};
} // namespace ml
#endif // ML_RBM_CRBM_H_
