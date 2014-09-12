// Copyright 2014 lijiankou. All Rights Reserved.  // Author: lijk_start@163.com
#ifndef ML_RBM_HRBM_H_
#define ML_RBM_HRBM_H_
#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/rbm/rbm_gaussian.h"

//W: the first dimension is the rating dimension, 
//W[0].cols(), the visual dimension,
//W[0].rows(), the hidden feature dimension
namespace ml {
class HRBM {
 public:
  HRBM(int user_num, int item_num, int hidden_user, int hidden_item);
  void Train(int niter, double alpha);

  void ExpectV(const EVec &h, const SpVec &t, SpVec *v);
  void LoadData(const SpMat &train_u, const SpMat &test_u,
                const SpMat &train_i, const SpMat &test_i); 

  double Predict(const SpMat &test_u, const SpMat &test_i);
 private:
  RBMGaussian rbm_u;
  RBMGaussian rbm_i;

  EMat h0_u;
  EMat h0_i;

  SpMat v1_u;
  SpMat v1_i;
  
  const SpMat* train_u;
  const SpMat* test_u;
  const SpMat* train_i;
  const SpMat* test_i;
};
} // namespace ml
#endif // ML_RBM_HRBM_H_
