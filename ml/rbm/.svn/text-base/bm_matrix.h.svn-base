// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#ifndef ML_BM_MATRIX_H_
#define ML_BM_MATRIX_H_
#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/util.h"

namespace ml {
class BM {
 public:
  BM(int bm_num, int k, double lambda);
  void Train(const SpMat &train, const SpMat &test, int niter, double alpha);
  double Predict(const SpMat &train, const SpMat &test);
  void LoadColdMovie(const SpMat &test);
  EVec SumCol(const SpMat &data, int n);
 
 private:
  void Gradient(const SpMat &data, double alpha);

  EVec b;
  EMat u;
  double lambda;
};
} // namespace ml
#endif // ML_BM_MATRIX_H_
