// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef RBM_RBM_UTIL_H_
#define RBM_RBM_UTIL_H_
#include "base/base_head.h"
#include "ml/util.h"
#include "thirdparty/Eigen/Sparse"
#include "thirdparty/Eigen/Dense"

namespace ml {
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::Triplet<double> T;

double SquareError(const VReal &lhs, const VReal &rhs);
size_t Size(const VVInt &item);
void ReadData(const Str &path, int rows, int cols, SpMat *m);
void ReadData(const Str &path, SpMat *mat);

inline double ExpectRating(const VReal &a){
  double s = 0;
  for(size_t i = 0; i < a.size(); ++i){
    s += (i + 1) * a[i];
  }
  return s;
}

using Eigen::MatrixXd;
using Eigen::VectorXd;

void Convert(const std::vector<MatrixXd> &src, VVVReal* des);
void Convert(const MatrixXd &src, VVReal* des);

inline double Var(const std::vector<MatrixXd> &src) {
  VVVReal tmp;
  Convert(src, &tmp);
  return Var(tmp);
}

inline double Mean(const std::vector<MatrixXd> &src) {
  VVVReal tmp;
  Convert(src, &tmp);
  return Mean(tmp);
}

inline double Var(const MatrixXd &src) {
  VVReal tmp;
  Convert(src, &tmp);
  return Var(tmp);
}

inline double Mean(const MatrixXd &src) {
  VVReal tmp;
  Convert(src, &tmp);
  return Mean(tmp);
}

// check the rating dimension, the max rating should equal k
inline bool CheckData(const SpMat &data, int k) {
  int max = 0;
  for (int i = 0; i < data.cols(); i++) {
    for (SpMat::InnerIterator it(data, i); it; ++it) {
      max = max > it.value() ? max : it.value();
    }
  }
  return max == k ? true : false;
}

inline double Range(double value, double max, double min) {
  value = value>max ? max:value;
  value = value<min ? min:value;
  return value;
}

inline void Range(const SpVec &src, SpVec* des) {
  des->resize(src.size());
  for (SpVec::InnerIterator it(src); it; ++it) {
    des->insert(it.index()) = Range(it.value(), 5.0, 1.0);
  }
}
}  // namespace ml
#endif // RBM_RBM_UTIL_H_
