// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "ml/rbm/rbm_util.h"

#include "base/base_head.h"
#include "ml/rbm/rbm.h"

namespace ml {
size_t Size(const VVInt &item) {
  size_t sum = 0;
  for(size_t i = 0; i < item.size(); i++) {
    sum += item[i].size();
  }
  return sum;
}

double SquareError(const VReal &lhs, const VReal &rhs) {
  double sum = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    sum += Square(lhs[i] - rhs[i]);
  }
  return sum;
}

void ReadData(const Str &path, int rows, int cols, SpMat *mat) {
  LOG(INFO) << path;
  if (!IsFile(path)) {
    LOG(INFO) << path + " not exist";
    assert(false);
  }
  FILE *fin = fopen(path.c_str(), "r");
  int u;
  int v;
  float r;
  int m = 0;
  int n = 0;
  std::vector<T> tripletList;
  while(fscanf(fin, "%d %d %f", &u, &v, &r) > 0) {
    tripletList.push_back(T(v, u, r));
    m = u>m ? u:m;
    n = v>n ? v:n;
  }
  cols = cols>(m + 1) ? cols:(m + 1);
  rows = rows>(n + 1) ? rows:(n + 1);
  LOG(INFO) << rows << " " << cols;
  mat->resize(rows, cols);
  mat->setFromTriplets(tripletList.begin(), tripletList.end());
}

void Convert(const std::vector<MatrixXd> &src, VVVReal* des) {
  Init(src.size(), src[0].rows(), src[0].cols(), 0.0, des);
  for (size_t i = 0; i < src.size(); ++i) {
    for(int j = 0; j < src[i].rows(); ++j) {
      for(int k = 0; k < src[i].cols(); ++k) {
        (*des)[i][j][k] = src[i](j, k);
      }
    }
  }
}

void Convert(const MatrixXd &src, VVReal* des) {
  Init(src.rows(), src.cols(), 0.0, des);
  for(int j = 0; j < src.rows(); ++j) {
    for(int k = 0; k < src.cols(); ++k) {
      (*des)[j][k] = src(j, k);
    }
  }
}
}  // namespace ml
