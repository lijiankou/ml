// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#include "ml/rbm/bm_matrix.h"

#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/util.h"

DEFINE_double(bm_negative_pro, 0.2, "the accept probability of negative sample");

namespace ml {
//bm_num is the num of unit, k is the num of effective component
BM::BM(int bm_num, int k, double lambda) {
  LOG(INFO) << bm_num << " " << k;
  b.resize(bm_num);
  u.resize(k, bm_num);
  LOG(INFO) << u.rows() << " " << u.cols();
  NormalRandom(&b);
  NormalRandom(&u);
  this->lambda = lambda;
}

EVec BM::SumCol(const SpMat &data, int n) {
  EVec res(u.rows());
  res.setZero();
  for (SpMat::InnerIterator it(data, n); it; ++it) {
    if (it.value() == 1) {
      res += u.col(it.index());
    } 
  }
  return res;
}

void BM::Gradient(const SpMat &data, double alpha) {
  // actor start from 1
  for (int k = 1; k < data.cols(); k++) {
    EVec res = SumCol(data, k);
    for (SpMat::InnerIterator it(data, k); it; ++it) {
      int id = it.index();
      if (it.value() == 1) {
        res -= u.col(it.index());
      }
      double sig = Sigmoid(b[id] + u.col(id).dot(res));
      u.col(id) -= alpha *((sig - it.value()) * res + lambda * u.col(id));
      b[id] -= alpha * (sig - it.value() + lambda *b[id]);
      for (SpMat::InnerIterator it2(data, k); it2; ++it2) {
        int id2 = it2.index();
        if (id2 != id) {
          u.col(id2) -= alpha * ((sig - it.value()) 
               * u.col(id) * it2.value() + lambda * u.col(id2));
        }
      }
    }
  }
}

void BM::Train(const SpMat &train, const SpMat &test, int niter, double alpha) {
  LOG(INFO) << train.rows() << " " << train.cols();
  for (int i = 0; i < niter; ++i) {
    EMat con_p;
    Gradient(train, alpha);
    LOG(INFO) << i << " " << Predict(train, train) << " " << Predict(train, test);
  }
}

double BM::Predict(const SpMat &train, const SpMat &test) {
  double rmse = 0;
  for (int i = 1; i < test.cols(); i++) {
    EVec res = SumCol(train, i);
    for (SpMat::InnerIterator it(test, i); it; ++it) {
      rmse += Square(b[it.index()] + u.col(it.index()).dot(res) - it.value());
    }
  }
  return sqrt(rmse/test.nonZeros());
}
} // namespace ml
