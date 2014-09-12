// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_MATRIX_FACTORIZATION_
#define ML_MATRIX_FACTORIZATION_
#include "base/base_head.h"
#include "ml/eigen.h"

namespace ml {
struct MF {
  EMat u;
  EMat v;
  EVec bu;
  EVec bv;
  double b_total;
};

//EMat:resize(rows_num, cols_num)
//mf->u: rows:latent dimension, cols: user_num
//mf->v: rows:latent dimension, cols: item_num
inline void RandomInit(int u, int v, int k, MF* mf) {
  mf->u.resize(k, u);
  mf->v.resize(k, v);
  mf->bu.resize(u);
  mf->bv.resize(v);
  NormalRandom(&(mf->u));
  NormalRandom(&(mf->v));
  NormalRandom(&(mf->bu));
  NormalRandom(&(mf->bv));
  mf->b_total = 0;
}

inline int Min(const SpMat &mat) {
  int min = 1000000000;
  for (int m = 0; m < mat.cols(); ++m) {
    for (SpMatInIt it(mat, m); it; ++it) {
      min = min<it.value() ? min:it.value();
    }
  }
  return min;
}

inline int Max(const SpMat &mat) {
  int max = -1;
  for (int m = 0; m < mat.cols(); ++m) {
    for (SpMatInIt it(mat, m); it; ++it) {
      max = max>it.value() ? max:it.value();
    }
  }
  return max;
}

//Matrix Factriozation with bias
inline double Test(const SpMat &mat, const MF &mf) {
  int max = Max(mat); 
  int min = Min(mat);
  double rmse = 0;
  for (int m = 0; m < mat.cols(); ++m) {
    for (SpMatInIt it(mat, m); it; ++it) {
      double a = mf.u.col(m).dot(mf.v.col(it.index())) + mf.bu(m) +
                                             mf.bv(it.index()) + mf.b_total;
      a = a>max ? max:a;
      a = a<min ? min:a;
      rmse += Square(a - it.value());
    }
  }
  return std::sqrt(rmse/mat.nonZeros());
}

inline void SGD(int it_num, double eta, double lambda, const SpMat &rating,
                            const SpMat &test, MF* mf) {
  for (int i = 0; i < it_num; ++i) {
    for (int j = 0; j < mf->u.cols(); ++j) {
      for (SpMatInIt it(rating, j); it; ++it) {
        EVec pu = mf->u.col(j);
        const int &k = it.index();
        double e = pu.dot(mf->v.col(k)) + mf->bu[j] + mf->bv[k] +
                   mf->b_total - it.value();
        mf->u.col(j) -= eta*(e*mf->v.col(k) + lambda*pu);
        mf->v.col(k) -= eta*(e*pu + mf->v.col(k)*lambda);
        mf->bu[j] -= eta*(e + lambda*mf->bu[j]);
        mf->bv[k] -= eta*(e + lambda*mf->bv[k]);
        mf->b_total -= eta*(e + lambda*mf->b_total);
      }
    }
    LOG_IF(INFO, i % 100 == 0) << i << "  " << Test(rating, *mf) <<
                               ":" << Test(test, *mf);
  }
}
} // namespace ml
#endif // ML_MATRIX_FACTORIZATION_
