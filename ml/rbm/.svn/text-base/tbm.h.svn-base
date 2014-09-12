// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com
#ifndef ML_BM_TBM_H_
#define ML_BM_TBM_H_
#include "base/base_head.h"
#include "ml/eigen.h"

namespace ml {
class TBM {
 public:
  TBM(int n_feature, double lambda);
  void Train(const SpMat &u_m, const SpMat &m_f, const SpMat &test,
             int niter, double alpha);
  double Predict(const EMat &u_f_num, const SpMat &m_f, const SpMat &test);
  void LoadColdMovie(const SpMat &test);

  void Task3IROC(const EMat &u_f_num, const SpMat &m_f,
                 const SpMat &test, const Str &path);

  void Task1IROC(const EMat &u_f_num, const SpMat &m_f,
                 const SpMat &test, const Str &path);
 
 private:
  double Condition(const EMat &u_f_num, const SpMat &m_f,
                   int u_id, int m_id) const;
  void Gradient(const EMat &adj, const SpMat &f_m, const EMat &con_p,
                                 const EMat &u_f_num, double alpha);
  void Condition(const EMat &u_f_num, const SpMat &m_f, EMat* p) const;

  void Task1GROC(const EMat &u_f_num, const SpMat &m_f,
                 const EMat &test_adj, const Str &path);
  void Task3GROC(const EMat &u_f_num, const SpMat &m_f,
                            const SpMat &test, const Str &path);
  EVec mu;
  EVec eta;
  double lambda;

  SInt cold_dic;
};

inline double PseudoLikelihood(const EMat &p, const EMat &adj) {
  double res = 0;
  for (int i = 1; i < p.cols(); i++) {
    for (int j = 1; j < p.rows(); j++) {
      res += LogBer(adj(j, i), p(j, i));
    }
  }
  return res;
}
} // namespace ml
#endif // ML_BM_TBM_H_
