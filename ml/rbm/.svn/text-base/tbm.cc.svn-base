// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com
#include "ml/rbm/tbm.h"

#include "base/base_head.h"
#include "ml/eigen.h"
#include "ml/rbm/roc.h"
#include "ml/util.h"

DEFINE_double(negative_pro, 0.2, "the accept probability of negative sample");
DEFINE_string(tbm_task1_groc, "ab", "cd");
DEFINE_string(tbm_task2_groc, "", "");
DEFINE_string(tbm_task3_groc, "", "");
DEFINE_string(tbm_task1_iroc, "", "");
DEFINE_string(tbm_task2_iroc, "", "");
DEFINE_string(tbm_task3_iroc, "", "");
DEFINE_bool(task1_flag, false, "");
DEFINE_bool(task2_flag, false, "");
DEFINE_bool(task3_flag, false, "");

namespace ml {
TBM::TBM(int n_feature, double lambda) {
  assert(n_feature != 0);

  mu.resize(n_feature);
  eta.resize(n_feature);
  NormalRandom(&mu);
  NormalRandom(&eta);
  eta.setZero();
  mu.setZero();
 
  this->lambda = lambda;
}

void TBM::LoadColdMovie(const SpMat &test) {
  for (int i = 1; i < test.cols(); i++) {
    for (SpMat::InnerIterator it(test, i); it; ++it) {
      cold_dic.insert(it.index());
    } 
  }
}

//for a user, calculate the number of movies that include an actor
void CalUserFeatureNum(const SpMat &u_v, const SpMat &m_f, EMat* dic) {
  dic->resize(m_f.rows(), u_v.cols());
  dic->setZero();
  for (int i = 1; i < u_v.cols(); i++) {
    for (SpMat::InnerIterator it(u_v, i); it; ++it) {
      for (SpMat::InnerIterator it2(m_f, it.index()); it2; ++it2) {
        (*dic)(it2.index(), i) += 1;
      }
    }
  }
}

double InnerProd(const SpVec &vec, const EVec &v) {
  double sum = 0;
  for (SpVec::InnerIterator it(vec); it; ++it) {
    sum += it.value() * v[it.index()]; 
  }
  return sum;
}

void TBM::Condition(const EMat &u_f_num, const SpMat &m_f, EMat* p) const {
  p->resize(m_f.cols(), u_f_num.cols());
  for (int i = 1; i < u_f_num.cols(); i++) {
    for (int j = 1; j < m_f.cols(); j++) {
      double res = InnerProd(m_f.col(j), mu);  
      for (SpMat::InnerIterator it(m_f, j); it; ++it) {
        res += eta[it.index()] * (u_f_num(it.index(), i) - 1);
      }
      (*p)(j, i) = Sigmoid(res);
    }
  }
}

void TBM::Gradient(const EMat &adj, const SpMat &f_m, const EMat &con_p,
                   const EMat &u_f_num, double alpha) {
  // actor start from 1
  for (int k = 1; k < mu.size(); k++) {
    for (SpMat::InnerIterator it(f_m, k); it; ++it) {
      for (int i = 1; i < adj.cols(); i++) {
        int user_id = i;
        if (adj(it.index(), user_id) == 0) {
          if (Random1() < 1 - FLAGS_negative_pro) {
            continue;
          }
        }
        double d_mu = con_p(it.index(), i) - adj(it.index(), user_id);
        mu[k] -= alpha * (d_mu + lambda * mu[k]);
        double d_eta = d_mu * (u_f_num(k, user_id) - 1);
        eta[k] -= alpha * (d_eta + lambda * eta[k]);
      }
    }
  }
}

void DelColdStartMovie(const SpMat &m_f, const SInt &dic, SpMat* des) {
  des->resize(m_f.rows(), m_f.cols());
  for (int i = 1; i < m_f.cols(); i++) {
    if (dic.find(i) == dic.end()) {
      des->col(i) = m_f.col(i);
    }
  }
}

void TBM::Train(const SpMat &train, const SpMat &m_f, const SpMat &test,
                                    int niter, double alpha) {
  LOG(INFO) << m_f.rows() << " " << m_f.cols();
  SpMat m_u = train.transpose();
  EMat adj(train);
  EMat test_adj(test);

  EMat u_f_num;
  CalUserFeatureNum(train, m_f, &u_f_num);
  
  SpMat m_f2;
  DelColdStartMovie(m_f, cold_dic, &m_f2);
  SpMat f_m = m_f2.transpose();
 
  for (int i = 0; i < niter; ++i) {
    EMat con_p;
    Condition(u_f_num, m_f, &con_p);
    Gradient(adj, f_m, con_p, u_f_num, alpha);
    LOG(INFO) << i << " " << Predict(u_f_num, m_f, test);
    if (FLAGS_task1_flag) {
      Task1GROC(u_f_num, m_f, test_adj, FLAGS_tbm_task1_groc);
      Task1IROC(u_f_num, m_f, test, FLAGS_tbm_task1_iroc);
    }
    if (FLAGS_task2_flag) {
      Task1GROC(u_f_num, m_f, test_adj, FLAGS_tbm_task2_groc);
      Task1IROC(u_f_num, m_f, test, FLAGS_tbm_task2_iroc);
    }
    if (FLAGS_task3_flag) {
      Task3GROC(u_f_num, m_f, test, FLAGS_tbm_task3_groc);
      Task3IROC(u_f_num, m_f, test, FLAGS_tbm_task3_iroc);
    }
  }
}

double TBM::Condition(const EMat &u_f_num, const SpMat &m_f, int u_id,
                                           int m_id) const {
  double res = InnerProd(m_f.col(m_id), mu);  
  for (SpMat::InnerIterator it(m_f, m_id); it; ++it) {
    res += eta(it.index()) * (u_f_num(it.index(), u_id));
  }
  return Sigmoid(res);
}

double TBM::Predict(const EMat &u_f_num, const SpMat &m_f, const SpMat &test) {
  SpMat test2 = test.transpose();
  double rmse = 0;
  for (SInt::iterator it = cold_dic.begin(); it != cold_dic.end(); ++it) {
    for (SpMat::InnerIterator it2(test2, *it); it2; ++it2) {
      rmse += Square(it2.value() - Condition(u_f_num, m_f, it2.index(), *it));
    }
  }
  return sqrt(rmse/test2.nonZeros());
}

void TBM::Task3GROC(const EMat &u_f_num, const SpMat &m_f,
                    const SpMat &test, const Str &path) {
  SpMat test2 = test.transpose();
  VVReal res(2);
  for (SInt::iterator it = cold_dic.begin(); it != cold_dic.end(); ++it) {
    for (SpMat::InnerIterator it2(test2, *it); it2; ++it2) {
      res[0].push_back(it2.value() + 1);
      res[1].push_back(Condition(u_f_num, m_f, it2.index(), *it));
    }
  }
  WriteStrToFile(Join(res, " ", "\n"), path);
}

void TBM::Task1GROC(const EMat &u_f_num, const SpMat &m_f,
                    const EMat &test_adj, const Str &path) {
  int user_num = u_f_num.cols();
  VVReal res(2);
  for (SInt::iterator it = cold_dic.begin(); it != cold_dic.end(); ++it) {
    for (int j = 1; j < user_num; j++) {
      res[0].push_back(test_adj(*it, j) + 1);
      res[1].push_back(Condition(u_f_num, m_f, j, *it));
    }
  }
  WriteStrToFile(Join(res, " ", "\n"), path);
}

void TBM::Task3IROC(const EMat &u_f_num, const SpMat &m_f,
                    const SpMat &test, const Str &path) {
  int user_num = u_f_num.cols();
  int item_num = m_f.cols();
  SpMat pre(user_num, item_num);
  SpMat test2 = test.transpose();
  for (SInt::iterator it = cold_dic.begin(); it != cold_dic.end(); ++it) {
    SpVec tmp(user_num);
    for (SpMat::InnerIterator it2(test2, *it); it2; ++it2) {
      tmp.insert(it2.index()) = Condition(u_f_num, m_f, it2.index(), *it);
    }
    pre.col(*it) = tmp;
  }
  VVInt res;
  CROC(pre, test2, &res);
  WriteStrToFile(Join(res, " ", "\n"), path);
}

void TBM::Task1IROC(const EMat &u_f_num, const SpMat &m_f,
                    const SpMat &test, const Str &path) {
  int user_num = u_f_num.cols();
  int item_num = m_f.cols();
  SpMat pre(user_num, item_num);
  SpMat real(user_num, item_num);
  SpMat test2 = test.transpose();
  EMat adj(test2);
  for (SInt::iterator it = cold_dic.begin(); it != cold_dic.end(); ++it) {
    SpVec tmp(user_num);
    SpVec tmp2(user_num);
    for (int i = 1; i < user_num; ++i) {
      tmp.insert(i) = Condition(u_f_num, m_f, i, *it);
      tmp2.insert(i) = adj(i, *it);
    }
    pre.col(*it) = tmp;
    real.col(*it) = tmp2;
  }
  VVInt res;
  CROC(pre, real, &res);
  WriteStrToFile(Join(res, " ", "\n"), path);
}
} // namespace ml
