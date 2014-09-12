// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#include "ml/rbm/crbm.h"

#include "base/base_head.h"
#include "ml/rbm/rbm_util.h"
#include "ml/roc.h"
#include "ml/kmean.h"
#include "ml/eigen.h"
#include "ml/util.h"

DEFINE_int32(save_start_iter, 200, "");

DECLARE_double(lambda_weight);
DEFINE_double(lambda_feature, 1, "");

DEFINE_string(crbm_task1_groc, "", "");
DEFINE_string(crbm_task2_groc, "", "");
DEFINE_string(crbm_task3_groc, "", "");
DEFINE_string(crbm_task1_iroc, "", "");
DEFINE_string(crbm_task2_iroc, "", "");
DEFINE_string(crbm_task3_iroc, "", "");
DECLARE_bool(task1_flag);
DECLARE_bool(task2_flag);
DECLARE_bool(task3_flag);

namespace ml {
//w.rows(), the hidden feature dimension
//w.cols(), the visual rating dimension,
//u.rows(), the hidden feature dimension
//u.cols(), the visual feature dimension,
CRBM::CRBM(int n_feature, int nv, int nh) {
   assert(n_feature != 0);
   assert(nv != 0);
   assert(nh != 0);
   LOG(INFO) << "nv:" << nv << " nh:" << nh;
   w.resize(nh, nv);
   NormalRandom(&w);
   dw.resize(nh, nv);

   u.resize(nh, n_feature);
   NormalRandom(&u);
   du.resize(nh, n_feature);

   bv.resize(nv);
   NormalRandom(&bv);
   dv.resize(nv);

   bf.resize(nv);
   NormalRandom(&bf);
   df.resize(nv);

   bh.resize(nh);
   bh.setZero();
   dh.resize(nh);

   v0.resize(nv);
   h0.resize(nh);
   f0.resize(n_feature);
   expect_h0.resize(nh);
   vk.resize(nv);
   hk.resize(nh);
   fk.resize(n_feature);
   LOG(INFO) << "over";
}

void CRBM::InitW() {
   NormalRandom(&w);
   NormalRandom(&bv);
   bh.setZero();
}

void CRBM::InitFeature() {
   NormalRandom(&u);
   NormalRandom(&bf);
}

void CRBM::ExpectV(const EVec &h, const SpVec &t, VReal* des) {
  for (SpVec::InnerIterator it(t); it; ++it){
    des->push_back(Sigmoid(bv(it.index()) +  w.col(it.index()).dot(h)));
  }
}

void CRBM::ExpectV(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = vec[i];
  }
}

//this function is not effieicy, vec should be defined before
void CRBM::SampleV(const EVec &h, const SpVec &t, SpVec *v) {
  v->setZero();
  VReal vec;
  ExpectV(h, t, &vec);
  int i = 0;
  for (SpVec::InnerIterator it(t); it; ++it, ++i){
    v->insert(it.index()) = Sample1(vec[i]);
  }
}

void CRBM::ExpectH(const SpVec &v, const SpVec &f, EVec *h) {
  for (int i = 0; i < h->rows(); ++i) {
    double s = 0;
    for (SpVec::InnerIterator it(v); it; ++it) {
      s += w(i, it.index()) * it.value();
    }
    for (SpVec::InnerIterator it(f); it; ++it) {
      s += u(i, it.index()) * it.value();
    }
    (*h)[i] = Sigmoid(s + bh[i]);
  }
}

void CRBM::SampleH(const SpVec &v, const SpVec &f, EVec *h) {
  ExpectH(v, f, h);
  ::Sample(h);
}

void CRBM::PartGrad(const SpVec &v, const SpVec &f, const EVec &h, double coe) {
  for (SpVec::InnerIterator it(v); it; ++it) {
    dv(it.index()) += coe * it.value();
    for (int j = 0; j < h.rows(); ++j) {
      dw(j, it.index()) += coe * h[j] * it.value();
    }
  }
  for (SpVec::InnerIterator it(f); it; ++it) {
    for (int j = 0; j < h.rows(); ++j) {
      du(j, it.index()) += coe * h[j] * it.value();
    }
  }
  for (int j = 0; j < h.rows(); ++j) {
    dh(j) += coe * h[j];
  }
}

void CRBM::InitGradient(){
  du.setZero();
  dw.setZero();
  dh.setZero();
  dv.setZero();
}

void CRBM::UpdateGradient(double alpha, int batch_size) {
  double r = alpha / batch_size;
  w += r * (dw + w * FLAGS_lambda_weight);
  u += r * (du + u * FLAGS_lambda_feature);
  bh += r * dh;
  bv += r * dv;
}

void CRBM::PreUpdateGradient(double alpha, int batch_size) {
  UpdateGradient(alpha, batch_size);
  double r = alpha / batch_size;
  bf += r * df;
}

void CRBM::Gradient(const SpVec &v0, const SpVec &f, int step) {
  ExpectH(v0, f, &expect_h0);
  ::Sample(expect_h0, &h0);
  SampleV(h0, v0, &vk);
  for (int k = 0; k < step - 1; ++k) {
    SampleH(vk, f, &hk);
    SampleV(hk, v0, &vk);
  }
  ExpectH(vk, f, &hk);
  PartGrad(v0, f, expect_h0, 1); //positive phase
  PartGrad(vk, f, hk, -1); //negative phase
}

void CRBM::PreGradient(const SpVec &v0, const SpVec &f0, int step) {
  ExpectH(v0, f0, &expect_h0);
  ::Sample(expect_h0, &h0);
  SampleV(h0, v0, &vk);
  SampleV(h0, f0, &fk);
  for (int k = 0; k < step - 1; ++k) {
    SampleH(vk, fk, &hk);
    SampleV(hk, v0, &vk);
    SampleV(hk, f0, &fk);
  }
  ExpectH(vk, fk, &hk);
  PartGrad(v0, f0, expect_h0, 1); //positive phase
  PartGrad(vk, fk, hk, -1); //negative phase
}

void CRBM::PreTrain(const SpMat &train, const SpMat &feature, const SpMat &test,
                                        int niter, double alpha, int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = 1;
  for (int i = 0; i < niter; ++i) {
    for (int n = 1; n < train.cols(); n++) {
      PreGradient(train.col(n), feature.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        PreUpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    double pre_error = Predict(train, feature, test);
    LOG(INFO) << i << " " << Predict(train, feature, train) << " " << pre_error;
  }
}

//it's important using expect and not use sample
double CRBM::Predict(const SpMat &train, const SpMat &f, const SpMat &test) {
  double rmse = 0;
  for(int n = 0; n < train.cols(); n++) {
    h0.setZero();
    v0.setZero();
    if (f.col(n).size() != 0) {
      ExpectH(train.col(n), f.col(n), &h0);
      ExpectV(h0, test.col(n), &v0);
      v0 -= test.col(n);
      rmse += v0.cwiseAbs2().sum();
    }
  }
  return sqrt(rmse/test.nonZeros());
}

//it's important using expect and not use sample
void CRBM::Task3IROC(const SpMat &train, const SpMat &f, const SpMat &test) {
  SpMat pre(train.rows(), train.cols());
  for(int n = 0; n < train.cols(); n++) {
    v0.setZero();
    h0.setZero();
    if (f.col(n).size() != 0) {
      ExpectH(train.col(n), f.col(n), &h0);
      ExpectV(h0, test.col(n), &v0);
      pre.col(n) = v0;
    }
  }
  VVInt res;
  CROC(pre, test, &res);
  WriteStrToFile(Join(res, " ", "\n"), FLAGS_crbm_task3_iroc);
}

void CreateTask2Test(const SpVec &vec, SpVec* des) {
  MIntReal dic;
  for (SpVec::InnerIterator it(vec); it; ++it) {
    dic.insert(std::make_pair(it.index(), it.value()));
  }
  for (int i = 1; i < vec.size(); i++) {
    MIntReal::iterator it = dic.find(i);
    if (it == dic.end()) {
      des->insert(i) = 0;
    } else {
      des->insert(i) = it->second;
    }
  }
}

double CRBM::Task2GROC(const SpMat &train, const SpMat &f, const SpMat &test,
                       VVReal *res) {
  for (SInt::iterator it = cold_dic.begin(); it != cold_dic.end(); ++it) {
    ExpectH(train.col(*it), f.col(*it), &h0);
    SpVec t;
    CreateTask2Test(test.col(*it), &t);
    ExpectV(h0, t, &v0);
    for (SpVec::InnerIterator it_r(t), it_p(v0); it_r; ++it_r, ++it_p) {
      (*res)[0].push_back(it_r.value() + 1);
      (*res)[1].push_back(it_p.value());
    }
  }
  return AUC(res->at(0), res->at(1));
  //WriteStrToFile(Join(res, " ", "\n"), path);
}

//it's important using expect and not use sample
void CRBM::Task2IROC(const SpMat &train, const SpMat &f, const SpMat &test,
                                                         const Str &path) {
  SpMat pre(train.rows(), train.cols());
  SpMat real(train.rows(), train.cols());
  for (SInt::iterator it = cold_dic.begin(); it != cold_dic.end(); ++it) {
    ExpectH(train.col(*it), f.col(*it), &h0);
    SpVec t;
    CreateTask2Test(test.col(*it), &t);
    ExpectV(h0, t, &v0);
    real.col(*it) = t;
    pre.col(*it) = v0;
  }
  VVInt res;
  CROC(pre, real, &res);
  WriteStrToFile(Join(res, " ", "\n"), path);
}

void CRBM::Visual() {
  VVInt c;
  int k = 10;
  int iter = 5000;
  KMean(u, k, iter, &c);
  LOG(INFO) << Join(c, " ", "\n");
  WriteStrToFile(Join(c, " ", "\n"), "data/actor_community");
}

void CRBM::VisualW() {
  VVInt c;
  int k = 10;
  int iter = 5000;
  KMean(w, k, iter, &c);
}

void CRBM::LoadColdMovie(const SpMat &test) {
  for (int i = 1; i < test.cols(); i++) {
    for (SpMat::InnerIterator it(test, i); it; ++it) {
      cold_dic.insert(it.index());
    } 
  }
}

double CRBM::Task3GROC(const SpMat &train, const SpMat &f, const SpMat &test,
                     VVReal* res) {
  for(int n = 1; n < train.cols(); n++) {
    if (f.col(n).size() == 0) {
      continue;
    }
    ExpectH(train.col(n), f.col(n), &h0);
    ExpectV(h0, test.col(n), &v0);
    SpVec::InnerIterator it_p(v0);
    for (SpMat::InnerIterator it_r(test, n); it_r; ++it_r, ++it_p) {
      (*res)[0].push_back(int(it_r.value()) + 1);
      (*res)[1].push_back(it_p.value());
    }
  }
  return AUC((*res)[0], (*res)[1]);
}

void CRBM::Train(const SpMat &train, const SpMat &feature, const SpMat &test,
                 int niter, double alpha, int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = 1;
  double auc_max = 0;
  for (int i = 0; i < niter; ++i) {
    for (int n = 1; n < train.cols(); n++) {
      Gradient(train.col(n), feature.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    double pre_error = Predict(train, feature, test);
    double train_error = Predict(train, feature, train);
    LOG(INFO) << i << " " << train_error << " " << pre_error;

    // if (pre_error < min_error && i > 350) {
    if (i > FLAGS_save_start_iter) {
      if (FLAGS_task1_flag) {
        VVReal task1(2);
        double auc = Task2GROC(train, feature, test, &task1);
        LOG(INFO) << auc;
        if (auc_max < auc) {
          auc_max = auc;
          WriteStrToFile(Join(task1, " ", "\n"), FLAGS_crbm_task1_groc);
        }
        //Task2IROC(train, feature, test, FLAGS_crbm_task1_iroc);
      }
      if (FLAGS_task2_flag) {
        VVReal task2(2);
        double auc = Task2GROC(train, feature, test, &task2);
        LOG(INFO) << auc;
        if (auc_max < auc) {
          auc_max = auc;
          WriteStrToFile(Join(task2, " ", "\n"), FLAGS_crbm_task2_groc);
        }
        //Task2IROC(train, feature, test, FLAGS_crbm_task2_iroc);
      }
      if (FLAGS_task3_flag) {
        VVReal task3(2);
        double auc = Task3GROC(train, feature, test, &task3);
        LOG(INFO) << auc << " " << auc_max;
        if (auc_max < auc) {
          auc_max = auc;
          WriteStrToFile(Join(task3, " ", "\n"), FLAGS_crbm_task3_groc);
        }
        //Task3IROC(train, feature, test);
      }
    }
  }
}

//it's important using expect and not use sample
void CRBM::PrecisionRecall(const SpMat &train, const SpMat &f, const SpMat &test,
                           VVReal* res) {
  for(int n = 0; n < train.cols(); n++) {
    h0.setZero();
    v0.setZero();
    if (f.col(n).size() != 0) {
      ExpectH(train.col(n), f.col(n), &h0);
      ExpectV(h0, test.col(n), &v0);
      SpMat::InnerIterator it2(test, n);
      for (SpVec::InnerIterator it(v0); it; ++it, ++it2) {
        res->at(0).push_back(it2.value() + 1);
        res->at(1).push_back(it.value());
      }
    }
  }
}

void CRBM::Train2(const SpMat &train, const SpMat &feature, const SpMat &t_f,
                  const SpMat &test, int niter, double alpha, int batch_size) {
  InitGradient();
  int curr_samples = 0;
  int nCD = 1;
  for (int i = 0; i < niter; ++i) {
    for (int n = 1; n < train.cols(); n++) {
      Gradient(train.col(n), feature.col(n), nCD);
      curr_samples++;
      if(curr_samples == batch_size) {
        UpdateGradient(alpha, batch_size);
        curr_samples = 0;
        InitGradient();
      }
    }
    if (i > FLAGS_save_start_iter) {
      VVReal v(2);
      PrecisionRecall(train, feature, test, &v);
      double p = DecisionPro(v[0], v[1]);
      LOG(INFO) << i << " " << F1Score(v[0], v[1]) << " " << p;
    } else {
      LOG(INFO) << i;
    }
  }
}
} // namespace ml
