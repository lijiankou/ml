// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com
#include "ml/rbm/hrbm.h"

#include "base/base_head.h"
#include "ml/rbm/rbm_util.h"
#include "ml/eigen.h"
#include "ml/util.h"

DECLARE_double(lambda_feature);
DECLARE_double(lambda_weight);
namespace ml {
HRBM::HRBM(int user_num, int item_num, int hidden_user, int hidden_item): 
                         rbm_u(item_num, hidden_user),
                         rbm_i(user_num, hidden_item) {
  h0_u.resize(hidden_user, user_num);
  h0_i.resize(hidden_item, item_num);

  v1_u.resize(item_num, user_num);
  v1_i.resize(user_num, item_num);
}

//must using sample instead of expect 
double HRBM::Predict(const SpMat &test_u, const SpMat &test_i) {
  rbm_i.ExpectH(test_i, &h0_i);
  ::Sample(&h0_i);
  rbm_i.ExpectV(test_i, h0_i, &v1_i);

  rbm_u.ExpectH(test_u, &h0_u);
  ::Sample(&h0_u);
  rbm_u.ExpectV(test_u, h0_u, &v1_u);

  v1_u = (v1_u + SpMat(v1_i.transpose())) / 2;

  double rmse = 0;
  for(int n = 1; n < test_u.cols(); n++) {
    SpVec tmp;
    Range(v1_u.col(n), &tmp);
    tmp -= test_u.col(n);
    //rmse += tmp.cwiseAbs2().sum();
    rmse += tmp.cwiseAbs().sum();
  }
  //return sqrt(rmse/test_u.nonZeros());
  return rmse/test_u.nonZeros();
}

void HRBM::LoadData(const SpMat &train_u, const SpMat &test_u,
                      const SpMat &train_i, const SpMat &test_i) {
  this->train_u = &train_u;
  this->test_u = &test_u;
  this->train_i = &train_i;
  this->test_i = &test_i;
}

void HRBM::Train(int niter, double alpha) {
  rbm_u.InitGradient();
  rbm_i.InitGradient();
  for (int i = 0; i < niter; ++i) {
    rbm_i.ExpectH(*train_i, &h0_i);
    ::Sample(&h0_i);
    rbm_i.ExpectV(*train_i, h0_i, &v1_i);

    rbm_u.ExpectH(*train_u, &h0_u);
    ::Sample(&h0_u);
    rbm_u.ExpectV(*train_u, h0_u, &v1_u);

    v1_u = (v1_u + SpMat(v1_i.transpose())) / 2;
    v1_i = v1_u.transpose();

    rbm_u.Gradient(*train_u, h0_u, v1_u);
    rbm_i.Gradient(*train_i, h0_i, v1_i);

    rbm_u.UpdateGradient(alpha, train_u->cols());
    rbm_i.UpdateGradient(alpha, train_i->cols());

    rbm_u.InitGradient();
    rbm_i.InitGradient();
    LOG(INFO) << i << " " << Predict(*train_u, *train_i) << " " << Predict(*test_u, *test_i);
  }
}
} // namespace ml
