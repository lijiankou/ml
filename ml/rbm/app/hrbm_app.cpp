// Copyright 2014 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#include "base/base_head.h"
#include "ml/rbm/hrbm.h"
#include "ml/rbm/rbm_util.h"
#include "ml/util.h"
#include "ml/rbm/roc.h"
#include "ml/document.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>

DEFINE_int32(bach_size, 100, "bach size");
DEFINE_int32(k, 5, "class size");
DEFINE_int32(m, 2000, "visual size");
DEFINE_int32(it_num, 1000, "iter number");
DEFINE_int32(algorithm_type, 1, "iter number");

DEFINE_int32(feature_size, 1000, "feature number");

DEFINE_string(train_path, "tmp/train1", "");
DEFINE_string(test_path, "tmp/test1", "");
DEFINE_string(feature_path, "tmp/test1", "feature path");

DEFINE_int32(nCD, 1, "step size of CD");

DEFINE_double(alpha, 0.01, "learning rate");
DEFINE_int32(hidden_user, 100, "learning rate");
DEFINE_int32(hidden_item, 100, "learning rate");

namespace ml {
void HRBMApp(const Str &train_path, const Str &test_path, const Str &f_path) {
  SpMat u_v;
  ReadData(train_path, 0, 0, &u_v);
  LOG(INFO) << "load train over";
  if (!CheckData(u_v, FLAGS_k)) {
    LOG(INFO) << "rating scale is wrong, please check and reset k";
    assert(false);
  }
  SpMat v_u = u_v.transpose();
  SpMat test_u_v;
  LOG(INFO) << FLAGS_test_path;
  ReadData(test_path, u_v.rows(), u_v.cols(), &test_u_v);
  LOG(INFO) << "load test over";
  SpMat test_v_u = test_u_v.transpose();
  //read data over
 
  int user_num = u_v.cols();
  int item_num = u_v.rows();
  HRBM hrbm(user_num, item_num, FLAGS_hidden_user, FLAGS_hidden_item);
  hrbm.LoadData(u_v, test_u_v, v_u, test_v_u);
  hrbm.Train(FLAGS_it_num, FLAGS_alpha);
}
} // namespace ml

int main(int argc, char* argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  ml::HRBMApp(FLAGS_train_path, FLAGS_test_path, FLAGS_feature_path);
  return  0;
}
