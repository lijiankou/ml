// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#include "ml/kmean.h"
#include "ml/eigen.h"

void EStep(const EMat &data, const VVInt &cluster, EMat* center) {
  for (size_t i = 0; i < cluster.size(); i++) {
    EVec vec(data.rows());
    vec.setZero();
    for (size_t j = 0; j < cluster[i].size(); j++) {
      vec += data.col(cluster[i][j]);
    }
    vec /= cluster[i].size();
    center->col(i) = vec;
  }
}

void MStep(const EMat &data, const EMat &center, VVInt* des) {
  des->resize(center.cols());
  for (int i = 0; i < data.cols(); i++) {
    double min = 100000000;
    size_t min_pos = 0;
    for (int j = 0; j < center.cols(); j++) {
      double t = (data.col(i) - center.col(j)).cwiseAbs2().sum();
      if (t < min) {
        min_pos = j; 
        min = t;
      }
    }
    des->at(min_pos).push_back(i);
  } 
}

double Error(const EMat &data, const EMat &center, VVInt &cluster) {
  double sum = 0;
  for (size_t i = 0; i < cluster.size(); i++) {
    for (size_t j = 0; j < cluster[i].size(); j++) {
      EVec t = data.col(cluster[i][j]) - center.col(i);
      sum += t.cwiseAbs2().sum();
    }
  } 
  return sum;
}

void KMean(const EMat &data, int k, int iter, VVInt* cluster) {
  EMat center(data.rows(), k);
  for (int i = 0; i < center.cols(); ++i) {
    center.col(i) = data.col(i);
  }
  for (int i = 0; i < iter; i++) {
    cluster->clear();
    MStep(data, center, cluster);
    EStep(data, *cluster, &center);
    //LOG(INFO) << Error(data, center, *cluster);
  }
  LOG(INFO) << Join(center);
}
