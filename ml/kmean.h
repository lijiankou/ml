// Copyright 2014 lijiankou. All Rights Reserved.
// author: lijk_start@163.com (jiankou li)
#ifndef ML_KMAEN_H_
#define ML_KMAEN_H_
#include "base/base_head.h"
#include "ml/eigen.h"

void KMean(const EMat &data, int k, int iter, VVInt* cluster);

#endif // ML_KMAEN_H_
