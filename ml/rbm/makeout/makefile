BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/rbm

name:=rbm_unittest
$(name)_update_dep:=\
  crbm.o
$(name)_remain_dep:=\
  eigen.o\
  tbm.o\
  rbm.o\
  kmean.o\
  rbm_bin.o\
  softmax_crbm.o\
  rbm_gaussian.o\
  rbm_util.o\
  ais.o\
  join.o\
  util.o\
  repsoftmax.o\
  probability.o\
  string_util.o\
  document.o\
  base.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=ali_app
name:=rbm_app
$(name)_update_dep:=\
  bm_matrix.o
$(name)_remain_dep:=\
  crbm.o\
  rbm_gaussian.o\
  rbm_bin.o\
  eigen.o\
  tbm.o\
  rbm.o\
  kmean.o\
  softmax_crbm.o\
  rbm_util.o\
  ais.o\
  join.o\
  util.o\
  repsoftmax.o\
  probability.o\
  string_util.o\
  document.o\
  base.o
include post_app_makefile

name:=hrbm_app
$(name)_update_dep:=\
  rbm_gaussian.o\
  hrbm.o
$(name)_remain_dep:=\
  rbm_util.o\
  kmean.o\
  eigen.o\
  join.o\
  util.o\
  probability.o\
  string_util.o\
  document.o\
  base.o
include post_app_makefile


name:=rbm_unittest
name:=hrbm_app
name:=ali_app
name:=rbm_app
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
