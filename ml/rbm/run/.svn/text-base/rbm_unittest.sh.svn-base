#! /bin/sh -f
Lib="$HOME/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_unittest.o
  --gtest_filter=EigenRBMTest.MovieLenTest
  --gtest_filter=RepSoftmaxTest.AisTest
  --gtest_filter=AisTest.UniformSampleTest
  --gtest_filter=Ais.AisTest
  --gtest_filter=Ais.WAisTest
  --gtest_filter=Ais.LogPartitionTest
  --gtest_filter=ROC.AUCTest
  "
gdb="
  gdb ./makeout/rbm_unittest.o
  "

#exec $gdb
exec $cmd
