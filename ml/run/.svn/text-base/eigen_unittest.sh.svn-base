#! /bin/sh -f
Include="$HOME/lib/include"
Lib="$HOME/lib/lib"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/eigen_unittest.o
  --gtest_filter=Eigen.NormalRandomTest
  --gtest_filter=Eigen.DemoTest
  --gtest_filter=Eigen.ReadDataTest
  --gtest_filter=MatrixFactorization.FengXing
  --gtest_filter=Eigen.StrTest
  --gtest_filter=MatrixFactorization.GradientDescent
  --gtest_filter=MatrixFactorization.GradientDescent2
  --gtest_filter=KMean.KMean
  --gtest_filter=Eigen.SpMat
  --gtest_filter=Eigen.CreateAdjTest
  --gtest_filter=Eigen.Join
  --gtest_filter=Eigen.SampleTest
  --gtest_filter=Eigen.BaseTest
  --gtest_filter=EigenUtil.LogSumTest
  "
exec $cmd

