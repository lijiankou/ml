#! /bin/sh -f
Include="$HOME/lib/include"
Lib="$HOME/lib/lib:"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/base_unittest.o
  "
exec $cmd
