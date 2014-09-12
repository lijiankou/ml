#! /bin/sh -f
Include="$HOME/lib/include"
Lib="$HOME/lib/lib"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rtm_app.o
  --net_path=rtm_network
  --cor_path=rtm_corpus
  --alpha=0.01
  --topic_num=10
  "

gdb="
  gdb ./makeout/rtm_app.o
  "
#exec $gdb
exec $cmd
