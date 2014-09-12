#! /bin/sh -f
Lib="$HOME/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.001
  --bmmatrix_lambda=3
  --m=2000
  --k=1
  --nCD=1
  --lambda_weight=1
  --hidden=200
  --it_num=10000
  --train_path=data/movielen_binary/u2.base
  --test_path=data/movielen_binary/u2.test
  --type=bmmatrix
  "
gdb="
  gdb ./makeout/hrbm_app.o
  "
exec $cmd
#exec $gdb
