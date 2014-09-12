#! /bin/sh -f
Lib="$HOME/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/hrbm_app.o
  --alpha=0.1
  --m=2000
  --k=5
  --nCD=1
  --lambda_weight=1
  --hidden_user=60
  --hidden_item=60
  --it_num=1000
  --train_path=data/movielen_binary/u2.base
  --test_path=data/movielen_binary/u2.test
  --train_path=data/movielen_data/u2.base
  --test_path=data/movielen_data/u2.test
  "
gdb="
  gdb ./makeout/hrbm_app.o
  "
exec $cmd
#exec $gdb
