#! /bin/sh -f
Lib="$HOME/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.00000001
  --tbm_lambda=0.05
  --m=2000
  --k=1
  --feature_size=173
  --it_num=4000
  --type=tbm
  --train_path=data/movielen_binary/train
  --train_path=data/movielen_binary/tbm_task1_train
  --train_path=data/movielen_binary/tbm_task3_train
  --test_path=data/movielen_binary/tbm_task1_test
  --test_path=data/movielen_binary/test
  --feature_path=data/actor_list
  --negative_pro=0.04
  --tbm_task1_groc=./data/roc/tbm_task1_groc
  --tbm_task2_groc=./data/roc/tbm_task2_groc
  --tbm_task3_groc=./data/roc/tbm_task3_groc
  --tbm_task1_iroc=./data/roc/tbm_task1_iroc
  --tbm_task2_iroc=./data/roc/tbm_task2_iroc
  --tbm_task3_iroc=./data/roc/tbm_task3_iroc
  --task1_flag=false
  --task2_flag=true
  --task3_flag=false
  "
gdb="
  gdb ./makeout/rbm_app.o
  "
exec $cmd
#exec $gdb
