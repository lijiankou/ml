#! /bin/sh -f
Lib="$HOME/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.01
  --bach_size=200
  --m=2000
  --k=1
  --nCD=1
  --lambda_weight=0.5
  --lambda_feature=120
  --hidden=110
  --feature_size=173
  --it_num=200
  --type=crbm
  --train_path=data/cold_train
  --test_path=data/cold_test
  --train_path=data/movielen_data/u2.base
  --test_path=data/movielen_data/u2.test
  --test_path=data/movielen_binary/tbm_task1_test
  --train_path=data/movielen_binary/crbm_task1_train
  --train_path=data/movielen_binary/crbm_task2_train
  --train_path=data/movielen_binary/train
  --test_path=data/movielen_binary/crbm_task1_test
  --test_path=data/movielen_binary/test
  --crbm_task1_groc=data/roc/crbm/crbm_task1_groc
  --crbm_task2_groc=data/roc/crbm/crbm_task2_groc
  --crbm_task3_groc=data/roc/crbm/crbm_task3_groc
  --crbm_task1_iroc=data/roc/crbm/crbm_task1_iroc
  --crbm_task2_iroc=data/roc/crbm/crbm_task2_iroc
  --crbm_task3_iroc=data/roc/crbm/crbm_task3_iroc
  --feature_path=data/actor_list
  --task1_flag=false
  --task2_flag=false
  --task3_flag=true
  --save_start_iter=200
  "
gdb="
  gdb ./makeout/rbm_app.o
  "
exec $cmd
#exec $gdb
