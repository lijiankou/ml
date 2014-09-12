#! /bin/sh -f
Lib="$HOME/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/rbm_app.o
  --eta=0.05
  --bach_size=200
  --m=2000
  --k=1
  --nCD=1
  --lambda_weight=1
  --lambda_feature=20
  --hidden=100
  --feature_size=173
  --it_num=4000
  --type=softmax
  --type=softmax_crbm
  --type=crbm
  --type=tbm
  --type=gaussian_rbm
  --type=rating_rbm
  --type=bin_rbm
  --train_path=data/small_data/train
  --test_path=data/small_data/test
  --train_path=data/cold_train
  --test_path=data/cold_test
  --train_path=data/movielen_data/u2.base
  --test_path=data/movielen_data/u2.test
  --train_path=../../data/ali/train
  --test_path=../../data/ali/test
  --feature_path=data/actor_list
  --train_path=data/movielen_binary/train
  --test_path=data/movielen_binary/test
  "
gdb="
  gdb ./makeout/rbm_app.o
  "
#exec $cmd
exec $gdb
