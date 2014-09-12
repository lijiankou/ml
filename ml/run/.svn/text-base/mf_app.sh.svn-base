#! /bin/sh -f
Include="$HOME/google-library/include"
Lib="$HOME/google-library/lib:$HOME/lib/gmp/lib:/home/lijk/lib/lib/"
root_path="."
RootPath=$root_path"/experiment/data/benchmark"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./makeout/mf_app.o
  --train_path=libsvm/data/train
  --test_path=libsvm/data/test
  --train_path=rbm/data/movielen_binary/u2.base
  --test_path=rbm/data/movielen_binary/u2.test
  --train_path=rbm/data/movielen_data/u4.base
  --test_path=rbm/data/movielen_data/u4.test
  --k=20
  --lambda=0.1
  --eta=0.001
  --flag_mf=false
  --flag_mfbias=true
  "
exec $cmd
