BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/lda

name:=lda_unittest
$(name)_update_dep:=\
  lda_var_em.o\
  document.o\
  lda_model.o\
  lda_gibbs.o\
  string_util.o
$(name)_remain_dep:=\
  lda.o\
  base.o\
  util.o\
  cokus.o\
  join.o\
  $(GTestDir)/gtest-all.o
include post_unittest_makefile

name:=lda_app
$(name)_update_dep:=\
  lda_var_em.o\
  document.o\
  lda_model.o\
  lda_gibbs.o\
  string_util.o
$(name)_remain_dep:=\
  lda.o\
  base.o\
  util.o\
  cokus.o\
  join.o
include post_app_makefile

name:=lda_unittest
name:=lda_app
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)

