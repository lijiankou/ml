BaseDir:=../../..
include build_makefile
WorkDir:=$(BaseDir)/ml/topic

name:=rtm_app
$(name)_update_dep:=\
  rtm.o\
  document.o\
  rtm_var_em.o
$(name)_remain_dep:=\
  eigen.o\
  string_util.o\
  base.o\
  util.o\
  cokus.o\
  join.o
include post_app_makefile

name:=rtm_app
include main_makefile

include update

.PHONY:clean
clean:
	rm $(clean_file)
