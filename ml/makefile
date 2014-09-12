BaseDir:=../
cmd:=\
  ctags -R *

.PHONY:main
main:
	$(cmd); cd makeout; make clean; make

name:=./run/info_unittest.sh
name:=./run/eigen_app.sh
name:=./run/util_unittest.sh
name:=./run/mf_app.sh
name:=./run/document_unittest.sh
name:=./run/eigen_unittest.sh
.PHONY:run
run:
	$(name)
