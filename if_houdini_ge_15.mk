# Check if houdini version is greater than or equal to 15
hou_ge_15 := $(shell test $(HOUDINI_MAJOR_RELEASE) -ge 15 && echo 'true')
ifeq ($(hou_ge_15),true)
  GCC_HOME:=/tools/package/gcc/$(GCC_VERSION)
  CXX:=$(GCC_HOME)/bin/g++
  CXXFLAGS += -std=c++11
  #GCC_LIB_DIR := $(ICC_HOME)/compiler/lib/intel64
  #let the makefile know we're using icc
  using_newer_gcc := yes
endif
