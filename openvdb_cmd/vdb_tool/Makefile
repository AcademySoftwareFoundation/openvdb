USE_NANO := yes
USE_PNG  := no
USE_EXR  := no
USE_JPG  := no
USE_ABC  := no
# use this with enable all of the above
USE_ALL  := no

CC = g++
ABI_FLAG =# -DOPENVDB_ABI_VERSION_NUMBER=9
CFLAGS = -march=native -std=c++14 -Wall -w
INCLUDES = -I$(HOME)/local/include -I./include
LIBRARY = -L$(HOME)/local/lib -L$(HOME)/local/lib64
LIBS = -lopenvdb -ltbb -ltbbmalloc_proxy -ltbbmalloc -lHalf -lpthread -lm
DEP = Makefile include/*.h

ifeq (yes,$(USE_ALL))
	USE_NANO = yes
    USE_PNG  = yes
    USE_EXR  = yes
    USE_JPG  = yes
    USE_ABC  = yes
endif

ifeq (yes,$(USE_NANO))
	CFLAGS += -DVDB_TOOL_USE_NANO
endif

ifeq (yes,$(USE_PNG))
	CFLAGS += -DVDB_TOOL_USE_PNG
	LIBS += -lpng
endif

ifeq (yes,$(USE_JPG))
	CFLAGS += -DVDB_TOOL_USE_JPG
	LIBS += -ljpeg
endif

ifeq (yes,$(USE_EXR))
	CFLAGS += -DVDB_TOOL_USE_EXR
    LIBS += -lIlmImf
endif

ifeq (yes,$(USE_ABC))
	CFLAGS += -DVDB_TOOL_USE_ABC
	INCLUDES += -I$(HOME)/local/include/OpenEXR
	LIBS += -lAlembic -lImath -lIexMath -lIex -lIlmThread
endif

BUILD_DEBUG   = $(CC) $(CFLAGS) $(ABI_FLAG) -O1 -g
BUILD_RELEASE = $(CC) $(CFLAGS) $(ABI_FLAG) -DNDEBUG -O3

# Avoid associating these build target with files of the same name
.PHONY: all clean test

all: release debug test archive

def: release test

.DEFAULT_GOAL := def

#########

release/main.o: $(DEP) src/main.cpp
	@mkdir -p release
	$(BUILD_RELEASE) $(INCLUDES) -o release/main.o -c src/main.cpp

release: release/main.o
	$(BUILD_RELEASE) $(INCLUDES) $(LIBRARY) -o release/vdb_tool release/main.o $(LIBS)

#########

debug/main.o: $(DEP) src/main.cpp
	@mkdir -p debug
	$(BUILD_DEBUG) $(INCLUDES) -o debug/main.o -c src/main.cpp

debug: debug/main.o
	$(BUILD_DEBUG)  $(INCLUDES) $(LIBRARY) -o debug/vdb_tool debug/main.o $(LIBS)

#########

debug/unittest.o: $(DEP) src/unittest.cpp
	@mkdir -p debug data
	$(BUILD_DEBUG) $(INCLUDES) -o debug/unittest.o -c src/unittest.cpp

test: debug/unittest.o
	$(BUILD_DEBUG)  $(INCLUDES) $(LIBRARY) -o debug/test debug/unittest.o $(LIBS) -lgtest

#########

archive:
	@tar -C .. -czf vdb_tool.tgz vdb_tool/CMakeLists.txt vdb_tool/Makefile vdb_tool/src vdb_tool/include vdb_tool/examples vdb_tool/README.md vdb_tool/CHANGES.md

#########

clean:
	@rm -rf data debug release *.abc *.vdb *.ply *.obj *.stl *.ppm *.png *.exr *.jpg vdb_tool.tgz
