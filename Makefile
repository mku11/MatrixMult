### CC
OUTPUT_DIR=bin
CC?=gcc
EXE_NAME=matmult.exe

##### OPEN CL
# Comment this if you don't have opencl
USE_OPENCL=true
# CFLAGS+= -DCL_TARGET_OPENCL_VERSION=120
# CFLAGS+= -DCL_TARGET_OPENCL_VERSION=200
CFLAGS+= -DCL_TARGET_OPENCL_VERSION=300

# open cl for windows (cygwin)
OPENCL_ROOT=/cygdrive/d/tools/OpenCL-SDK-v2024.05.08-Win-x64
# use the sdk lib, make sure you update LD_LIBRARY_PATH before execution
OPENCL_LIB=$(OPENCL_ROOT)/bin
# use the installed opencl lib
#OPENCL_LIB=/cygdrive/c/windows/system32
OPENCL_INCLUDE=$(OPENCL_ROOT)/include

# uncomment if you build on linux
#OPENCL_LIB=/usr/lib/x86_64-linux-gnu
#OPENCL_INCLUDE=/usr/include

##### DO NOT EDIT below this line
CFLAGS+=-fPIC -g -O3
LIBS=-lm
OBJ_DIR=obj
SRC= src/main.c src/matmult.c src/mat_tools.c
CFLAGS+=-Wno-deprecated-declarations -Wno-unused-variable
SRC_ROOT=src
ifeq ($(USE_OPENCL),true)
	CFLAGS+=-DUSE_OPENCL
	SRC+= src/opencl_matmult.c src/opencl_tools.c
	LDFLAGS+= -L$(OPENCL_LIB)
	CFLAGS+= -I$(OPENCL_INCLUDE)
	LIBS+= -lOpenCL
	_OPENCL_DEPS = CL/opencl.h
	OPENCL_DEPS = $(patsubst %,$(OPENCL_INCLUDE)/%,$(_OPENCL_DEPS))
endif
OBJECT := $(SRC:.c=.o)

.PHONY: all
all: $(EXE_NAME)

%.o: %.c
	@mkdir -p $(subst $(SRC_ROOT),$(OBJ_DIR),$(@D))
	$(CC) -c -o $(subst $(SRC_ROOT),$(OBJ_DIR),$@) $< $(CFLAGS)

$(EXE_NAME): $(OBJECT)
	@mkdir -p $(OUTPUT_DIR)
	$(CC) -o $(OUTPUT_DIR)/$@ $(subst $(SRC_ROOT),$(OBJ_DIR),$^) $(CFLAGS) $(LIBS) $(LDFLAGS)
	
clean:
	rm $(OBJ_DIR)/*.o $(OUTPUT_DIR)/*.exe

run: $(EXE_NAME)
	cd $(OUTPUT_DIR) && time ./$(EXE_NAME)

debug: $(EXE_NAME)
	cd $(OUTPUT_DIR) && gdb ./$(EXE_NAME)

.PHONY: all
