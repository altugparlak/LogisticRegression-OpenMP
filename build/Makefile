CC = clang

CFLAGS = -fopenmp -lstdc++

HEADERS_INCLUDE = $(shell pwd)/../include

OPENCV_INCLUDE = $(shell brew --prefix opencv)/include/opencv4
OPENCV_LIB = $(shell brew --prefix opencv)/lib

INCLUDE_PATH = $(shell brew --prefix libomp)/include
LIBRARY_PATH = $(shell brew --prefix libomp)/lib

TARGET = main.exe

SRC = ../src/main.cpp ../src/utils.cpp ../tests/test.cpp

INCLUDE_FLAGS = -I$(HEADERS_INCLUDE) -I$(OPENCV_INCLUDE) -I$(INCLUDE_PATH)

$(TARGET): $(SRC)
	CPATH=$(INCLUDE_PATH):$(OPENCV_INCLUDE) LIBRARY_PATH=$(LIBRARY_PATH):$(OPENCV_LIB) \
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(INCLUDE_FLAGS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui

clean:
	rm -f $(TARGET)
