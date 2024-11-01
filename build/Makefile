CC = clang

CFLAGS = -fopenmp -lstdc++

OPENCV_INCLUDE = /usr/local/Cellar/opencv/4.10.0_1/include/opencv4
OPENCV_LIB = /usr/local/Cellar/opencv/4.10.0_1/lib

INCLUDE_PATH = /usr/local/opt/libomp/include/
LIBRARY_PATH = /usr/local/opt/libomp/lib/

TARGET = main.exe

SRC = ../src/main.cpp ../src/utils.cpp ../src/test.cpp

$(TARGET): $(SRC)
	CPATH=$(INCLUDE_PATH):$(OPENCV_INCLUDE) LIBRARY_PATH=$(LIBRARY_PATH):$(OPENCV_LIB) $(CC) $(SRC) -o $(TARGET) $(CFLAGS) -lopencv_core -lopencv_imgcodecs -lopencv_highgui

clean:
	rm -f $(TARGET)
