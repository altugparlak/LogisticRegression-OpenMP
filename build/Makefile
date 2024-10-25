CC = clang

CFLAGS = -fopenmp -lstdc++

INCLUDE_PATH = /usr/local/opt/libomp/include/
LIBRARY_PATH = /usr/local/opt/libomp/lib/

TARGET = main.exe

SRC = ../src/main.cpp

$(TARGET): $(SRC)
	CPATH=$(INCLUDE_PATH) LIBRARY_PATH=$(LIBRARY_PATH) $(CC) $(SRC) -o $(TARGET) $(CFLAGS)

clean:
	rm -f $(TARGET)
