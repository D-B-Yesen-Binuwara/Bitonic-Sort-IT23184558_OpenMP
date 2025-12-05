CC = gcc
CFLAGS = -fopenmp -O2 -Wall
TARGET = bitonicOmp02
SOURCE = bitonicOmp02.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 1024 4

.PHONY: all clean run