CC = gcc
CFLAGS = -O2 -Wall
TARGET = bitonic
SOURCE = bitonic.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 1024

.PHONY: all clean run