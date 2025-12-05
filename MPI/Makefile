CC = mpicc
CFLAGS = -O2 -Wall
TARGET = bitonicMPI_fixed
SOURCE = bitonicMPI_fixed.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	mpirun -np 4 ./$(TARGET) 1024

.PHONY: all clean run