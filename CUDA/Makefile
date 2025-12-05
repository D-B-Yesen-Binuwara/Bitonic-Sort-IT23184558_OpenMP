NVCC = nvcc
CFLAGS = -O2
TARGET = bitonicCUDA
SOURCE = bitonicCUDA.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 1024 256

.PHONY: all clean run