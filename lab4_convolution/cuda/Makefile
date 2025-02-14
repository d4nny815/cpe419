NVCC = nvcc

NVCC_FLAGS = -g -arch=sm_61 -Wno-deprecated-gpu-targets 

TARGET = convolution

SOURCES = $(TARGET).cu

OBJECTS = $(SOURCES:.cu=.o)

.PHONY: clean run log prof debug
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(OBJECTS)
	@echo "CUDA compilation complete: $(TARGET) created"

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Clean complete"

run: $(TARGET)
	./$(TARGET)

log: $(TARGET)
	@touch results.csv
	@echo "Nx, Ny, N, time (sec)"
	@./$(TARGET) 1000 1000 >> results.csv
	@./$(TARGET) 5000 5000 >> results.csv
	@./$(TARGET) 10000 10000 >> results.csv
	@./$(TARGET) 20000 50000 >> results.csv
	@./$(TARGET) 2000 500000 >> results.csv
	@./$(TARGET) 50000 50000 >> results.csv
	@./$(TARGET) 10000 100000 >> results.csv
	@./$(TARGET) 15000 100000 >> results.csv
	@./$(TARGET) 1000000 1000000 >> results.csv

prof: $(TARGET)
	# nvprof ./$(TARGET) 100000 100000> /dev/null
	nvprof ./$(TARGET) 10000 10000

debug:
	$(NVCC) $(NVCC_FLAGS) -G -g -o $(TARGET) $(OBJECTS)
	cuda-gdb ./$(TARGET)
