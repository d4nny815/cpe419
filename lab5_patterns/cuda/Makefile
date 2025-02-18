NVCC = nvcc

NVCC_FLAGS = -g -arch=sm_61 -Wno-deprecated-gpu-targets 

TARGET = reduction

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
	@echo "N, sumNeighbors, tNeighbors, sumStride, tStride" > results.csv
	@./$(TARGET) 10000 >> results.csv
	@./$(TARGET) 50000 >> results.csv
	@./$(TARGET) 100000 >> results.csv
	@./$(TARGET) 1000000 >> results.csv
	@./$(TARGET) 10000000 >> results.csv
	@./$(TARGET) 100000000 >> results.csv
	@./$(TARGET) 250000000 >> results.csv
	@./$(TARGET) 250000001 >> results.csv

prof: $(TARGET)
	# nvprof ./$(TARGET) > /dev/null
	nvprof ./$(TARGET) 1000000

debug:
	$(NVCC) $(NVCC_FLAGS) -G -g -o $(TARGET) $(OBJECTS)
	cuda-gdb ./$(TARGET)
