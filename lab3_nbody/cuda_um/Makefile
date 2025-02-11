NVCC = nvcc

# GTX 1080: Compute Capability 6.1, suppress warning
NVCC_FLAGS = -g -arch=sm_61 -Wno-deprecated-gpu-targets 

TARGET = nbody

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
	./$(TARGET) 100 5

log: $(TARGET)
	./$(TARGET) 10000 | sort > log10
	./$(TARGET) 30000 | sort > log30
	./$(TARGET) 100000 | sort > log100
	./$(TARGET) 1000000 | sort > log1000 & 

prof: $(TARGET)
	nvprof ./$(TARGET) 10000 | sort > log10
	# nvprof ./$(TARGET) 30000 | sort > log30

debug:
	$(NVCC) $(NVCC_FLAGS) -g -o $(TARGET) $(OBJECTS)
	cuda-gdb./$(TARGET)
