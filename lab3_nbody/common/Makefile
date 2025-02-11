NVCC = nvcc

# GTX 1080: Compute Capability 6.1, suppress warning
NVCC_FLAGS = -g -arch=sm_61 -Wno-deprecated-gpu-targets 

TARGET = system_info

SOURCES = $(TARGET).cu

OBJECTS = $(SOURCES:.cu=.o)

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

prof: $(TARGET)
	nvprof ./$(TARGET)

debug:
	$(NVCC) $(NVCC_FLAGS) -g -o $(TARGET) $(OBJECTS)
	cuda-gdb./$(TARGET)
