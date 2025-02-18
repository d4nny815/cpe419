TARGET = reduction

CC = gcc

CFLAGS = -Wall -O3 -std=gnu99 -fopenmp -g
LFLAGS = -lm

SOURCES = $(TARGET).cu

OBJECTS = $(SOURCES:.cu=.o)

.PHONY: run log clean
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LFLAGS)
	@echo "CUDA compilation complete: $(TARGET) created"

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@ $(LFLAGS)

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

clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Clean complete"
