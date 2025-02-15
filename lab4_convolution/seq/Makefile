TARGET = convolution

CC = gcc

CFLAGS = -Wall -O3 -std=gnu99 -fopenmp
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
	@echo "Nx, Ny, N, time (sec)" > results.csv
	@./$(TARGET) 1000 1000 >> results.csv
	@./$(TARGET) 5000 5000 >> results.csv
	@./$(TARGET) 10000 10000 >> results.csv
	@./$(TARGET) 20000 50000 >> results.csv
	@./$(TARGET) 2000 500000 >> results.csv
	@./$(TARGET) 25000 25000 >> results.csv
	@./$(TARGET) 30000 30000 >> results.csv

clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Clean complete"
