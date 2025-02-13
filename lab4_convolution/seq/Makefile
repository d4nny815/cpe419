TARGET = convolution

CC = gcc

CFLAGS = -Wall -O3 -std=gnu99 -g
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
	@echo "Nx, Ny, time (sec)" > results.csv
	./$(TARGET) 1000 1000 >> results.csv
	./$(TARGET) 5000 5000 >> results.csv
	./$(TARGET) 1000 5000 >> results.csv
	./$(TARGET) 10000 10000 >> results.csv
	./$(TARGET) 10000 50000 >> results.csv
	./$(TARGET) 100000 500000 >> results.csv

clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Clean complete"
