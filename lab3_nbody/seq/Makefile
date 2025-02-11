TARGET = nbody

CC = gcc

CFLAGS = -Wall -O3 -fopenmp
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
	./$(TARGET) 100 5

log: $(TARGET)
	./$(TARGET) 10000 10 | sort > log10
	./$(TARGET) 30000 10 | sort > log30
	./$(TARGET) 100000 10 | sort > log100
	./$(TARGET) 500000 10 | sort > log500
	./$(TARGET) 1000000 10 | sort > log1000 &

clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Clean complete"
