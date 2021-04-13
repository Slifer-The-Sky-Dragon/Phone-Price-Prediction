BUILD_DIR = build
EXECUTABLE_FILE = PhonePricePrediction.out

CC = g++
CFLAGS = -std=c++11
LINK_FLAGS = -I $(INCLUDE_DIR) -I $(INLINE_DIR) -pthread

.PHONY: all make_build_dir build_objects clean

all: make_build_dir build_objects

OBJECTS = \
	$(BUILD_DIR)/main.o \

make_build_dir:
	mkdir -p $(BUILD_DIR)

build_objects: $(OBJECTS)
	$(CC) $(CFLAGS) $(LINK_FLAGS) $(OBJECTS) -o $(EXECUTABLE_FILE)

clean:
	rm -rf $(BUILD_DIR) *.o *.out

$(BUILD_DIR)/main.o: main.cpp
	$(CC) $(CFLAGS) $(LINK_FLAGS) -c main.cpp -o $(BUILD_DIR)/main.o