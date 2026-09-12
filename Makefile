BIN:=bin
LIB:=lib
INC:=include
SRC:=src
CFLAGS=-std=c17 -g -Wall -I$(INC) -O2

SRC_FILES:=$(wildcard $(SRC)/*.c)
OBJ_PATHS=$(patsubst src/%.c, $(LIB)/%.o, $(SRC_FILES))
TARGET=$(BIN)/nn

all: $(TARGET)

install:
	tar -xzf TrainingData/data.tar.gz -C TrainingData/

$(TARGET): $(OBJ_PATHS)
	$(CC) $(CFLAGS) $^ -o $@

$(LIB)/%.o: $(SRC)/%.c 
	$(CC) $(CFLAGS) -c -o $@ $< 

clean:
	rm -f $(OBJ_PATHS) TrainingData/*.bin
