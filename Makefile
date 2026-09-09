TARGET:=bin/nn
OBJS:=src/main.o src/matrix.o
CFLAGS:=-Wall -Iinclude

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< 

clean:
	rm -f $(OBJS)
