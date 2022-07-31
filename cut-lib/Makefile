CC = g++
CFLAGS = -std=c++20 -O3 -Wall -Wextra -Wpedantic

.DEFAULT_GOAL := include/cut-seg.so

bin/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c -fPIC -o $@ $^

include/cut-seg.so: src/cut-seg.cpp bin/segmenter.o
	$(CC) -shared -fPIC $(CFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f bin/*.o include/*.so
