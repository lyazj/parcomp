all: redist prog

clean:
	$(RM) redist prog

.PHONY: all clean

%: %.c
	mpicc $^ -o $@ -lm
