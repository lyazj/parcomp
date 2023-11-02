#include <stdlib.h>

#define G  6.67e-11 
struct FILE_SEG {
	int64_t offset;
	int64_t width;
	int64_t stride;
}; 

int64_t input_data(void *buf, int64_t count, FILE_SEG fseg);
int64_t input_data(void *bufa, int64_t counta, FILE_SEG fsega, void *bufb, int64_t countb, FILE_SEG fsegb);

int64_t output_data(void *buf, int64_t count, FILE_SEG fseg);
int64_t output_data(void *bufa, int64_t counta, FILE_SEG fsega, void *bufb, int64_t countb, FILE_SEG fsegb);

/**/
/*
size_t input_data(void *buf, size_t count);

size_t output_data( void *buf, size_t count);

size_t input_data(void* bufa, ssize_t counta, void* bufb, ssize_t countb);

size_t output_data(void* bufa, ssize_t counta, void* bufb, ssize_t countb);

size_t input_graph(void **pe, int64_t &nedge, int flag=0);

size_t input_graph(void **pe, int64_t &nedge, void **pv, int32_t &nvtx, int flag);
*/
