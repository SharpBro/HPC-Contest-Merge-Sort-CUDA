#ifndef F7DECB0C_0D9F_4636_9322_59BB010643F8
#define F7DECB0C_0D9F_4636_9322_59BB010643F8

#include <algorithm>
#include <iostream>
#include <time.h>
#include <string.h>
#include <cstdlib>

#define START_T(start)  start = clock()
#define STOP_T(t)  t = (clock() - t)/CLOCKS_PER_SEC // ms insead of s

typedef int DATATYPE;

//int mergesort(DATATYPE *list, DATATYPE *sorted, int n);
void merge(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end);
void initWithRandomData(DATATYPE* l, int size);
bool checkSolution(DATATYPE* l, int size);
void merge(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end);
void mergesort_gpu_seq(DATATYPE *list, DATATYPE *sorted, int n, int chunk);

int mergesort_global(DATATYPE *list, DATATYPE *sorted, int n);
//int mergesort_shared(DATATYPE *list, DATATYPE *sorted, int n);
int mergesort_streams(DATATYPE *list, DATATYPE *sorted, int n);
int mergesort_texture(DATATYPE *list, DATATYPE *sorted, int n);

#endif /* F7DECB0C_0D9F_4636_9322_59BB010643F8 */
