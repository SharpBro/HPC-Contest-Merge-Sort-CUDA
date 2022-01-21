/** 
 * Course: High Performance Computing 2021/2022
 *
 * Lecturer: Francesco Moscato    fmoscato@unisa.it
 *
 * Group:
 * Mario Pellegrino    0622701671  m.pellegrino42@studenti.unisa.it
 * Francesco Sonnessa   0622701672   f.sonnessa@studenti.unisa.it
 *
 * Copyright (C) 2021 - All Rights Reserved 
 *
 * This file is part of Contest-CUDA.
 *
 * Contest-CUDA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Contest-CUDA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Contest-CUDA.  If not, see <http://www.gnu.org/licenses/>. 
 */

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
