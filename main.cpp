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

#include "main.hpp"

int main(int argc, char const *argv[]) {

    //struct timespec start, stop;
    double elapsed_time;

    if(argc != 2){
        std::cout << "usage: " << argv[0] << " [VERSION]\n";
        std::cout << "\t 1 - work on global memory\n";
        std::cout << "\t 2 - work on texture memory\n";
        std::cout << "\t 3 - work with streams on global memory\n";
        std::cout << "\t default - work on global memory\n";
        return -1;
    }
    
    const int VERSION = atoi(argv[1]);

    int i, j;
    unsigned min_size = 1 << 16; // 2^16
    unsigned max_size = 1 << 24; // 2^24
    for(j=min_size; j<= max_size; j *= 2){
        std::cout << "############ LENGTH OF LIST: " << j << " ############\n";

        DATATYPE *unsorted = (DATATYPE *) malloc(j*sizeof(DATATYPE));
        DATATYPE *sorted_gpu = (DATATYPE *) malloc(j*sizeof(DATATYPE));
        DATATYPE *sorted_cpu = (DATATYPE *) malloc(j*sizeof(DATATYPE));

        initWithRandomData(unsorted,j);
        
        memcpy(sorted_cpu,unsorted,sizeof(unsorted));

        int (*mergesort)(DATATYPE*,DATATYPE*,int); // function pointer

        switch (VERSION){
        case 1: mergesort = &mergesort_global; break;
        case 2: mergesort = &mergesort_texture; break;
        case 3: mergesort = &mergesort_streams; break;
        //case 4: mergesort = &mergesort_shared; break;
        default: mergesort = &mergesort_global; break;
        }

        START_T(elapsed_time);
        (*mergesort) (unsorted, sorted_gpu, j); // calling the proper version
        STOP_T(elapsed_time);
        
        std::cout << "TIME TAKEN(Parallel GPU): "<< elapsed_time << " s\n";

        START_T(elapsed_time);
        std::sort(sorted_cpu, sorted_cpu + j);
        STOP_T(elapsed_time);
        
        std::cout << "TIME TAKEN(Sequential CPU): "<< elapsed_time << " s\n";
        
        bool valid = checkSolution(sorted_gpu,j);
        if(!valid){
            std::cout << "WRONG ANSWER\n";
            return -1;
        }
        else std::cout << "CORRECT ANSWER\n";

        free(unsorted);
        free(sorted_gpu);
        free(sorted_cpu);
        std::cout << "##################################################\n";
    }
    return 0;
}

void initWithRandomData(DATATYPE* l, int size){
    for(int i=0; i<size; i++){
        l[i] = rand();
    }
}

bool checkSolution(DATATYPE* l, int size){
    for(int i=1; i<size; i++){
        if(l[i-1] > l[i]) return false;
    }
    return true;
}

// Sequential Merge Sort for GPU when Number of Threads Required gets below 1 Warp Size
void mergesort_gpu_seq(DATATYPE *list, DATATYPE *sorted, int n, int chunk) {
    int chunk_id;
    for (chunk_id = 0; chunk_id * chunk <= n; chunk_id++) {
        int start = chunk_id * chunk, end, mid;
        if (start >= n)
            return;
        mid = std::min(start + chunk / 2, n);
        end = std::min(start + chunk, n);
        merge(list, sorted, start, mid, end);
    }
}

void merge(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end) {
    int ti=start, i=start, j=mid;
    while (i<mid || j<end) {
        if (j==end) sorted[ti] = list[i++];
        else if (i==mid) sorted[ti] = list[j++];
        else if (list[i]<list[j]) sorted[ti] = list[i++];
        else sorted[ti] = list[j++];
        ti++;
    }

    for (ti=start; ti<end; ti++)
        list[ti] = sorted[ti];
}
