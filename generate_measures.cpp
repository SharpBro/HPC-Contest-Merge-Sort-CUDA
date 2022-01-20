#include "main.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>

//typedef std::basic_ostringstream<char> ostringstream;

int main(int argc, char const *argv[]) {

    //struct timespec start, stop;
    double elapsed_time_GPU, merge_time_GPU, elapsed_time_CPU;

    if(argc < 3){
        std::cout << "usage: " << argv[0] << " [VERSION] [SIZE] {TEST}\n";
        std::cout << "VERSION values:\n";
        std::cout << "\t 1 - work on global memory\n";
        std::cout << "\t 2 - work on texture memory\n";
        std::cout << "\t 3 - work with streams on global memory\n";
        std::cout << "\t default - work on global memory\n";
        std::cout << "TEST (optional):\n";
        std::cout << "\t any numbers - check corect result\n";
        std::cout << "\t dafult - disabled\n";
        return -1;
    }
    
    const int VERSION = atoi(argv[1]);
    const int size = atoi(argv[2]);
    const bool test = (argc == 4) ? true : false;
    
    DATATYPE *unsorted = (DATATYPE *) malloc(size*sizeof(DATATYPE));
    DATATYPE *sorted_gpu = (DATATYPE *) malloc(size*sizeof(DATATYPE));
    DATATYPE *sorted_cpu = (DATATYPE *) malloc(size*sizeof(DATATYPE));

    initWithRandomData(unsorted, size);
        
    memcpy(sorted_cpu,unsorted,sizeof(unsorted));

    int (*mergesort)(DATATYPE*,DATATYPE*,int); // function pointer

    switch (VERSION){
        case 1: mergesort = &mergesort_global; break;
        case 2: mergesort = &mergesort_texture; break;
        case 3: mergesort = &mergesort_streams; break;
        //case 4: mergesort = &mergesort_shared; break;
        default: mergesort = &mergesort_global; break;
    }

    // Redirect cout.
    std::streambuf* oldCoutStreamBuf = std::cout.rdbuf();
    std::ostringstream strCout;
    std::cout.rdbuf( strCout.rdbuf() );

    START_T(elapsed_time_GPU);
        (*mergesort) (unsorted, sorted_gpu, size); // calling the proper version
    STOP_T(elapsed_time_GPU);

    // Restore old cout.
    std::cout.rdbuf( oldCoutStreamBuf );

    std::string str = strCout.str();
    str = str.substr(17, 7);
    merge_time_GPU = atof(str.c_str()) / 1000;
    
    START_T(elapsed_time_CPU);
        std::sort(sorted_cpu, sorted_cpu + size);
    STOP_T(elapsed_time_CPU);
    
    std::cout << size << ";" << merge_time_GPU << ";" << elapsed_time_GPU << ";" << elapsed_time_CPU << "\n";

    if (test){
        bool valid = checkSolution(sorted_gpu, size);
        if(!valid){
            std::cout << "WRONG ANSWER\n";
            return -1;
        }
        else std::cout << "CORRECT ANSWER\n";
    }
        
    free(unsorted);
    free(sorted_gpu);
    free(sorted_cpu);

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
