#include <algorithm>
#include <iostream>
#include <time.h>

#define START_T(start)  start = clock()
#define STOP_T(t)  t = (clock() - t)/CLOCKS_PER_SEC // ms insead of s

// Number of shared memory banks (true for every CC)
#define NB 32

typedef int DATATYPE;

int mergesort(DATATYPE *list, DATATYPE *sorted, int n);

void merge(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end);

void initWithRandomData(DATATYPE* l, int size);

bool checkSolution(DATATYPE* l, int size);

int main(int argc, char const *argv[]) {

    //struct timespec start, stop;
    double elapsed_time;

    int i, j;
    unsigned min_size = 2 << 16;
    unsigned max_size = 2 << 27;
    for(j=min_size; j<= max_size; j *= 2){
        std::cout << "############ LENGTH OF LIST: " << j << " ############\n";

        DATATYPE *unsorted = (DATATYPE *) malloc(j*sizeof(DATATYPE));
        DATATYPE *sorted_gpu = (DATATYPE *) malloc(j*sizeof(DATATYPE));
        DATATYPE *sorted_cpu = (DATATYPE *) malloc(j*sizeof(DATATYPE));

        initWithRandomData(unsorted,j);
        
        memcpy(sorted_cpu,unsorted,sizeof(unsorted));

        START_T(elapsed_time);
        mergesort(unsorted, sorted_gpu, j);
        STOP_T(elapsed_time);
        
        std::cout << "TIME TAKEN(Parallel GPU): "<< elapsed_time << " s\n";

        START_T(elapsed_time);
        std::sort(sorted_cpu, sorted_cpu + j);
        STOP_T(elapsed_time);
        
        std::cout << "TIME TAKEN(Sequential CPU): "<< elapsed_time << " s\n";
        
        printf("\n\nOUTPUT:\n");
        for(i=1; i<j; i++){
            printf("%d ", sorted_gpu[i]);
            if(sorted_gpu[i-1]>sorted_gpu[i]){
                std::cout << "WRONG ANSWER _1\n";
                return -1;
            }
        }
        printf("\n\n");
        bool valid = checkSolution(sorted_gpu,j);

        if(!valid){
            std::cout << "WRONG ANSWER _2\n";
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

// // // // // // // // // // // // // // // //
//  GPU Implementation                       //
// // // // // // // // // // // // // // // //
__device__ void merge_gpu(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end) {
    int k = start, i = start, j = mid;
    while (i < mid || j < end)
    {
        if (j == end)
            sorted[k] = list[i++];
        else if (i == mid)
            sorted[k] = list[j++];
        else if (list[i] < list[j])
            sorted[k] = list[i++];
        else
            sorted[k] = list[j++];
        k++;
    }
}

__global__ void mergesort_gpu(DATATYPE *list, DATATYPE *sorted, int n, int chunk) {

    extern __shared__ DATATYPE listS[];
    extern __shared__ DATATYPE sortedS[];

    int idx = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid > n)
        return;    
    listS[idx] = list[tid];
    __syncthreads();

    int start = idx * chunk;
    if (start >= blockDim.x)
        return;
    int mid, end;

    mid = min(start + chunk / 2, blockDim.x);
    end = min(start + chunk, blockDim.x);
    merge_gpu(listS, sortedS, start, mid, end);
    __syncthreads();

    sorted[tid] = sortedS[idx];
}

// Sequential Merge Sort for GPU when Number of Threads Required gets below 1 Warp Size
void mergesort_gpu_seq(DATATYPE *list, DATATYPE *sorted, int n, int chunk)
{
    int chunk_id;
    for (chunk_id = 0; chunk_id * chunk <= n; chunk_id++) {
        int start = chunk_id * chunk, end, mid;
        if (start >= n)
            return;
        mid = min(start + chunk / 2, n);
        end = min(start + chunk, n);
        merge(list, sorted, start, mid, end);
    }
}

void merge(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end)
{
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

int mergesort(DATATYPE *list, DATATYPE *sorted, int n) {

    DATATYPE *list_d;
    DATATYPE *sorted_d;
    int dummy;
    bool flag = false;
    bool sequential = false;

    int size = n * sizeof(DATATYPE);

    cudaMalloc((void **)&list_d, size);
    cudaMalloc((void **)&sorted_d, size);

    cudaMemcpy(list_d, list, size, cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error_2: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // vaues for sm_35 compute capability
    int max_active_blocks_per_sm = 16;
    if(prop.major > 3 && (prop.major < 8 && prop.minor < 5) || prop.major == 8)
        max_active_blocks_per_sm = 32;

    //const int max_active_warps_per_sm = 64;

    int warp_size = prop.warpSize;
    int max_grid_size = prop.maxGridSize[0];
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_procs_count = prop.multiProcessorCount;

    int max_active_blocks = max_active_blocks_per_sm * max_procs_count;
    //int max_active_warps = max_active_warps_per_sm * max_procs_count;

    int chunk_size;
    float total_elapsed_time = 0;

    for (chunk_size = 2; chunk_size < 2 * n; chunk_size *= 2) {
        int blocks_required = 0, threads_per_block = 0;
        int threads_required = (n % chunk_size == 0) ? n / chunk_size : n / chunk_size + 1;

        if (threads_required <= 3 * warp_size && !sequential) {
            sequential = true;
            if (flag)
                cudaMemcpy(list, sorted_d, size, cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(list, list_d, size, cudaMemcpyDeviceToHost);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("ERROR_4: %s\n", cudaGetErrorString(err));
                return -1;
            }
            cudaFree(list_d);
            cudaFree(sorted_d);
        }
        else if (threads_required < max_threads_per_block) {
            threads_per_block = 4 * warp_size;
            dummy = threads_required / threads_per_block;
            blocks_required = (threads_required % threads_per_block == 0) ? dummy : dummy + 1;
        }
        else if (threads_required < 4 * max_active_blocks * warp_size) {
            threads_per_block = max_threads_per_block / 2;
            dummy = threads_required / threads_per_block;
            blocks_required = (threads_required % threads_per_block == 0) ? dummy : dummy + 1;
        }
        else {
            dummy = threads_required / max_active_blocks;
            // int estimated_threads_per_block = (dummy%warp_size==0) ? dummy : (dummy/warp_size + 1)*warp_size;
            int estimated_threads_per_block = (threads_required % max_active_blocks == 0) ? dummy : dummy + 1;
            if (estimated_threads_per_block > max_threads_per_block) {
                threads_per_block = max_threads_per_block;
                dummy = threads_required / max_threads_per_block;
                blocks_required = (threads_required % max_threads_per_block == 0) ? dummy : dummy + 1;
            }
            else {
                threads_per_block = estimated_threads_per_block;
                blocks_required = max_active_blocks;
            }
        }

        if (blocks_required >= max_grid_size) {
            printf("ERROR_2: Too many Blocks Required\n");
            return -1;
        }

        if (sequential) {
            double elapsed;

            START_T(elapsed);
            mergesort_gpu_seq(list, sorted, n, chunk_size);
            STOP_T(elapsed);
            
            //std::cout << "sequential elapsed: " << elapsed << "\n";
            total_elapsed_time += elapsed;
        }
        else {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            //std::cout << "parallel mode\n";
            cudaEventRecord(start);
            int blockSize_byte = threads_per_block * sizeof(DATATYPE);
            if (flag){                
                mergesort_gpu<<<blocks_required, threads_per_block, blockSize_byte>>>(sorted_d, list_d, n, chunk_size);
            } else {
                mergesort_gpu<<<blocks_required, threads_per_block, blockSize_byte>>>(list_d, sorted_d, n, chunk_size);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);

            total_elapsed_time += elapsed;

            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("ERROR_3: %s\n", cudaGetErrorString(err));
                return -1;
            }
            flag = !flag;
        }
    }

    std::cout << "merge sort time: " << total_elapsed_time << " ms\n";

    return 0;
}
