#include <algorithm>
#include <iostream>

typedef int DATATYPE;

int mergesort(DATATYPE *list, DATATYPE *sorted, int n);

void merge(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end);

int main(int argc, char const *argv[]) {

    struct timespec start, stop;

    int i, j;
    unsigned min_size = 2 << 16;
    unsigned max_size = 2 << 27;
    for(j=min_size; j<= max_size; j *= 2){
        std::cout << "############ LENGTH OF LIST: " << j << " ############\n";

        DATATYPE *sorted = (DATATYPE *) malloc(j*sizeof(DATATYPE));
        DATATYPE *list = (DATATYPE *) malloc(j*sizeof(DATATYPE));
        DATATYPE *list_s = (DATATYPE *) malloc(j*sizeof(DATATYPE));
        for(i=0; i<j; i++){
            list[i] = rand()%10000;
            list_s[i] = list[i];
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        mergesort(list, sorted, j);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double result = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_nsec - start.tv_nsec) / 1e6;
        std::cout << "TIME TAKEN(Parallel GPU): "<< result << "ms\n";


        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        //mergesort_cpu(list_s, sorted_s, j);
        std::sort(list_s, list_s + j);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        result = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_nsec - start.tv_nsec) / 1e6;
        std::cout << "TIME TAKEN(Sequential CPU): "<< result << "ms\n";
        

        for(i=1; i<j; i++){
            if(sorted[i-1]>sorted[i]){
                std::cout << "WRONG ANSWER _1\n";
                return -1;
            }
        }
        std::cout << "CORRECT ANSWER\n";

        free(list);
        free(sorted);
        free(list_s);
        std::cout << "##################################################\n";
    }
    return 0;
}

// // // // // // // // // // // // // // // //
//  GPU Implementation                       //
// // // // // // // // // // // // // // // //
__device__ void merge_gpu(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end)
{
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

__global__ void mergesort_gpu(DATATYPE *list, DATATYPE *sorted, int n, int chunk)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid * chunk;
    if (start >= n)
        return;
    int mid, end;

    mid = min(start + chunk / 2, n);
    end = min(start + chunk, n);
    merge_gpu(list, sorted, start, mid, end);
}

// Sequential Merge Sort for GPU when Number of Threads Required gets below 1 Warp Size
void mergesort_gpu_seq(DATATYPE *list, DATATYPE *sorted, int n, int chunk)
{
    int chunk_id;
    for (chunk_id = 0; chunk_id * chunk <= n; chunk_id++)
    {
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
    while (i<mid || j<end)
    {
        if (j==end) sorted[ti] = list[i++];
        else if (i==mid) sorted[ti] = list[j++];
        else if (list[i]<list[j]) sorted[ti] = list[i++];
        else sorted[ti] = list[j++];
        ti++;
    }

    for (ti=start; ti<end; ti++)
        list[ti] = sorted[ti];
}

int mergesort(DATATYPE *list, DATATYPE *sorted, int n)
{

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
    if (err != cudaSuccess)
    {
        printf("Error_2: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    //int major = prop.major;
    //int minor = prop.minor;
    /* if (major != 3 || minor != 5)
    {
        printf("The Program is Optimized only for sm_35 Compute Capability..May NOT Work for Other CCs\n");
    } */
    // vaues for sm_35 compute capability
    int max_active_blocks_per_sm = 16;
    if(prop.major > 3)
        max_active_blocks_per_sm = 32;

    //const int max_active_warps_per_sm = 64;

    int warp_size = prop.warpSize;
    int max_grid_size = prop.maxGridSize[0];
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_procs_count = prop.multiProcessorCount;

    int max_active_blocks = max_active_blocks_per_sm * max_procs_count;
    //int max_active_warps = max_active_warps_per_sm * max_procs_count;

    int chunk_size;
    for (chunk_size = 2; chunk_size < 2 * n; chunk_size *= 2)
    {
        int blocks_required = 0, threads_per_block = 0;
        int threads_required = (n % chunk_size == 0) ? n / chunk_size : n / chunk_size + 1;

        if (threads_required <= 3 * warp_size && !sequential)
        {
            //std::cout << "sequential mode\n";
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
        else if (threads_required < max_threads_per_block)
        {
            threads_per_block = 4 * warp_size;
            dummy = threads_required / threads_per_block;
            blocks_required = (threads_required % threads_per_block == 0) ? dummy : dummy + 1;
        }
        else if (threads_required < 4 * max_active_blocks * warp_size)
        {
            threads_per_block = max_threads_per_block / 2;
            dummy = threads_required / threads_per_block;
            blocks_required = (threads_required % threads_per_block == 0) ? dummy : dummy + 1;
        }
        else
        {
            dummy = threads_required / max_active_blocks;
            // int estimated_threads_per_block = (dummy%warp_size==0) ? dummy : (dummy/warp_size + 1)*warp_size;
            int estimated_threads_per_block = (threads_required % max_active_blocks == 0) ? dummy : dummy + 1;
            if (estimated_threads_per_block > max_threads_per_block)
            {
                threads_per_block = max_threads_per_block;
                dummy = threads_required / max_threads_per_block;
                blocks_required = (threads_required % max_threads_per_block == 0) ? dummy : dummy + 1;
            }
            else
            {
                threads_per_block = estimated_threads_per_block;
                blocks_required = max_active_blocks;
            }
        }

        if (blocks_required >= max_grid_size)
        {
            printf("ERROR_2: Too many Blocks Required\n");
            return -1;
        }

        if (sequential)
        {

            mergesort_gpu_seq(list, sorted, n, chunk_size);
        }
        else
        {
            //std::cout << "parallel mode\n";
            if (flag)
                mergesort_gpu<<<blocks_required, threads_per_block>>>(sorted_d, list_d, n, chunk_size);
            else
                mergesort_gpu<<<blocks_required, threads_per_block>>>(list_d, sorted_d, n, chunk_size);
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
    return 0;
}
