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

__device__ void merge_gpu_streams(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end);
__global__ void mergesort_gpu_streams(DATATYPE *list, DATATYPE *sorted, int n, int chunk);

// // // // // // // // // // // // // // // //
//  GPU Implementation                       //
// // // // // // // // // // // // // // // //
__device__ void merge_gpu_streams(DATATYPE *list, DATATYPE *sorted, int start, int mid, int end) {
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

__global__ void mergesort_gpu_streams(DATATYPE *list, DATATYPE *sorted, int n, int chunk) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid * chunk;
    if (start >= n)
        return;
    int mid, end;

    mid = min(start + chunk / 2, n);
    end = min(start + chunk, n);
    merge_gpu_streams(list, sorted, start, mid, end);
}

int mergesort_streams(DATATYPE *list, DATATYPE *sorted, int n) {

    DATATYPE *list_d;
    DATATYPE *sorted_d;
    int dummy;
    bool flag = false;
    bool sequential = false;

    int size = n * sizeof(DATATYPE);

    cudaMalloc((void **)&list_d, size);
    cudaMalloc((void **)&sorted_d, size);

    cudaStream_t load_H2D_str, kernel_str, load_D2H_str;
    cudaStreamCreate(&load_H2D_str);
    cudaStreamCreate(&kernel_str);
    cudaStreamCreate(&load_D2H_str);

    cudaMemcpyAsync(list_d, list, size, cudaMemcpyHostToDevice, load_H2D_str);
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
                cudaMemcpyAsync(list, sorted_d, size, cudaMemcpyDeviceToHost,load_D2H_str);
            else
                cudaMemcpyAsync(list, list_d, size, cudaMemcpyDeviceToHost,load_D2H_str);
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
            if (flag){
                mergesort_gpu_streams<<<blocks_required, threads_per_block,0,kernel_str>>>(sorted_d, list_d, n, chunk_size);
            } else {
                mergesort_gpu_streams<<<blocks_required, threads_per_block,0,kernel_str>>>(list_d, sorted_d, n, chunk_size);
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
    
    cudaStreamDestroy(load_H2D_str);
    cudaStreamDestroy(kernel_str);
    cudaStreamDestroy(load_D2H_str);

    std::cout << "merge sort time: " << total_elapsed_time << " ms\n";

    return 0;
}
