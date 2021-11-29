/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

// TODO: implement hook for potentials by particles

/* Include configuratin header */
#include <mdcore_config.h>

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>

/* Include headers for overloaded vector functions. */
#include "cutil_math.h"

#include <cuda_runtime.h>
#include <curand.h>

/* Include some conditional headers. */
#ifdef HAVE_MPI
    #include <mpi.h>
#endif

/* Force single precision. */
#ifndef FPTYPE_SINGLE
    #define FPTYPE_SINGLE 1
#endif

/* Disable vectorization for the nvcc compiler's sake. */
#undef __SSE__
#undef __SSE2__
#undef __ALTIVEC__
#undef __AVX__

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "MxParticle_cuda.h"
#include "space_cell.h"
#include "space.h"
#include "task.h"
#include "MxPotential_cuda.h"
#include "MxBoundaryConditions_cuda.h"
#include "Flux.hpp"
#include "engine.h"
#include "runner_cuda.h"
#include "flux_cuda.h"

#ifndef CPU_TPS
#include <ctime>
#define CPU_TPS CLOCKS_PER_SEC
#endif

/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* The parts (non-texture access). */
__constant__ MxParticleCUDA *cuda_parts;

// Boundary conditions
__constant__ MxBoundaryConditionsCUDA cuda_bcs;

/* Diagonal entries and potential index lookup table. */
__constant__ unsigned int *cuda_pind;
__constant__ unsigned int *cuda_pind_cluster;

/* The mutex for accessing the cell pair list. */
__device__ int cuda_cell_mutex = 0;
__device__ int cuda_barrier = 0;
__device__ volatile int NAMD_barrier = 0;

/* The index of the next free cell pair. */
__device__ int cuda_pair_next = 0;

/* The list of cell pairs. */

__device__ int *cuda_taboo;
#ifdef TASK_TIMERS
/*x = block y = type z = start w = end*/
__device__ int4 NAMD_timers[26*10000];
#endif

/* The list of tasks. */
__constant__ struct task_cuda *cuda_tasks;
__constant__ int cuda_nr_tasks = 0;

/* The per-SM task queues. */
__device__ struct queue_cuda cuda_queues[ cuda_maxqueues ];
__device__ struct queue_cuda cuda_sorts[ cuda_maxqueues ];
__constant__ int cuda_queue_size;

/* Some constants. */
__constant__ float cuda_dt = 0.0f;
__constant__ float cuda_cutoff2 = 0.0f;
__constant__ float cuda_cutoff = 0.0f;
__constant__ float cuda_dscale = 0.0f;
__constant__ float cuda_maxdist = 0.0f;
__constant__ int cuda_maxtype = 0;
__constant__ struct MxPotentialCUDA *cuda_pots = NULL;
__constant__ struct MxPotentialCUDA *cuda_pots_cluster = NULL;

/* Sortlists for the Verlet algorithm. */
__device__ unsigned int *cuda_sortlists = NULL;

/* Cell origins. */
__constant__ float *cuda_corig;

// Cell dimensions
__constant__ float3 *cuda_cdims;

// Cell flags
__constant__ unsigned int *cuda_cflags;

/* Potential energy. */
__device__ float cuda_epot = 0.0f, cuda_epot_out;

/* Timers. */
__device__ float cuda_timers[ tid_count ];

// Random number generators
__device__ curandState *cuda_rand_norm;

__global__ void cuda_init_rand_norm_device(curandState *rand_norm, int nr_rands, unsigned long long seed) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < nr_rands) {
        curand_init(seed, tid, 0, &rand_norm[tid]);
        tid += stride;
    }
}

extern "C" int engine_cuda_rand_norm_init(struct engine *e) {

    for(int did = 0; did < e->nr_devices; did++) {

        if(e->rand_norm_init_cuda[did]) 
            continue;

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        int nr_threads = e->nr_threads[did];
        int nr_blocks = e->nr_blocks[did];
        int nr_rands = nr_threads * nr_blocks;

        if(cudaMalloc(&e->rand_norm_cuda[did], sizeof(curandState) * nr_rands) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        cuda_init_rand_norm_device<<<nr_blocks, nr_threads>>>((curandState *)e->rand_norm_cuda[did], nr_rands, e->rand_norm_seed_cuda);
        if(cudaPeekAtLastError() != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaMemcpyToSymbol(cuda_rand_norm, &e->rand_norm_cuda[did], sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        e->rand_norm_init_cuda[did] = true;

    }

    return engine_err_ok;

}

int engine_cuda_rand_norm_finalize(struct engine *e) {

    for(int did = 0; did < e->nr_devices; did++) { 

        if(!e->rand_norm_init_cuda[did]) 
            continue;

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaFree(e->rand_norm_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        e->rand_norm_init_cuda[did] = false;

    }
    
    return engine_err_ok;

}

/**
 * @brief Sets the random seed for the CUDA uniform number generators. 
 * 
 * @param e The #engine
 * @param seed The seed
 * @param onDevice A flag specifying whether the engine is current on the device
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_cuda_rand_norm_setSeed(struct engine *e, unsigned int seed, bool onDevice) {

    if(onDevice)
        if(engine_cuda_rand_norm_finalize(e) < 0)
            return cuda_error(engine_err_cuda);

    e->rand_norm_seed_cuda = seed;

    if(onDevice) {
        if(engine_cuda_rand_norm_init(e) < 0)
            return cuda_error(engine_err_cuda);

        for(int did = 0; did < e->nr_devices; did++) {

            if(cudaSetDevice(e->devices[did]) != cudaSuccess)
                return cuda_error(engine_err_cuda);

            if(cudaDeviceSynchronize() != cudaSuccess)
                return cuda_error(engine_err_cuda);

        }
    }

    return engine_err_ok;

}


/* Map sid to shift vectors. */
__constant__ float cuda_shiftn[13*3] = {
     5.773502691896258e-01 ,  5.773502691896258e-01 ,  5.773502691896258e-01 ,
     7.071067811865475e-01 ,  7.071067811865475e-01 ,  0.0                   ,
     5.773502691896258e-01 ,  5.773502691896258e-01 , -5.773502691896258e-01 ,
     7.071067811865475e-01 ,  0.0                   ,  7.071067811865475e-01 ,
     1.0                   ,  0.0                   ,  0.0                   ,
     7.071067811865475e-01 ,  0.0                   , -7.071067811865475e-01 ,
     5.773502691896258e-01 , -5.773502691896258e-01 ,  5.773502691896258e-01 ,
     7.071067811865475e-01 , -7.071067811865475e-01 ,  0.0                   ,
     5.773502691896258e-01 , -5.773502691896258e-01 , -5.773502691896258e-01 ,
     0.0                   ,  7.071067811865475e-01 ,  7.071067811865475e-01 ,
     0.0                   ,  1.0                   ,  0.0                   ,
     0.0                   ,  7.071067811865475e-01 , -7.071067811865475e-01 ,
     0.0                   ,  0.0                   ,  1.0                   ,
     };
__constant__ float cuda_shift[13*3] = {
     1.0 ,  1.0 ,  1.0 ,
     1.0 ,  1.0 ,  0.0 ,
     1.0 ,  1.0 , -1.0 ,
     1.0 ,  0.0 ,  1.0 ,
     1.0 ,  0.0 ,  0.0 ,
     1.0 ,  0.0 , -1.0 ,
     1.0 , -1.0 ,  1.0 ,
     1.0 , -1.0 ,  0.0 ,
     1.0 , -1.0 , -1.0 ,
     0.0 ,  1.0 ,  1.0 ,
     0.0 ,  1.0 ,  0.0 ,
     0.0 ,  1.0 , -1.0 ,
     0.0 ,  0.0 ,  1.0 ,
    };
    
/* The cell edge lengths and space dimensions. */
__constant__ float cuda_h[3];
__constant__ float cuda_dim[3];
    
    
/**
 * @brief Lock a device mutex.
 *
 * @param m The mutex.
 *
 * Loops until the mutex can be set. Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ void cuda_mutex_lock ( int *m ) {
    TIMER_TIC
    while ( atomicCAS( m , 0 , 1 ) != 0 );
    TIMER_TOC( tid_mutex )
}


/**
 * @brief Attempt to lock a device mutex.
 *
 * @param m The mutex.
 *
 * Try to grab the mutex. Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ int cuda_mutex_trylock ( int *m ) {
    TIMER_TIC
    int res = atomicCAS( m , 0 , 1 ) == 0;
    TIMER_TOC( tid_mutex )
    return res;
}


/**
 * @brief Lock a device mutex with an additional condition.
 *
 * @param m The mutex.
 * @param c the condition
 *
 * @return @c 1 if the mutex could be locked or zero if the condition @c c
 * was reached first.
 *
 * Loops until the mutex can be set or until @c *c is non-zero.
 * Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ int cuda_mutex_lock_cond ( int *m , int *c ) {
    TIMER_TIC
    while ( atomicCAS( c , 0 , 0 ) == 0 )
        if ( atomicCAS( m , 0 , 1 ) == 0 ) {
            TIMER_TOC( tid_mutex )
            return 1;
            }
    TIMER_TOC( tid_mutex )
    return 0;
}


/**
 * @brief Unlock a device mutex.
 *
 * @param m The mutex.
 *
 * Does not check if the mutex had been locked.
 */

__device__ void cuda_mutex_unlock ( int *m ) {
    TIMER_TIC
    atomicExch( m , 0 );
    TIMER_TOC( tid_mutex )
}
    
    
/**
 * @brief Get a task ID from the given queue.
 *
 */
 
__device__ int cuda_queue_gettask ( struct queue_cuda *q ) {

    int ind, tid = -1;
    
    /* Don't even try... */
    if ( q->rec_count == q->count )
        return -1;

    /* Get the index of the next task. */
    ind = atomicAdd( &q->first , 1 );
        
    /* Wrap the index. */
    ind %= cuda_queue_size; 

    /* Loop until there is a valid task at that index. */
    while ( q->rec_count < q->count && ( tid = q->data[ind] ) < 0 );
    
    /* Scratch the task from the queue */
    if ( tid >= 0 )
        q->data[ind] = -1;

    /* Return the acquired task ID. */
    return tid;
    
}


/**
 * @brief Put a task onto the given queue.
 *
 * @param tid The task ID to add to the end of the queue.
 */
 
__device__ void cuda_queue_puttask ( struct queue_cuda *q , int tid ) {

    int ind;

    /* Get the index of the next task. */
    ind = atomicAdd( &q->last , 1 ) % cuda_queue_size;
    
    /* Wait for the slot in the queue to be empty. */
    while ( q->data[ind] != -1 );

    /* Write the task back to the queue. */
    q->data[ind] = tid;
    
}
    
    
/**
 * @brief Get the ID of the block's SM.
 */
 
__noinline__ __device__ uint get_smid ( void ) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}


/**
 * @brief Get a task from the given task queue.
 *
 * Picks tasks from the queue sequentially and checks if they
 * can be computed. If not, they are returned to the queue.
 *
 * This routine blocks until a valid task is picked up, or the
 * specified queue is empty.
 */
 
__device__ int runner_cuda_gettask ( struct queue_cuda *q , int steal ) {

    int tid = -1;
    int cid, cjd;
    
    TIMER_TIC
    
    /* Main loop. */
    while ( ( tid = cuda_queue_gettask( q ) ) >= 0 ) {
    
        /* If this task is not even free, don't even bother. */
        if ( !cuda_tasks[tid].wait ) {
    
            /* Dfferent options for different tasks. */
            if ( cuda_tasks[tid].type == task_type_sort ) {
            
                /* No locking needed. */
                break;
            
            }
            else if ( cuda_tasks[tid].type == task_type_self ) {
            
                /* Decode this task. */
                cid = cuda_tasks[tid].i;

                /* Lock down this task? */
                if ( cuda_mutex_trylock( &cuda_taboo[ cid ] ) )
                    break;
                        
            }
            else if ( cuda_tasks[tid].type == task_type_pair ) {
            
                /* Decode this task. */
                cid = cuda_tasks[tid].i;
                cjd = cuda_tasks[tid].j;

                /* Lock down this task? */
                if ( cuda_mutex_trylock( &cuda_taboo[ cid ] ) )
                    if ( cuda_mutex_trylock( &cuda_taboo[ cjd ] ) ) 
                        break;
                    else
                        cuda_mutex_unlock( &cuda_taboo[ cid ] );
                        
            }

        }
                
        /* Put this task back into the queue. */
        cuda_queue_puttask( q , tid );
    
    }
        
    /* Put this task into the recycling queue, if needed. */
    if ( tid >= 0 ) {
        if ( steal )
            atomicSub( (int *)&q->count , 1 );
        else
            q->rec_data[ atomicAdd( (int *)&q->rec_count , 1 ) ] = tid;
    }
        
    TIMER_TOC(tid_queue);
        
    /* Return whatever we got. */
    return tid;

}

__device__ int runner_cuda_gettask_nolock ( struct queue_cuda *q , int steal ) {

    int tid = -1/*,cid,cjd*/;
    
    TIMER_TIC
    
    /* Main loop. */
    while ( ( tid = cuda_queue_gettask( q ) ) >= 0 ) {
    
        /* If this task is not even free, don't even bother. */
        if ( !cuda_tasks[tid].wait ) {
    
            break;

        }
        /*if( cuda_tasks[tid].type == task_type_pair )
        {
	       	cid = cuda_tasks[tid].i;
	       	cjd = cuda_tasks[tid].j;
	       	if(!( cuda_taboo[cid] || cuda_taboo[cjd] ))
	       		break;
       	}else{
       		break;
       	}*/
       	
                
        /* Put this task back into the queue. */
        cuda_queue_puttask( q , tid );
    
    }
        
    /* Put this task into the recycling queue, if needed. */
    if ( tid >= 0 ) {
        if ( steal )
            atomicSub( (int *)&q->count , 1 );
        else
            q->rec_data[ atomicAdd( (int *)&q->rec_count , 1 ) ] = tid;
    }
        
    TIMER_TOC(tid_queue);
        
    /* Return whatever we got. */
    return tid;

}

__device__ inline void w_cubic_spline_cuda(float r2, float h, float *result) {
    float r = rsqrt(r2);
    float x = r/h;
    float y;
    
    if(x < 1.f) {
        float x2 = x * x;
        y = 1.f - (3.f / 2.f) * x2 + (3.f / 4.f) * x2 * x;
    }
    else if(x >= 1.f && x < 2.f) {
        float arg = 2.f - x;
        y = (1.f / 4.f) * arg * arg * arg;
    }
    else {
        y = 0.f;
    }
    
    *result = y / (M_PI * h * h * h);
}

__device__ inline void w_cubic_spline_cuda(float r2, float h, float *result, float *_r) {
    float r = sqrt(r2);
    float x = 1.f/(r*h);
    float y;
    
    if(x < 1.f) {
        float x2 = x * x;
        y = 1.f - (3.f / 2.f) * x2 + (3.f / 4.f) * x2 * x;
    }
    else if(x >= 1.f && x < 2.f) {
        float arg = 2.f - x;
        y = (1.f / 4.f) * arg * arg * arg;
    }
    else {
        y = 0.f;
    }
    
    *result = y / (M_PI * h * h * h);
    *_r = r;
}
    
/**
 * @brief Copy bulk memory in a strided way.
 *
 * @param dest Pointer to destination memory.
 * @param source Pointer to source memory.
 * @param count Number of bytes to copy, must be a multiple of sizeof(int).
 */
 
__device__ inline void cuda_memcpy ( void *dest , void *source , int count ) {

    int k;
    int *idest = (int *)dest, *isource = (int *)source;

    int threadID = threadIdx.x;
    
    TIMER_TIC
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = threadID ; k < count/sizeof(int) ; k += blockDim.x )
        idest[k] = isource[k];
        
    TIMER_TOC(tid_memcpy)
        
}


    
__device__ inline void cuda_memcpy_old ( void *dest , void *source , int count ) {

    int k;
    volatile int *idest = (int *)dest, *isource = (int *)source;
    
    TIMER_TIC
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = 0 + threadIdx.x ; k < count/sizeof(int) ; k += cuda_frame )
        idest[k] = isource[k];
        
    TIMER_TOC(tid_memcpy)
        
}


/**
 * @brief Sum two vectors in a strided way.
 *
 * @param a Pointer to destination memory.
 * @param b Pointer to source memory.
 * @param count Number of floats to sum.
 *
 * Computes @c a[k] += b[k] for k=1..count.
 */
 
__device__ inline void cuda_sum ( float *a , float *b , int count ) {

    int i, j, k, icount = count / cuda_frame / cuda_sum_chunk;
    float chunk[cuda_memcpy_chunk];
    int threadID = threadIdx.x;
    
    TIMER_TIC
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = 0 ; k < icount ; k += 1 ) {
        #pragma unroll
        for ( j = 0 ; j < cuda_sum_chunk ; j++ ) {
            i = (cuda_sum_chunk*k+j)*cuda_frame + threadID;
            chunk[j] = a[i] + b[i];
        }
        #pragma unroll
        for ( j = 0 ; j < cuda_sum_chunk ; j++ )
            a[ (cuda_sum_chunk*k+j)*cuda_frame + threadID ] = chunk[j];
    }
    for ( k = cuda_sum_chunk*cuda_frame*icount + threadID ; k < count ; k += cuda_frame )
        a[k] += b[k];
        
    TIMER_TOC(tid_update)
        
}
    
    
    
    /**
 * @brief Sort the given data w.r.t. the lowest 16 bits in decending order.
 *
 * @param a The array to sort.
 * @param count The number of elements.
 */
 
__device__ inline void cuda_sort_descending ( unsigned int *a , int count ) {

    
    int i, j, k, threadID = threadIdx.x;
    int hi, lo, ind, jnd;
    unsigned int swap_i, swap_j;

    TIMER_TIC

    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count ; k *= 2 ) {
    
        /* First step. */
        for ( i = threadID ;  i < count ; i += blockDim.x ) {
            hi = i & ~(k-1); lo = i & (k-1);
            ind = i + hi; jnd = 2*(hi+k) - lo - 1;
            swap_i = ( jnd < count ) ? a[ind] : 0;
            swap_j = ( jnd < count ) ? a[jnd] : 0;
            if  ( ( swap_i & 0xffff ) < ( swap_j & 0xffff ) ) {
                a[ind] = swap_j;
                a[jnd] = swap_i;
            }
        }
            
        /* Let that last step sink in. */
            __syncthreads();
    
        /* Second step(s). */
        for ( j = k/2 ; j > 0 ; j /= 2 ) {
            for ( i = threadID ;  i < count ; i += blockDim.x ) {
                hi = i & ~(j-1);
                ind = i + hi; jnd = ind + j;
                swap_i = ( jnd < count ) ? a[ind] : 0;
                swap_j = ( jnd < count ) ? a[jnd] : 0;
                if  ( ( swap_i & 0xffff ) < ( swap_j & 0xffff ) ) {
                    a[ind] = swap_j;
                    a[jnd] = swap_i;
                }
            }
                __syncthreads();
        }
            
    }
        
    TIMER_TOC(tid_sort)

        
}

    


    
/**
 * @brief Sort the given data w.r.t. the lowest 16 bits in ascending order.
 *
 * @param a The array to sort.
 * @param count The number of elements.
 */
 
__device__ void cuda_sort_ascending ( unsigned int *a , int count ) {

    int i, j, k, threadID = threadIdx.x;
    int hi[2], lo[2], ind[2], jnd[2];
    unsigned int swap_i[2], swap_j[2];

    TIMER_TIC

    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count ; k *= 2 ) {
    
        /* First step. */
        for ( i = threadID ;  i < count ; i += 2*cuda_frame ) {
            hi[0] = i & ~(k-1); lo[0] = i & (k-1);
            hi[1] = (i + cuda_frame) & ~(k-1); lo[1] = (i + cuda_frame) & (k-1);
            ind[0] = i + hi[0]; jnd[0] = 2*(hi[0]+k) - lo[0] - 1;
            ind[1] = i + cuda_frame + hi[1]; jnd[1] = 2*(hi[1]+k) - lo[1] - 1;
            swap_i[0] = ( jnd[0] < count ) ? a[ind[0]] : 0;
            swap_i[1] = ( jnd[1] < count ) ? a[ind[1]] : 0;
            swap_j[0] = ( jnd[0] < count ) ? a[jnd[0]] : 0;
            swap_j[1] = ( jnd[1] < count ) ? a[jnd[1]] : 0;
            if  ( ( swap_i[0] & 0xffff ) > ( swap_j[0] & 0xffff ) ) {
                a[ind[0]] = swap_j[0];
                a[jnd[0]] = swap_i[0];
            }
            if  ( ( swap_i[1] & 0xffff ) > ( swap_j[1] & 0xffff ) ) {
                a[ind[1]] = swap_j[1];
                a[jnd[1]] = swap_i[1];
            }
        }
            
        /* Let that last step sink in. */
        // __threadfence_block();
    
        /* Second step(s). */
        for ( j = k/2 ; j > 0 ; j /= 2 ) {
            for ( i = threadID ;  i < count ; i += 2*cuda_frame ) {
                hi[0] = i & ~(j-1);
                hi[1] = (i + cuda_frame) & ~(j-1);
                ind[0] = i + hi[0]; jnd[0] = ind[0] + j;
                ind[1] = i + cuda_frame + hi[1]; jnd[1] = ind[1] + j;
                swap_i[0] = ( jnd[0] < count ) ? a[ind[0]] : 0;
                swap_i[1] = ( jnd[1] < count ) ? a[ind[1]] : 0;
                swap_j[0] = ( jnd[0] < count ) ? a[jnd[0]] : 0;
                swap_j[1] = ( jnd[1] < count ) ? a[jnd[1]] : 0;
                if  ( ( swap_i[0] & 0xffff ) > ( swap_j[0] & 0xffff ) ) {
                    a[ind[0]] = swap_j[0];
                    a[jnd[0]] = swap_i[0];
                }
                if  ( ( swap_i[1] & 0xffff ) > ( swap_j[1] & 0xffff ) ) {
                    a[ind[1]] = swap_j[1];
                    a[jnd[1]] = swap_i[1];
                }
            }
            // __threadfence_block();
        }
            
    }
        
    TIMER_TOC(tid_sort)
        
}

__device__ 
void engine_cuda_rand_norm(int threadID, int nr_rands, float *result) {
    int i;
    curandState *rand_norm = &cuda_rand_norm[threadID];
    for(i = 0; i < nr_rands; i++) 
        result[i] = curand_normal(rand_norm);
}

__device__ 
void engine_cuda_rand_norm_getState(int threadID, curandState **rand_norm) {
    *rand_norm = &cuda_rand_norm[threadID];
}

__device__ 
void engine_cuda_random_vector3(float vecLength, float *result) {
    curandState *rand_norm;
    engine_cuda_rand_norm_getState(threadIdx.x + blockIdx.x * blockDim.x, &rand_norm);
    result[0] = curand_normal(rand_norm);
    result[1] = curand_normal(rand_norm);
    result[2] = curand_normal(rand_norm);
    float len = sqrtf(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
    result[0] = result[0] * vecLength / len;
    result[1] = result[1] * vecLength / len;
    result[2] = result[2] * vecLength / len;
}


extern "C" int engine_cuda_boundary_conditions_load(struct engine *e) {

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        MxBoundaryConditionsCUDA bcs(e->boundary_conditions);

        if(cudaMemcpyToSymbol(cuda_bcs, &bcs, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}

__global__ 
void engine_cuda_boundary_conditions_finalize_device() {
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    cuda_bcs.finalize();
}

/**
 * @brief Finalize boundary conditions on device. 
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_cuda_boundary_conditions_finalize(struct engine *e) {

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        engine_cuda_boundary_conditions_finalize_device<<<1, 1>>>();
        if(cudaPeekAtLastError() != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}

/**
 * @brief Refresh boundary conditions on device. Can be safely called while on device. 
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_cuda_boundary_conditions_refresh(struct engine *e) {
    
    if(engine_cuda_boundary_conditions_finalize(e) < 0)
        return error(engine_err);

    if(engine_cuda_boundary_conditions_load(e) < 0)
        return error(engine_err);

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}


__device__ void potential_eval_cuda ( struct MxPotential *p , float r2 , float *e , float *f ) {

    int ind, k;
    float x, ee, eff, *c, ir, r;
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    
    /* compute the interval index */
    ind = fmaxf( 0.0f , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );
    
    /* get the table offset */
    c = &(p->c[ind * potential_chunk]);
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    ee = c[2] * x + c[3];
    eff = c[2];
    #pragma unroll
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
    }

    /* store the result */
    *e = ee; *f = eff * c[1] * ir;
        
    TIMER_TOC(tid_potential)
        
}


/** 
 * @brief Evaluates the given potential at the given point (interpolated).
 *
 * @param p The #potential to be evaluated.
 * @param ri Radius of the ith particle. 
 * @param rj Radius of the jth particle. 
 * @param r2 The radius at which it is to be evaluated, squared.
 * @param e Pointer to a floating-point value in which to store the
 *      interaction energy.
 * @param f Pointer to a floating-point value in which to store the
 *      magnitude of the interaction force divided by r.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 */
__device__ inline void potential_eval_ex_cuda (struct MxPotential *p, float ri, float rj, float r2, float *e, float *f, bool *result) {

    int ind, k;
    float x, ee, eff, *c, ir, r;
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    
    // cutoff min value, eval at lowest func interpolation.
    r = r < p->a ? p->a : r;
    
    if(p->flags & POTENTIAL_SCALED) {
        r = r / (ri + rj);
    }
    else if(p->flags & POTENTIAL_SHIFTED) {
        r = r - (ri + rj) + p->r0_plusone;
    }
    
    /* compute the interval index */
    ind = fmaxf( 0.0f , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );

    if(r > p->b || ind > p->n) {
        *result = false;
        return;
    }
    
    /* get the table offset */
    c = &(p->c[ind * potential_chunk]);
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    ee = c[2] * x + c[3];
    eff = c[2];
    #pragma unroll
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
    }

    /* store the result */
    *e = ee; *f = eff * c[1] * ir;

    *result = true;
        
    TIMER_TOC(tid_potential)
        
}


__device__ inline void dpd_eval_cuda(MxPotential pot, MxParticleCUDA pi, MxParticleCUDA pj, float3 dpd_cfs, float *dx, float r2, float *e, float *fi, float *fj, bool *result) {

    float delta = rsqrtf(cuda_dt);
    
    float ri = pi.radius;
    float rj = pj.radius;
    bool shifted = pot.flags & POTENTIAL_SHIFTED;
    
    float cutoff = shifted ? (pot.b + ri + rj) : pot.b;
    
    if(r2 > cutoff * cutoff) {
        *result = false;
        return;
    }
    
    float r = sqrtf(r2);
    
    if(r < pot.a) {
        *result = false;
        return;
    }
    
    // unit vector
    float3 unit_vec{dx[0] / r, dx[1] / r, dx[2] / r};
    
    float3 v{pi.v.x - pj.v.x, pi.v.y - pj.v.y, pi.v.z - pj.v.z};
    
    float shifted_r = shifted ? r - ri - rj : r;
    
    // conservative force
    float omega_c = shifted_r < 0.f ?  1.f : (1 - shifted_r / cutoff);
    
    float fc = dpd_cfs.x * omega_c;
    
    // dissapative force
    float omega_d = omega_c * omega_c;
    
    float fd = - dpd_cfs.y * omega_d * (unit_vec.x * v.x + unit_vec.y * v.y + unit_vec.z * v.z);
    
    float fr = dpd_cfs.z * omega_c * delta;
    
    float f = fc + fd + fr;
    
    fi[0] += f * unit_vec.x;
    fi[1] += f * unit_vec.y;
    fi[2] += f * unit_vec.z;
    fj[0] -= f * unit_vec.x;
    fj[1] -= f * unit_vec.y;
    fj[2] -= f * unit_vec.z;
    
    // TODO: correct energy
    *e = 0;

    *result = true;

}


__device__ inline void dpd_boundary_eval_cuda(MxPotential pot, MxParticleCUDA pi, float3 velocity, float3 dpd_cfs, float *dx, float r2, float *e, float *force, bool *result) {

    float delta = rsqrtf(cuda_dt);
    
    float cutoff = pot.b;
    
    if(r2 > cutoff * cutoff) {
        *result = false;
        return;
    }
    
    float r = sqrtf(r2);
    
    if(r < pot.a) {
        *result = false;
        return;
    }
    
    // unit vector
    float3 unit_vec{dx[0] / r, dx[1] / r, dx[2] / r};
    
    float3 v{pi.v.x - velocity.x, pi.v.y - velocity.y, pi.v.z - velocity.z};
    
    // conservative force
    float omega_c = 1 - r / cutoff;
    
    float fc = dpd_cfs.x * omega_c;
    
    // dissapative force
    float omega_d = omega_c * omega_c;
    
    float fd = - dpd_cfs.y * omega_d * (unit_vec.x * v.x + unit_vec.y * v.y + unit_vec.z * v.z);
    
    float fr = dpd_cfs.z * omega_c * delta;
    
    float f = fc + fd + fr;
    
    force[0] += f * unit_vec.x;
    force[1] += f * unit_vec.y;
    force[2] += f * unit_vec.z;
    
    // TODO: correct energy
    *e = 0;

    *result = true;

}


// Underlying evaluation call; using templates to eliminate instructional overhead
template<uint32_t kind> 
__device__ inline void _potential_eval_super_ex_cuda(MxPotentialCUDA p_cuda, 
                                                    MxParticleCUDA pi, 
                                                    MxParticleCUDA pj, 
                                                    float *dx, 
                                                    float r2, 
                                                    float *epot, 
                                                    float *fi, 
                                                    float *fj, 
                                                    bool *result) 
{
    float e;
    *result = false;
    float num_dens, _dx[3], _r2;
    auto flags = p_cuda.pot.flags;

    auto a = p_cuda.pot.a;
    
    if(flags & POTENTIAL_PERIODIC) {
        // Images don't contribute to density
        num_dens = 0.0;
        // Assuming elsewhere there's a corresponding potential in the opposite direction
        _dx[0] = dx[0] - p_cuda.pot.offset[0];
        _dx[1] = dx[1] - p_cuda.pot.offset[1];
        _dx[2] = dx[2] - p_cuda.pot.offset[2];
        _r2 = _dx[0]*_dx[0] + _dx[1]*_dx[1] + _dx[2]*_dx[2];
    }
    else {
        _r2 = r2;
        for (int k = 0; k < 3; k++) _dx[k] = dx[k];
    }
    
    // if distance is less that potential min distance, define random
    // for repulsive force.
    if(_r2 < a * a) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        curandState *rand_norm = &cuda_rand_norm[tid];
        _dx[0] = curand_normal(rand_norm);
        _dx[1] = curand_normal(rand_norm);
        _dx[2] = curand_normal(rand_norm);
        float len = std::sqrt(_dx[0] * _dx[0] + _dx[1] * _dx[1] + _dx[2] * _dx[2]);
        _dx[0] = _dx[0] * p_cuda.pot.a / len;
        _dx[1] = _dx[1] * p_cuda.pot.a / len;
        _dx[2] = _dx[2] * p_cuda.pot.a / len;
        _r2 = p_cuda.pot.a * p_cuda.pot.a;
    }
    
    if(kind == POTENTIAL_KIND_DPD) {
        /* update the forces if part in range */
        dpd_eval_cuda(p_cuda.pot, pi, pj, p_cuda.dpd_cfs, _dx, _r2 , &e, fi, fj, result);
        
        if(*result) {
            
            // the number density is a union after the force 3-vector.
            fi[3] += num_dens;
            fj[3] += num_dens;
            
            /* tabulate the energy */
            *epot += e;
        }
    }
    else if(kind == POTENTIAL_KIND_BYPARTICLES) {
        // Currently not supported... ignoring
        *result = false;
    }
    else if(kind == POTENTIAL_KIND_COMBINATION) {
        if(flags & POTENTIAL_SUM) {
            bool resulta, resultb;
            potential_eval_super_ex_cuda(MxPotentialCUDA(p_cuda.pot.pca, false), pi, pj, _dx, _r2, epot, fi, fj, &resulta);
            potential_eval_super_ex_cuda(MxPotentialCUDA(p_cuda.pot.pcb, false), pi, pj, _dx, _r2, epot, fi, fj, &resultb);
            *result = resulta || resultb;
        }
    }
    else {
        float f;
    
        /* update the forces if part in range */
        potential_eval_ex_cuda(&p_cuda.pot, pi.radius, pj.radius, _r2 , &e , &f, result);
        if(*result) {
            
            for (int k = 0 ; k < 3 ; k++ ) {
                float w = f * _dx[k];
                fi[k] -= w;
                fj[k] += w;
            }
            
            // the number density is a union after the force 3-vector.
            fi[3] += num_dens;
            fj[3] += num_dens;
            
            /* tabulate the energy */
            *epot += e;
        }
    }
}

__device__ inline void potential_eval_super_ex_cuda(MxPotentialCUDA p_cuda, 
                                                    MxParticleCUDA pi, 
                                                    MxParticleCUDA pj, 
                                                    float *dx, 
                                                    float r2, 
                                                    float *epot, 
                                                    float *fi, 
                                                    float *fj, 
                                                    bool *result) 
{
    uint32_t kind = p_cuda.pot.kind;

    if(kind == POTENTIAL_KIND_POTENTIAL) {
        _potential_eval_super_ex_cuda<POTENTIAL_KIND_POTENTIAL>(p_cuda, pi, pj, dx, r2, epot, fi, fj, result);
    }
    else if(kind == POTENTIAL_KIND_COMBINATION) {
        _potential_eval_super_ex_cuda<POTENTIAL_KIND_COMBINATION>(p_cuda, pi, pj, dx, r2, epot, fi, fj, result);
    }
    else if(kind == POTENTIAL_KIND_DPD) {
        _potential_eval_super_ex_cuda<POTENTIAL_KIND_DPD>(p_cuda, pi, pj, dx, r2, epot, fi, fj, result);
    }
}

// Underlying evaluation call; using templates to eliminate instructional overhead
template<uint32_t kind> 
__device__ inline void _boundary_eval_cuda_ex_cuda(MxPotentialCUDA p_cuda, 
                                                   MxParticleCUDA part, 
                                                   MxBoundaryConditionCUDA bc, 
                                                   float *dx, 
                                                   float r2, 
                                                   float *epot, 
                                                   float *force, 
                                                   bool *result) 
{
    float e;
    *result = false;
    float _dx[3], _r2;

    auto a = p_cuda.pot.a;
    
    // if distance is less that potential min distance, define random
    // for repulsive force.
    if(r2 < a * a) {
        _dx[0] = bc.normal.x * a;
        _dx[1] = bc.normal.y * a;
        _dx[2] = bc.normal.z * a;
        _r2 = a * a;
    }
    else {
        _r2 = r2;
        for (int k = 0; k < 3; k++) _dx[k] = dx[k];
    }
    
    if(kind == POTENTIAL_KIND_DPD) {
        /* update the forces if part in range */
        dpd_boundary_eval_cuda(p_cuda.pot, part, bc.velocity, p_cuda.dpd_cfs, _dx, _r2, &e, force, result);
        
        if(*result) {
            /* tabulate the energy */
            *epot += e;
        }
    }
    else {
        float f;
    
        /* update the forces if part in range */
        potential_eval_ex_cuda(&p_cuda.pot, part.radius, bc.radius, _r2 , &e , &f, result);
        if(*result) {
            
            for (int k = 0 ; k < 3 ; k++ ) {
                force[k] -= f * _dx[k];
            }
            
            /* tabulate the energy */
            *epot += e;
        }
    }
}

__device__ inline void boundary_eval_cuda_ex_cuda(MxPotentialCUDA p_cuda, 
                                                  MxParticleCUDA part, 
                                                  MxBoundaryConditionCUDA bc, 
                                                  float *dx, 
                                                  float r2, 
                                                  float *epot, 
                                                  float *force, 
                                                  bool *result) 
{
    uint32_t kind = p_cuda.pot.kind;

    if(kind == POTENTIAL_KIND_POTENTIAL) {
        _boundary_eval_cuda_ex_cuda<POTENTIAL_KIND_POTENTIAL>(p_cuda, part, bc, dx, r2, epot, force, result);
    }
    else if(kind == POTENTIAL_KIND_DPD) {
        _boundary_eval_cuda_ex_cuda<POTENTIAL_KIND_DPD>(p_cuda, part, bc, dx, r2, epot, force, result);
    }
}

__device__ 
void boundary_eval_cuda(MxParticleCUDA part, float3 cell_dim, unsigned int cell_flags, float *force, float *epot, bool *result) {
    
    MxPotentialCUDA potc;
    MxPotential *pot;
    float r;
    MxBoundaryConditionCUDA bc;
    bool eval_result = false;
    
    float dx[3] = {0.f, 0.f, 0.f};
    
    if(cell_flags & cell_active_left) {
        bc = cuda_bcs.bcs[0];
        potc = bc.pots[part.typeId];
        if(!potc.empty) {
            r = part.x.x;
            pot = &potc.pot;
            if(r < pot->b) {
                dx[0] = r;
                boundary_eval_cuda_ex_cuda(potc, part, bc, dx, r * r, epot, force, &eval_result);
                *result |= eval_result;
            }
        }
    }
    
    if(cell_flags & cell_active_right) {
        bc = cuda_bcs.bcs[1];
        potc = bc.pots[part.typeId];
        if(!potc.empty) {
            r = cell_dim.x - part.x.x;
            pot = &potc.pot;
            if(r < pot->b) {
                dx[0] = -r;
                boundary_eval_cuda_ex_cuda(potc, part, bc, dx, r * r, epot, force, &eval_result);
                *result |= eval_result;
            }
        }
    }
    
    if(cell_flags & cell_active_front) {
        bc = cuda_bcs.bcs[2];
        potc = bc.pots[part.typeId];
        if(!potc.empty) {
            r = part.x.y;
            pot = &potc.pot;
            if(r < pot->b) {
                dx[1] = r;
                boundary_eval_cuda_ex_cuda(potc, part, bc, dx, r * r, epot, force, &eval_result);
                *result |= eval_result;
            }
        }
    }
    
    if(cell_flags & cell_active_back) {
        bc = cuda_bcs.bcs[3];
        potc = bc.pots[part.typeId];
        if(!potc.empty) {
            r = cell_dim.y - part.x.y;
            pot = &potc.pot;
            if(r < pot->b) {
                dx[1] = -r;
                boundary_eval_cuda_ex_cuda(potc, part, bc, dx, r * r, epot, force, &eval_result);
                *result |= eval_result;
            }
        }
    }
    
    if(cell_flags & cell_active_bottom) {
        bc = cuda_bcs.bcs[4];
        potc = bc.pots[part.typeId];
        if(!potc.empty) {
            r = part.x.z;
            pot = &potc.pot;
            if(r < pot->b) {
                dx[2] = r;
                boundary_eval_cuda_ex_cuda(potc, part, bc, dx, r * r, epot, force, &eval_result);
                *result |= eval_result;
            }
        }
    }
    
    if(cell_flags & cell_active_top) {
        bc = cuda_bcs.bcs[5];
        potc = bc.pots[part.typeId];
        if(!potc.empty) {
            r = cell_dim.z - part.x.z;
            pot = &potc.pot;
            if(r < pot->b) {
                dx[2] = -r;
                boundary_eval_cuda_ex_cuda(potc, part, bc, dx, r * r, epot, force, &eval_result);
                *result |= eval_result;
            }
        }
    }

}


__device__ inline 
void flux_fick_cuda(MxFluxCUDA flux, int i, float si, float sj, float *result) {
    *result *= flux.coef[i] * (si - sj);
}

__device__ inline 
void flux_secrete_cuda(MxFluxCUDA flux, int i, float si, float sj, float *result) {
    float q = flux.coef[i] * (si - flux.target[i]);
    float scale = q > 0.f;  // forward only, 1 if > 0, 0 if < 0.
    *result *= scale * q;
}

__device__ inline 
void flux_uptake_cuda(MxFluxCUDA flux, int i, float si, float sj, float *result) {
    float q = flux.coef[i] * (flux.target[i] - sj) * si;
    float scale = q > 0.f;
    *result *= scale * q;
}

__device__ inline 
void flux_eval_ex_cuda(MxFluxesCUDA fluxes, float r, float *states_i, float *states_j, int type_i, int type_j, float *qvec_i) {

    // Do calculations

    float ssi, ssj;
    float q;
    
    MxFluxCUDA flux = fluxes.fluxes[0];
    float term = 1. - r / cuda_cutoff;
    term = term * term;

    int qind;
    
    for(int i = 0; i < flux.size; ++i) {

        if(type_i == flux.type_ids[i].a) {
            qind = flux.indices_a[i];
            ssi = states_i[qind];
            ssj = states_j[flux.indices_b[i]];
            q = - term;
        }
        else {
            qind = flux.indices_b[i];
            ssi = states_j[flux.indices_a[i]];
            ssj = states_i[qind];
            q = term;
        }

        switch(flux.kinds[i]) {
            case FLUX_FICK:
                flux_fick_cuda(flux, i, ssi, ssj, &q);
                break;
            case FLUX_SECRETE:
                flux_secrete_cuda(flux, i, ssi, ssj, &q);
                break;
            case FLUX_UPTAKE:
                flux_uptake_cuda(flux, i, ssi, ssj, &q);
                break;
            default:
                __builtin_unreachable();
        }

        qvec_i[qind] += q - 0.5 * flux.decay_coef[i] * states_i[qind];
    }
}

__device__ inline 
void flux_eval_ex_cuda(MxFluxesCUDA fluxes, float r, float *states_i, float *states_j, int type_i, int type_j, float *qvec_i, float *qvec_j) {

    // Do calculations
    
    MxFluxCUDA flux = fluxes.fluxes[0];
    float term = 1. - r / cuda_cutoff;
    term = term * term;

    float *qi, *qj, *si, *sj;
    
    for(int i = 0; i < flux.size; ++i) {

        if(type_i == flux.type_ids[i].a) {
            si = states_i;
            sj = states_j;
            qi = qvec_i;
            qj = qvec_j;
        }
        else {
            si = states_j;
            sj = states_i;
            qi = qvec_j;
            qj = qvec_i;
        }
        
        float ssi = si[flux.indices_a[i]];
        float ssj = sj[flux.indices_b[i]];
        float q =  term;
        float mult;
        
        switch(flux.kinds[i]) {
            case FLUX_FICK:
                flux_fick_cuda(flux, i, ssi, ssj, &mult);
                q *= mult;
                break;
            case FLUX_SECRETE:
                flux_secrete_cuda(flux, i, ssi, ssj, &mult);
                q *= mult;
                break;
            case FLUX_UPTAKE:
                flux_uptake_cuda(flux, i, ssi, ssj, &mult);
                q *= mult;
                break;
            default:
                __builtin_unreachable();
        }
        
        float half_decay = flux.decay_coef[i] * 0.5;
        qi[flux.indices_a[i]] -= q + half_decay * ssi;
        qj[flux.indices_b[i]] += q - half_decay * ssj;
    }
}


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
__device__ void runner_dopair_unsorted_cuda ( MxParticleCUDA *parts_i , int count_i , MxParticleCUDA *parts_j , int count_j , float *forces_i , float *forces_j , float *shift , float *epot_global ) {
    int k, pid, pjd, ind, wrap_i, threadID;
    int pjoff;
    int pind;
    float epot = 0.0f, dx[3], pjf[4], r2;
    float ee = 0.0f;
    MxParticleCUDA pi, pj;
    float4 pjx;
    bool eval_result;
    float number_density;
    MxPotentialCUDA *cu_pots;
    unsigned int *cu_pind;
    bool iscluster_maybe;
    
    TIMER_TIC

    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Get the wraps. */
    wrap_i = (count_i < cuda_frame) ? cuda_frame : count_i;

    /* Loop over the particles in cell_j, frame-wise. */
    for ( pjd = threadID ; pjd < count_j ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        pj = parts_j[ pjd ];
        pjoff = pj.typeId * cuda_maxtype;
        pjx = float4(pj.x);
        pjx.x += shift[0]; pjx.y += shift[1]; pjx.z += shift[2];
        pjf[0] = 0.0f; pjf[1] = 0.0f; pjf[2] = 0.0f; pjf[3] = 0.0f;
        iscluster_maybe = pj.flags & PARTICLE_CLUSTER;
        
        /* Loop over the particles in cell_i. */
        for ( ind = 0 ; ind < wrap_i ; ind++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pid = ind + threadID ) >= wrap_i )
                pid -= wrap_i;
            
            if ( pid < count_i ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                pi = parts_i[ pid ];

                /* Compute the radius between pi and pj. */
                r2 = 0.0f;
                dx[0] = pi.x.x - pjx.x; r2 += dx[0]*dx[0];
                dx[1] = pi.x.y - pjx.y; r2 += dx[1]*dx[1];
                dx[2] = pi.x.z - pjx.z; r2 += dx[2]*dx[2];

                if ( r2 >= cuda_cutoff2 ) {
                    continue;
                }

                w_cubic_spline_cuda(r2, cuda_cutoff, &number_density);

                pjf[3] += number_density;
                forces_i[4*pid + 3] += number_density;

                if(iscluster_maybe && pi.flags & PARTICLE_CLUSTER && pi.clusterId == pj.clusterId) {
                    cu_pots = cuda_pots_cluster;
                    cu_pind = cuda_pind_cluster;
                }
                else {
                    cu_pots = cuda_pots;
                    cu_pind = cuda_pind;
                }

                pind = cu_pind[pjoff + pi.typeId];
                if(pind == 0) 
                    continue;
            
                /* Interact particles pi and pj. */
                potential_eval_super_ex_cuda(cu_pots[pind], pi, pj, dx, r2, &ee, &forces_i[4*pid], pjf, &eval_result);

                /* Store the interaction force and energy. */
                epot += ee;

            } /* valid pid? */
        
        } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 4 ; k++ )
            forces_j[ 4*pjd + k ] += pjf[k];

    } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;

    TIMER_TOC(tid_pair)
        
}


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
__device__ void runner_dopair4_unsorted_cuda ( MxParticleCUDA *parts_i , int count_i , MxParticleCUDA *parts_j , int count_j , float *forces_i , float *forces_j , float *shift , float *epot_global ) {
    int k, pjd, ind, wrap_i, threadID;
    int pjoff;
    MxParticleCUDA pi[4], pj;
    float4 pjx;
    int4 pot, pid;
    char4 valid;
    float4 r2, ee;
    float epot = 0.0f, dx[12], pjf[4];
    bool eval_result;
    float4 number_density;
    MxPotentialCUDA *cu_pots[4];
    unsigned int *cu_pind[4];
    bool iscluster_maybe;
    
    TIMER_TIC

    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Get the wraps. */
    wrap_i = (count_i < cuda_frame) ? cuda_frame : count_i;

    /* Loop over the particles in cell_j, frame-wise. */
    for ( pjd = threadID ; pjd < count_j ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        pj = parts_j[ pjd ];
        pjx = float4(pj.x);
        pjoff = pj.typeId * cuda_maxtype;
        pjx.x += shift[0]; pjx.y += shift[1]; pjx.z += shift[2];
        for ( k = 0 ; k < 4 ; k++ )
            pjf[k] = 0.0f;

        iscluster_maybe = pj.flags & PARTICLE_CLUSTER;
        
        /* Loop over the particles in cell_i. */
        for ( ind = 0 ; ind < wrap_i ; ind += 4 ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pid.x = ind + threadID ) >= wrap_i )
                pid.x -= wrap_i;
            if ( ( pid.y = ind + threadID + 1 ) >= wrap_i )
                pid.y -= wrap_i;
            if ( ( pid.z = ind + threadID + 2 ) >= wrap_i )
                pid.z -= wrap_i;
            if ( ( pid.w = ind + threadID + 3 ) >= wrap_i )
                pid.w -= wrap_i;
                
            /* Get the particle pointers. */
            pi[0] = ( valid.x = ( pid.x < count_i ) ) ? parts_i[ pid.x] : pj;
            pi[1] = ( valid.y = ( pid.y < count_i ) && ( ind + 1 < wrap_i ) ) ? parts_i[ pid.y ] : pj;
            pi[2] = ( valid.z = ( pid.z < count_i ) && ( ind + 2 < wrap_i ) ) ? parts_i[ pid.z ] : pj;
            pi[3] = ( valid.w = ( pid.w < count_i ) && ( ind + 3 < wrap_i ) ) ? parts_i[ pid.w ] : pj;
            if(iscluster_maybe) {
                for(k = 0; k < 4; k++) {
                    if(pi[k].flags & PARTICLE_CLUSTER && pi[k].clusterId == pj.clusterId) {
                        cu_pind[k] = cuda_pind_cluster;
                        cu_pots[k] = cuda_pots_cluster;
                    }
                    else {
                        cu_pind[k] = cuda_pind;
                        cu_pots[k] = cuda_pots;
                    }
                }
            }
            else {
                for(k = 0; k < 4; k++) {
                    cu_pind[k] = cuda_pind;
                    cu_pots[k] = cuda_pots;
                }
            }
            
            /* Compute the pairwise distances. */
            r2 = make_float4( 0.0f );
            dx[0] = pjx.x - pi[0].x.x; r2.x += dx[0] * dx[0];
            dx[1] = pjx.y - pi[0].x.y; r2.x += dx[1] * dx[1];
            dx[2] = pjx.z - pi[0].x.z; r2.x += dx[2] * dx[2];
            dx[3] = pjx.x - pi[1].x.x; r2.y += dx[3] * dx[3];
            dx[4] = pjx.y - pi[1].x.y; r2.y += dx[4] * dx[4];
            dx[5] = pjx.z - pi[1].x.z; r2.y += dx[5] * dx[5];
            dx[6] = pjx.x - pi[2].x.x; r2.z += dx[6] * dx[6];
            dx[7] = pjx.y - pi[2].x.y; r2.z += dx[7] * dx[7];
            dx[8] = pjx.z - pi[2].x.z; r2.z += dx[8] * dx[8];
            dx[9] = pjx.x - pi[3].x.x; r2.w += dx[9] * dx[9];
            dx[10] = pjx.y - pi[3].x.y; r2.w += dx[10] * dx[10];
            dx[11] = pjx.z - pi[3].x.z; r2.w += dx[11] * dx[11];

            /* Get the potentials. */
            valid.x = ( valid.x && r2.x < cuda_cutoff2 );
            valid.y = ( valid.y && r2.y < cuda_cutoff2 );
            valid.z = ( valid.z && r2.z < cuda_cutoff2 );
            valid.w = ( valid.w && r2.w < cuda_cutoff2 );
            pot.x = valid.x ? cu_pind[0][ pjoff + pi[0].typeId ] : 0;
            pot.y = valid.y ? cu_pind[1][ pjoff + pi[1].typeId ] : 0;
            pot.z = valid.z ? cu_pind[2][ pjoff + pi[2].typeId ] : 0;
            pot.w = valid.w ? cu_pind[3][ pjoff + pi[3].typeId ] : 0;
            
            /* Update the forces. */
            if ( valid.x ) {
                w_cubic_spline_cuda(r2.x, cuda_cutoff, &number_density.x);
                pjf[3] += number_density.x;
                forces_i[4*pid.x + 3] += number_density.x;
                if(pot.x != 0) {
                    potential_eval_super_ex_cuda(cu_pots[0][pot.x], pi[0], pj, &dx[0], r2.x, &ee.x, pjf, &forces_i[4*pid.x], &eval_result);
                    epot += ee.x;
                }
            }
            if ( valid.y ) {
                w_cubic_spline_cuda(r2.y, cuda_cutoff, &number_density.y);
                pjf[3] += number_density.y;
                forces_i[4*pid.y + 3] += number_density.y;
                if(pot.y != 0) {
                    potential_eval_super_ex_cuda(cu_pots[1][pot.y], pi[1], pj, &dx[3], r2.y, &ee.y, pjf, &forces_i[4*pid.y], &eval_result);
                    epot += ee.y;
                }
            }
            if ( valid.z ) {
                w_cubic_spline_cuda(r2.z, cuda_cutoff, &number_density.z);
                pjf[3] += number_density.z;
                forces_i[4*pid.z + 3] += number_density.z;
                if(pot.z != 0) {
                    potential_eval_super_ex_cuda(cu_pots[2][pot.z], pi[2], pj, &dx[6], r2.z, &ee.z, pjf, &forces_i[4*pid.z], &eval_result);
                    epot += ee.z;
                }
            }
            if ( valid.w ) {
                w_cubic_spline_cuda(r2.w, cuda_cutoff, &number_density.w);
                pjf[3] += number_density.w;
                forces_i[4*pid.w + 3] += number_density.w;
                if(pot.w != 0) {
                    potential_eval_super_ex_cuda(cu_pots[3][pot.w], pi[3], pj, &dx[9], r2.w, &ee.w, pjf, &forces_i[4*pid.w], &eval_result);
                    epot += ee.w;
                }
            }
        
        } /* loop over parts in cell_i. */

        /* Update the force on pj. */
        for ( k = 0 ; k < 4 ; k++ )
            forces_j[ 4*pjd + k ] += pjf[k];
            
    } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
    TIMER_TOC(tid_pair)
        
}


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
__device__ inline void runner_dosort_cuda ( MxParticleCUDA *parts_i , int count_i , unsigned int *sort_i , int sid ) {
    int k, threadID = threadIdx.x;
    MxParticleCUDA pi;
    // int4 spid;
    float nshift, shift[3], shiftn[3];
    
    TIMER_TIC
    
    /* Get the shift vector from the sid. */
    shift[0] = cuda_shift[ 3*sid + 0 ] * cuda_h[0];
    shift[1] = cuda_shift[ 3*sid + 1 ] * cuda_h[1];
    shift[2] = cuda_shift[ 3*sid + 2 ] * cuda_h[2];

    /* Pre-compute the inverse norm of the shift. */
    nshift = sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    shiftn[0] = cuda_shiftn[ 3*sid + 0 ];
    shiftn[1] = cuda_shiftn[ 3*sid + 1 ];
    shiftn[2] = cuda_shiftn[ 3*sid + 2 ];



    /* Pack the parts into the sort arrays. */
    for ( k = threadID ; k < count_i ; k += blockDim.x ) {
        pi = parts_i[ k ];
        sort_i[k] = ( k << 16 ) | (unsigned int)( cuda_dscale * (nshift + pi.x.x*shiftn[0] + pi.x.y*shiftn[1] + pi.x.z*shiftn[2]) );
        }

    TIMER_TOC(tid_pack)
    __syncthreads();
    /* Sort using normalized bitonic sort. */
    cuda_sort_descending( sort_i , count_i );

}
    
    
/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
__device__ void runner_dopair_cuda ( MxParticleCUDA *parts_i , int count_i , MxParticleCUDA *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , float *epot_global ) {
    int k, pid, pjd, spid, spjd, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int dmaxdist;
    MxParticleCUDA pi, pj;
    float4 pix;
    int pind;
    float epot = 0.0f, r2, ee = 0.0f;
    float dx[3], pif[4];
    bool eval_result;
    float number_density;
    MxPotentialCUDA *cu_pots;
    unsigned int *cu_pind;
    bool iscluster_maybe;
    
    TIMER_TIC

    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;

    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[count_j-cj] & 0xffff ) + dshift - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dmaxdist )
            cj -= 1;
        if ( cj == 0 )
            break;
        else if ( cj < cuda_frame )
            wrap = max( cj , min( count_i - (pid & ~(cuda_frame - 1)) , cuda_frame ) );
        else
            wrap = cj;
            
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[pid] >> 16;
        pi = parts_i[ spid ];
        pix = float4(pi.x);
        pioff = pi.typeId * cuda_maxtype;
        pix.x -= shift[0]; pix.y -= shift[1]; pix.z -= shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f; pif[3] = 0.0f;
        iscluster_maybe = pi.flags & PARTICLE_CLUSTER;
        
        /* Loop over the particles in cell_i. */

        for ( pjdid = 0 ; pjdid < wrap ; pjdid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd = pjdid + threadID ) >= wrap )
                pjd -= wrap;
            
            /* Do we have a pair? */
            if ( pjd < cj ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                spjd = sort_j[count_j-1-pjd] >> 16;
                pj = parts_j[ spjd ];

                /* Compute the radius between pi and pj. */
                r2 = 0.0f;
                dx[0] = pix.x - pj.x.x; r2 += dx[0]*dx[0];
                dx[1] = pix.y - pj.x.y; r2 += dx[1]*dx[1];
                dx[2] = pix.z - pj.x.z; r2 += dx[2]*dx[2];
                
                if ( r2 >= cuda_cutoff2 ) {
                    continue;
                }

                w_cubic_spline_cuda(r2, cuda_cutoff, &number_density);

                pif[3] += number_density;
                forces_j[4*spjd + 3] += number_density;

                if(iscluster_maybe && pj.flags & PARTICLE_CLUSTER && pi.clusterId == pj.clusterId) {
                    cu_pots = cuda_pots_cluster;
                    cu_pind = cuda_pind_cluster;
                }
                else {
                    cu_pots = cuda_pots;
                    cu_pind = cuda_pind;
                }

                pind = cu_pind[pioff + pj.typeId];
                if(pind == 0) 
                    continue;

                /* Interact particles pi and pj. */
                potential_eval_super_ex_cuda(cu_pots[pind], pi, pj, dx, r2, &ee, pif, &forces_j[4*spjd], &eval_result);

                /* Store the interaction energy. */
                epot += ee;

            } /* do we have a pair? */
        
        } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 4 ; k++ )
            forces_i[ 4*spid + k ] += pif[k];
        
    } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
    TIMER_TOC(tid_pair)
    
}


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
template<bool is_stateful> 
__device__  void runner_dopair_left_cuda ( MxParticleCUDA * parts_i , int count_i , MxParticleCUDA *parts_j , int count_j , float *forces_i , float *forces_j , float *fluxes_i , float *fluxes_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , unsigned int nr_states , float *epot_global ) {
    int k, pjd, spid, spjd, threadID;
    unsigned int dmaxdist, di;
    MxParticleCUDA pi, pj;
    float4 pix;
    int pind, i;
    float epot = 0.0f, r2, r, ee = 0.0f;
    float dx[3], pif[4], pjf[4], pifx[MX_SIMD_SIZE];
    bool eval_result;
    float number_density;
    MxPotentialCUDA *cu_pots;
    unsigned int *cu_pind;
    bool iscluster_maybe;
    MxFluxesCUDA *fluxes_cuda;
    unsigned int *cu_fxind, fxind;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;

    if(is_stateful) {
        MxFluxCUDA_getFluxes(&cu_fxind, &fluxes_cuda);
    }
    
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;
    
    for ( i = threadID ; i < count_i ;  i += blockDim.x  ) {

        di = sort_i[i]&0xffff;
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[i] >> 16;
        pi = parts_i[ spid ];
        pix = float4(pi.x);
        pix.x -= shift[0]; pix.y -= shift[1]; pix.z -= shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f; pif[3] = 0.0f;
        iscluster_maybe = pi.flags & PARTICLE_CLUSTER;

        if(is_stateful) {
            memset(pifx, 0.f, nr_states * sizeof(float));
        }

        /* Loop over the particles in cell_j. */
        for ( pjd = count_j-1 ; pjd >=0 && (sort_j[pjd]&0xffff)+dshift<=dmaxdist+di ; pjd-- ) {
                 
            /* Get a handle on the wrapped particle pid in cell_i. */

            spjd = sort_j[pjd] >> 16;
            pj = parts_j[ spjd ];

            /* Compute the radius between pi and pj. */
            r2 = 0.0f;
            dx[0] = pix.x - pj.x.x; r2 += dx[0]*dx[0];
            dx[1] = pix.y - pj.x.y; r2 += dx[1]*dx[1];
            dx[2] = pix.z - pj.x.z; r2 += dx[2]*dx[2];

            if ( r2 >= cuda_cutoff2 ) {
                continue;
            }

            w_cubic_spline_cuda(r2, cuda_cutoff, &number_density, &r);

            pif[3] += number_density;

            if(iscluster_maybe && pj.flags & PARTICLE_CLUSTER && pi.clusterId == pj.clusterId) {
                cu_pots = cuda_pots_cluster;
                cu_pind = cuda_pind_cluster;
            }
            else {
                cu_pots = cuda_pots;
                cu_pind = cuda_pind;
            }

            pind = cu_pind[ pi.typeId * cuda_maxtype + pj.typeId ];
            if(pind != 0) {
                /* Interact particles pi and pj. */
                potential_eval_super_ex_cuda(cu_pots[pind], pi, pj, dx, r2, &ee, pif, pjf, &eval_result);

                /* Store the interaction energy. */
                epot += ee;
            }

            if(is_stateful) { 
                fxind = cu_fxind[pi.typeId * cuda_maxtype + pj.typeId];
                if(fxind != 0)
                    flux_eval_ex_cuda(fluxes_cuda[fxind], r, pi.states, pj.states, pi.typeId, pj.typeId, pifx);
            }

        } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 4 ; k++ )
        	atomicAdd( &forces_i[ 4*spid + k], pif[k] );

        if(is_stateful) 
            for(k = 0; k < nr_states; k++)
                atomicAdd(&fluxes_i[nr_states*spid + k], pifx[k]);
        
    } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
    TIMER_TOC(tid_pair)
    
}




/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
template<bool is_stateful> 
__device__ void runner_dopair_right_cuda ( MxParticleCUDA *parts_i , int count_i , MxParticleCUDA *parts_j , int count_j , float *forces_i , float *forces_j , float *fluxes_i , float *fluxes_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , unsigned int nr_states , float *epot_global ) {
    int k, pjd, spid, spjd, threadID;
    unsigned int dmaxdist, dj;
    MxParticleCUDA pi, pj;
    float4 pix;
    int pind, i;
    float epot = 0.0f, r2, r, ee = 0.0f;
    float dx[3], pif[4], pjf[4], pifx[MX_SIMD_SIZE];
    bool eval_result;
    float number_density;
    MxPotentialCUDA *cu_pots;
    unsigned int *cu_pind;
    bool iscluster_maybe;
    MxFluxesCUDA *fluxes_cuda;
    unsigned int *cu_fxind, fxind;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;

    if(is_stateful) {
        MxFluxCUDA_getFluxes(&cu_fxind, &fluxes_cuda);
    }
    
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;

    for ( i = threadID ; i < count_i ;  i += blockDim.x  ) {

        dj = sort_i[i]&0xffff;
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[i] >> 16;
        pi = parts_i[ spid ];
        pix = float4(pi.x);
        pix.x += shift[0]; pix.y += shift[1]; pix.z += shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f; pif[3] = 0.0f;
        iscluster_maybe = pi.flags & PARTICLE_CLUSTER;

        if(is_stateful) {
            memset(pifx, 0.f, nr_states * sizeof(float));
        }
        
        /* Loop over the particles in cell_j. */
        for ( pjd = 0 ; pjd < count_j && dj+ dshift <= dmaxdist+(sort_j[pjd]&0xffff) ; pjd++ ) {

            /* Get a handle on the wrapped particle pid in cell_i. */
            spjd = sort_j[pjd] >> 16;
            pj = parts_j[ spjd ];

            /* Compute the radius between pi and pj. */
            r2 = 0.0f;
            dx[0] = pix.x - pj.x.x; r2 += dx[0]*dx[0];
            dx[1] = pix.y - pj.x.y; r2 += dx[1]*dx[1];
            dx[2] = pix.z - pj.x.z; r2 += dx[2]*dx[2];
                
            if ( r2 >= cuda_cutoff2 ) {
                continue;
            }

            w_cubic_spline_cuda(r2, cuda_cutoff, &number_density, &r);

            pif[3] += number_density;

            if(iscluster_maybe && pj.flags & PARTICLE_CLUSTER && pi.clusterId == pj.clusterId) {
                cu_pots = cuda_pots_cluster;
                cu_pind = cuda_pind_cluster;
            }
            else {
                cu_pots = cuda_pots;
                cu_pind = cuda_pind;
            }

            pind = cu_pind[pi.typeId * cuda_maxtype + pj.typeId];
            if(pind != 0) {
                /* Interact particles pi and pj. */
                potential_eval_super_ex_cuda(cu_pots[pind], pi, pj, dx, r2, &ee, pif, pjf, &eval_result);

                /* Store the interaction force and energy. */
                epot += ee;
            }

            if(is_stateful) { 
                fxind = cu_fxind[pi.typeId * cuda_maxtype + pj.typeId];
                if(fxind != 0)
                    flux_eval_ex_cuda(fluxes_cuda[fxind], r, pi.states, pj.states, pi.typeId, pj.typeId, pifx);
            }

        } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 4 ; k++ )
        	atomicAdd( &forces_i[ 4*spid + k] , pif[k]);

        if(is_stateful) 
            for(k = 0; k < nr_states; k++)
                atomicAdd(&fluxes_i[nr_states*spid + k], pifx[k]);
        
    } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
    TIMER_TOC(tid_pair)
    
}

/**
 * @brief Compute the self interactions for the given cell on a CUDA device.
 *
 * @param iparts Array of parts in this cell.
 * @param count Number of parts in the cell.
 * @param parts Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
template<bool is_stateful> 
__device__ void runner_doself_cuda (MxParticleCUDA *parts, int count, int cell_id, float *forces, float *fluxes, unsigned int nr_states, float *epot_global) {
    int k, pid, threadID;
    int pjoff;
    MxParticleCUDA pi, pj;
    int pind, i;
    float epot = 0.0f, dx[3], pif[4], pjf[4], r2, r, ee, pjfx[MX_SIMD_SIZE];
    bool eval_result;
    float number_density;
    unsigned int cell_flags = cuda_cflags[cell_id];
    bool boundary = cell_flags & cell_active_any;
    MxPotentialCUDA *cu_pots;
    unsigned int *cu_pind;
    bool iscluster_maybe;
    MxFluxesCUDA *fluxes_cuda;
    unsigned int *cu_fxind, fxind;
    
    TIMER_TIC

    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;

    if(is_stateful) {
        MxFluxCUDA_getFluxes(&cu_fxind, &fluxes_cuda);
    }

    /* Loop over the particles in the cell, frame-wise. */
    for ( i = threadID ; i < count ;  i += blockDim.x  ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        pj = parts[ i ];
        pjoff = pj.typeId * cuda_maxtype;
        pjf[0] = 0.0f; pjf[1] = 0.0f; pjf[2] = 0.0f; pjf[3] = 0.0f;
        iscluster_maybe = pj.flags & PARTICLE_CLUSTER;

        if(is_stateful) {
            memset(pjfx, 0.f, nr_states * sizeof(float));
        }

        if(boundary) {
            boundary_eval_cuda(pj, cuda_cdims[cell_id], cell_flags, pjf, &epot, &eval_result);
        }
        
        /* Loop over the particles in cell_i. */
        for ( pid = 0 ; pid < count ; pid++ ) {
        	if(i != pid ) {
                /* Get a handle on the wrapped particle pid in cell_i. */
                pi = parts[ pid ];

                /* Compute the radius between pi and pj. */
                r2 = 0.0f;
                dx[0] = pi.x.x - pj.x.x; r2 += dx[0]*dx[0];
                dx[1] = pi.x.y - pj.x.y; r2 += dx[1]*dx[1];
                dx[2] = pi.x.z - pj.x.z; r2 += dx[2]*dx[2];

                if ( r2 >= cuda_cutoff2 ) {
                    continue;
                }

                w_cubic_spline_cuda(r2, cuda_cutoff, &number_density, &r);

                pjf[3] += number_density;

                if(iscluster_maybe && pi.flags & PARTICLE_CLUSTER && pi.clusterId == pj.clusterId) {
                    cu_pots = cuda_pots_cluster;
                    cu_pind = cuda_pind_cluster;
                }
                else {
                    cu_pots = cuda_pots;
                    cu_pind = cuda_pind;
                }

                pind = cu_pind[pjoff + pi.typeId];
                if(pind != 0) {
                    /* Interact particles pi and pj. */
                    potential_eval_super_ex_cuda(cu_pots[pind], pi, pj, dx, r2, &ee, pif, pjf, &eval_result);

                    /* Store the interaction force and energy. */
                    epot += ee;
                }

                if(is_stateful) { 
                    fxind = cu_fxind[pjoff + pi.typeId];
                    if(fxind != 0)
                        flux_eval_ex_cuda(fluxes_cuda[fxind], r, pj.states, pi.states, pj.typeId, pi.typeId, pjfx);
                }

			}
        } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 4 ; k++ )
        	atomicAdd( &forces[ 4*i + k], pjf[k] );

        if(is_stateful) 
            for(k = 0; k < nr_states; k++)
                atomicAdd(&fluxes[nr_states*i + k], pjfx[k]);

    } /* loop over the particles in cell_j. */

    /* Store the potential energy. */
    *epot_global += epot;
        
    TIMER_TOC(tid_self)
    
}



/**
 * @brief Our very own memset for the particle forces as cudaMemsetAsync requires
 *        a device switch when using streams on different devices.
 *
 */
 
__global__ void cuda_memset_float ( float *data , float val , int N ) {

    int k, tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for ( k = tid ; k < N ; k += stride )
        data[k] = val;

    }


/** This set of defines and includes produces kernels with buffers for multiples
 *  of 32 particles up to 512 cuda_maxparts.
 */
 
#define cuda_nparts 32
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 64
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 96
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 128
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 160
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 192
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 224
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 256
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 288
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 320
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 352
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 384
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 416
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 448
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 480
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 512
    #include "runner_cuda_main.h"
#undef cuda_nparts


template<bool is_stateful> 
void engine_nonbond_cuda_kernel_launch(int maxcount, dim3 nr_blocks, dim3 nr_threads, cudaStream_t stream, 
    float *forces_cuda, float *fluxes_next_cuda, int *counts_cuda, int *ind_cuda, int verlet_rebuild, unsigned int nr_states) 
{
    switch ( (maxcount + 31) / 32 ) {
        case 1:
            runner_run_cuda_32<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 2:
            runner_run_cuda_64<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 3:
            runner_run_cuda_96<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 4:
            runner_run_cuda_128<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 5:
            runner_run_cuda_160<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 6:
            runner_run_cuda_192<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 7:
            runner_run_cuda_224<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 8:
            runner_run_cuda_256<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 9:
            runner_run_cuda_288<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 10:
            runner_run_cuda_320<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 11:
            runner_run_cuda_352<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 12:
            runner_run_cuda_384<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 13:
            runner_run_cuda_416<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 14:
            runner_run_cuda_448<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 15:
            runner_run_cuda_480<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        case 16:
            runner_run_cuda_512<is_stateful><<<nr_blocks,nr_threads,0,stream>>>(forces_cuda, fluxes_next_cuda, counts_cuda, ind_cuda, verlet_rebuild, nr_states);
            break;
        }
}


/**
 * @brief Offload and compute the nonbonded interactions on a CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_nonbond_cuda ( struct engine *e ) {

    int k, cid, did, pid, fid, maxcount = 0;
    cudaStream_t stream;
    cudaEvent_t tic, toc_load, toc_run, toc_unload;
    float ms_load, ms_run, ms_unload;
    struct MxParticle *p;
    MxParticleCUDA *parts_cuda = (MxParticleCUDA *)e->parts_cuda_local, *buff_part;
    struct space *s = &e->s;
    FPTYPE maxdist = s->cutoff + 2*s->maxdx;
    int *counts = e->counts_cuda_local[ 0 ], *inds = e->ind_cuda_local[ 0 ];
    float *forces_cuda[ engine_maxgpu ], *fluxes_next_cuda[engine_maxgpu], epot[ engine_maxgpu ], *buff_force, *buff_flux;
    unsigned int nr_states = e->nr_fluxes_cuda - 1;
    #ifdef TIMERS
        float timers[ tid_count ];
        double icpms = 1000.0 / 1.4e9; 
    #endif
    
    /* Create the events. */
    if ( cudaSetDevice( e->devices[e->nr_devices-1] ) ||
         cudaEventCreate( &tic ) != cudaSuccess ||
         cudaEventCreate( &toc_load ) != cudaSuccess ||
         cudaEventCreate( &toc_run ) != cudaSuccess ||
         cudaEventCreate( &toc_unload ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Start the clock on the first stream. */
    if ( cudaEventRecord( tic , (cudaStream_t)e->streams[e->nr_devices-1] ) != cudaSuccess )
        cuda_error(engine_err_cuda);
    
    /* Re-set timers */
    #ifdef TIMERS
        for ( int k = 0 ; k < tid_count ; k++ )
            timers[k] = 0.0f;
        for ( did = 0 ; did < e->nr_devices ; did++ )
            if ( cudaMemcpyToSymbolAsync( cuda_timers , timers , sizeof(float) * tid_count , 0 , cudaMemcpyHostToDevice , (cudaStream_t)e->streams[did] ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
    #endif
    
    /* Loop over the devices and call the different kernels on each stream. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
        /* Load the particle data onto the device. */
        // tic = getticks();
        // if ( ( maxcount = engine_cuda_load_parts( e ) ) < 0 )
        //     return error(engine_err);
        // e->timers[ engine_timer_cuda_load ] += getticks() - tic;
        counts = e->counts_cuda_local[ did ];
        inds = e->ind_cuda_local[ did ];
        /* Clear the counts array. */
        bzero( counts , sizeof(int) * s->nr_cells );

        /* Load the counts. */
        for( maxcount = 0, k = 0; k < e->cells_cuda_nr[did] ; k++ )
        if( ( counts[e->cells_cuda_local[did][k]] = s->cells[e->cells_cuda_local[did][k]].count ) > maxcount )
            maxcount = counts[ e->cells_cuda_local[did][k]];
    /*    for ( maxcount = 0 , k = 0 ; k < s->nr_marked ; k++ )
            if ( ( counts[ s->cid_marked[k] ] = s->cells[ s->cid_marked[k] ].count ) > maxcount )
                maxcount = counts[ s->cid_marked[k] ];*/

        /* Raise maxcount to the next multiple of 32. */
        maxcount = ( maxcount + (cuda_frame - 1) ) & ~(cuda_frame - 1);
        // printf( "engine_cuda_load_parts: maxcount=%i.\n" , maxcount );

        /* Compute the indices. */
        inds[0] = 0;
        for ( k = 1 ; k < e->cells_cuda_nr[did] ; k++ )
            inds[k] = inds[k-1] + counts[k-1];

        /* Loop over the marked cells. */
        for ( k = 0 ; k < e->cells_cuda_nr[did] ; k++ ) {

            /* Get the cell id. */
            cid = e->cells_cuda_local[did][k];

            /* Copy the particle data to the device. */
            buff_part = (MxParticleCUDA *)&parts_cuda[ inds[cid] ];

            if(nr_states > 0) {
                for ( pid = 0 ; pid < counts[cid] ; pid++ ) {
                    buff_part[ pid ] = MxParticleCUDA(&s->cells[cid].parts[pid], nr_states);
                }
            } 
            else {
                for ( pid = 0 ; pid < counts[cid] ; pid++ ) {
                    buff_part[ pid ] = MxParticleCUDA(&s->cells[cid].parts[pid]);
                }
            }

        }

	/* Start by setting the maxdist on the device. */
        if ( cudaMemcpyToSymbolAsync( cuda_maxdist , &maxdist , sizeof(float) , 0 , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the counts onto the device. */
        if ( cudaMemcpyAsync( e->counts_cuda[did] , counts , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the inds onto the device. */
        if ( cudaMemcpyAsync( e->ind_cuda[did] , inds , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Bind the particle positions. */
        if ( cudaMemcpyAsync( e->parts_cuda[did] , parts_cuda , sizeof(MxParticleCUDA) * s->nr_parts , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

    /* Start the clock. */
    // tic = getticks();
	}
    
    /* Lap the clock on the last stream. */
    if ( cudaEventRecord( toc_load , (cudaStream_t)e->streams[e->nr_devices-1] ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    

        
        

	/* Loop over the devices and call the different kernels on each stream. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {

	    /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
        /* Clear the force array. */
        // if ( cudaMemsetAsync( e->forces_cuda[did] , 0 , sizeof( float ) * 3 * s->nr_parts , stream ) != cudaSuccess )
        //     return cuda_error(engine_err_cuda);
        cuda_memset_float <<<8,512,0,stream>>> ( e->forces_cuda[did] , 0.0f , 4 * s->nr_parts );
        if(nr_states > 0) 
            cuda_memset_float<<<8,512,0,stream>>>( e->fluxes_next_cuda[did] , 0.0f , nr_states * s->nr_parts );

        dim3 nr_threads(e->nr_threads[did], 1, 1);
        dim3 nr_blocks(std::min(e->nrtasks_cuda[did], e->nr_blocks[did]), 1, 1);
        
        /* Start the appropriate kernel. */
        if(nr_states > 0) {
            engine_nonbond_cuda_kernel_launch<true>(maxcount, nr_blocks, nr_threads, stream, 
                e->forces_cuda[did] , e->fluxes_next_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild, nr_states);
        }
        else {
            engine_nonbond_cuda_kernel_launch<false>(maxcount, nr_blocks, nr_threads, stream, 
                e->forces_cuda[did] , e->fluxes_next_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild, nr_states);
        }
}

	for( did = 0; did < e->nr_devices ; did ++ ) {
	
	/* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
        
        /* Get the forces from the device. */
        if ( ( forces_cuda[did] = (float *)malloc( sizeof(float) * 4 * s->nr_parts ) ) == NULL )
            return error(engine_err_malloc);
        if ( cudaMemcpyAsync( forces_cuda[did] , e->forces_cuda[did] , sizeof(float) * 4 * s->nr_parts , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the potential energy. */
        if ( cudaMemcpyFromSymbolAsync( &epot[did] , cuda_epot_out , sizeof(float) , 0 , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        if(nr_states > 0) {
            // Get the flux data
            if ( ( fluxes_next_cuda[did] = (float *)malloc( sizeof(float) * nr_states * s->nr_parts ) ) == NULL )
                return error(engine_err_malloc);
            if ( cudaMemcpyAsync( fluxes_next_cuda[did] , e->fluxes_next_cuda[did] , sizeof(float) * nr_states * s->nr_parts , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        }
        
    }
        
    // e->timers[ engine_timer_cuda_dopairs ] += getticks() - tic;
    
    /* Lap the clock on the last stream. */
    if ( cudaEventRecord( toc_run , (cudaStream_t)e->streams[e->nr_devices-1] ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Get and dump timers. */
    #ifdef TIMERS
        if ( cudaMemcpyFromSymbolAsync( timers , cuda_timers , sizeof(float) * tid_count , 0 , cudaMemcpyDeviceToHost , (cudaStream_t)e->streams[0] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        printf( "engine_nonbond_cuda: timers = [ %.2f " , icpms * timers[0] );
        for ( int k = 1 ; k < tid_count ; k++ )
            printf( "%.2f " , icpms * timers[k] );
        printf( "] ms\n" );
    #endif

    #ifdef TASK_TIMERS
		int4 NAMD_timers_local[26*cuda_maxcells*3];
		if(cudaMemcpyFromSymbol( NAMD_timers_local, NAMD_timers, sizeof(int4)*26*cuda_maxcells*3 , 0 , cudaMemcpyDeviceToHost) != cudaSuccess )
			return cuda_error(engine_err_cuda);	
		for(int i = 0; i < e->s.nr_tasks ; i++)
		printf("Task: %i %i %i %i\n", NAMD_timers_local[i].x, NAMD_timers_local[i].y, NAMD_timers_local[i].z, NAMD_timers_local[i].w);

    #endif
    
    /* Check for any missed CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        

    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
    
        /* Set the device ID. */
        // if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
        //     return cuda_error(engine_err_cuda);

        /* Wait for the chickens to come home to roost. */
        if ( cudaStreamSynchronize( stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    
        /* Get the potential energy. */
        e->s.epot += epot[did];
        
        /* Loop over the marked cells. */
        for ( k = 0 ; k < e->cells_cuda_nr[did] ; k++ ) {

            /* Get the cell id. */
            cid = e->cells_cuda_local[did][k];

            /* Copy the particle data from the device. */
            buff_force = &forces_cuda[did][ 4*e->ind_cuda_local[did][cid] ];
            if(nr_states > 0) {
                buff_flux = &fluxes_next_cuda[did][nr_states * e->ind_cuda_local[did][cid] ];

                for ( pid = 0 ; pid < s->cells[cid].count ; pid++ ) {
                    p = &s->cells[cid].parts[pid];
                    p->f[0] += buff_force[ 4*pid ];
                    p->f[1] += buff_force[ 4*pid + 1 ];
                    p->f[2] += buff_force[ 4*pid + 2 ];
                    p->f[3] += buff_force[ 4*pid + 3 ];

                    for(fid = 0; fid < nr_states; fid++) 
                        p->state_vector->q[fid] += buff_flux[nr_states * pid + fid];
                }
            }
            else {
                for ( pid = 0 ; pid < s->cells[cid].count ; pid++ ) {
                    p = &s->cells[cid].parts[pid];
                    p->f[0] += buff_force[ 4*pid ];
                    p->f[1] += buff_force[ 4*pid + 1 ];
                    p->f[2] += buff_force[ 4*pid + 2 ];
                    p->f[3] += buff_force[ 4*pid + 3 ];
                }
            }

        }

        /* Deallocate the parts array and counts array. */
        free( forces_cuda[did] );
        if(nr_states > 0)
            free(fluxes_next_cuda[did]);
        
    }
        
    /* Check for any missed CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Stop the clock on the last stream. */
    if ( cudaEventRecord( toc_unload , (cudaStream_t)e->streams[e->nr_devices-1] ) != cudaSuccess ||
         cudaStreamSynchronize( (cudaStream_t)e->streams[e->nr_devices-1] ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Check for any missed CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Store the timers. */
    if ( cudaEventElapsedTime( &ms_load , tic , toc_load ) != cudaSuccess ||
         cudaEventElapsedTime( &ms_run , toc_load , toc_run ) != cudaSuccess ||
         cudaEventElapsedTime( &ms_unload , toc_run , toc_unload ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    e->timers[ engine_timer_cuda_load ] += ms_load / 1000 * CPU_TPS;
    e->timers[ engine_timer_cuda_dopairs ] += ms_run / 1000 * CPU_TPS;
    e->timers[ engine_timer_cuda_unload ] += ms_unload / 1000 * CPU_TPS;
    
    /* Go away. */
    return engine_err_ok;
    
}



/**
 * @brief Load the cell data onto the CUDA device.
 *
 * @param e The #engine.
 *
 * @return The maximum number of parts per cell or < 0
 *      on error (see #engine_err).
 */
 
extern "C" int engine_cuda_load_parts ( struct engine *e ) {
    
    int k, did, cid, pid, maxcount = 0;
    MxParticleCUDA *parts_cuda = (MxParticleCUDA *)e->parts_cuda_local, *buff;
    struct space *s = &e->s;
    FPTYPE maxdist = s->cutoff + 2*s->maxdx;
    int *counts = e->counts_cuda_local[0], *inds = e->ind_cuda_local[0];
    cudaStream_t stream;
    
    /* Clear the counts array. */
    bzero( counts , sizeof(int) * s->nr_cells );

    /* Load the counts. */
    for ( maxcount = 0 , k = 0 ; k < s->nr_marked ; k++ )
        if ( ( counts[ s->cid_marked[k] ] = s->cells[ s->cid_marked[k] ].count ) > maxcount )
            maxcount = counts[ s->cid_marked[k] ];

    /* Raise maxcount to the next multiple of 32. */
    maxcount = ( maxcount + (cuda_frame - 1) ) & ~(cuda_frame - 1);
    // printf( "engine_cuda_load_parts: maxcount=%i.\n" , maxcount );

    /* Compute the indices. */
    inds[0] = 0;
    for ( k = 1 ; k < s->nr_cells ; k++ )
        inds[k] = inds[k-1] + counts[k-1];

    /* Loop over the marked cells. */
    for ( k = 0 ; k < s->nr_marked ; k++ ) {

        /* Get the cell id. */
        cid = s->cid_marked[k];

        /* Copy the particle data to the device. */
        buff = (MxParticleCUDA *)&parts_cuda[ inds[cid] ];
        for ( pid = 0 ; pid < counts[cid] ; pid++ ) {
            buff[ pid ] = MxParticleCUDA(&s->cells[cid].parts[pid]);
        }

    }

    // printf( "engine_cuda_load_parts: packed %i cells with %i parts each (%i kB).\n" , s->nr_cells , maxcount , (sizeof(float4)*maxcount*s->nr_cells)/1024 );

    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
        
        /* Start by setting the maxdist on the device. */
        if ( cudaMemcpyToSymbolAsync( cuda_maxdist , &maxdist , sizeof(float) , 0 , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the counts onto the device. */
        if ( cudaMemcpyAsync( e->counts_cuda[did] , counts , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the inds onto the device. */
        if ( cudaMemcpyAsync( e->ind_cuda[did] , inds , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Bind the particle positions. */
        if ( cudaMemcpyAsync( e->parts_cuda[did] , parts_cuda , sizeof(MxParticleCUDA) * s->nr_parts , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Clear the force array. */
        if ( cudaMemsetAsync( e->forces_cuda[did] , 0 , sizeof( float ) * 4 * s->nr_parts , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
    }
    
    /* Our work is done here. */
    return maxcount;

}
    
    

/**
 * @brief Load the cell data from the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_unload_parts ( struct engine *e ) {

    int k, did, cid, pid;
    struct MxParticle *p;
    float *forces_cuda[ engine_maxgpu ], *buff, epot[ engine_maxgpu ];
    struct space *s = &e->s;
    cudaStream_t stream;
    
    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
    
        /* Get the forces from the device. */
        if ( ( forces_cuda[did] = (float *)malloc( sizeof(float) * 4 * s->nr_parts ) ) == NULL )
            return error(engine_err_malloc);
        if ( cudaMemcpyAsync( forces_cuda[did] , e->forces_cuda[did] , sizeof(float) * 4 * s->nr_parts , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the potential energy. */
        if ( cudaMemcpyFromSymbolAsync( &epot[did] , cuda_epot_out , sizeof(float) , 0 , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        
    }

    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
    
        /* Set the device ID. */
        // if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
        //     return cuda_error(engine_err_cuda);

        /* Wait for the chickens to come home to roost. */
        if ( cudaStreamSynchronize( stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    
        /* Get the potential energy. */
        e->s.epot += epot[did];
        
        /* Loop over the marked cells. */
        for ( k = 0 ; k < s->nr_marked ; k++ ) {

            /* Get the cell id. */
            cid = s->cid_marked[k];

            /* Copy the particle data from the device. */
            buff = &forces_cuda[did][ 4*e->ind_cuda_local[did][cid] ];
            for ( pid = 0 ; pid < s->cells[cid].count ; pid++ ) {
                p = &s->cells[cid].parts[pid];
                p->f[0] += buff[ 4*pid ];
                p->f[1] += buff[ 4*pid + 1 ];
                p->f[2] += buff[ 4*pid + 2 ];
                p->f[3] += buff[ 4*pid + 3 ];
                }

            }

        /* Deallocate the parts array and counts array. */
        free( forces_cuda[did] );
        
    }
        
    /* Our work is done here. */
    return engine_err_ok;

}

/**
 * @brief Load the queues onto the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_cuda_queues_load ( struct engine *e ) {
    
    int did, nr_queues, qid, k, qsize, nr_tasks = e->s.nr_tasks;
    struct cudaDeviceProp prop;
    int *data;
    struct queue_cuda queues[ cuda_maxqueues ];
    
    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Get the device properties. */
        if ( cudaGetDeviceProperties( &prop , e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Get the number of SMs on the current device. */
        nr_queues = 1; // prop.multiProcessorCount;

        /* Get the local number of tasks. */
        nr_tasks = e->nrtasks_cuda[did];

        /* Set the size of each queue. */
        qsize = 3 * nr_tasks / min( nr_queues , e->nr_runners );
        if ( cudaMemcpyToSymbol( cuda_queue_size , &qsize , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Allocate a temporary buffer for the queue data. */
        if ( ( data = (int *)malloc( sizeof(int) * qsize ) ) == NULL )
            return error(engine_err_malloc);

        /* Set the number of queues. */
        e->nr_queues_cuda = nr_queues;

        /* Init each queue separately. */
        for ( qid = 0 ; qid < nr_queues ; qid++ ) {

            /* Fill the data for this queue. */
            queues[qid].count = 0;
            for ( k = qid ; k < nr_tasks ; k += nr_queues )
                data[ queues[qid].count++ ] = k;
            for ( k = queues[qid].count ; k < qsize ; k++ )
                data[k] = -1;

            /* Allocate and copy the data. */
            if ( cudaMalloc( &queues[qid].data , sizeof(int) * qsize ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
            if ( cudaMemcpy( (void *)queues[qid].data , data , sizeof(int) * qsize , cudaMemcpyHostToDevice ) != cudaSuccess )
                return cuda_error(engine_err_cuda);

            /* Allocate and copy the recycling data. */
            for ( k = 0 ; k < queues[qid].count ; k++ )
                data[k] = -1;
            if ( cudaMalloc( &queues[qid].rec_data , sizeof(int) * qsize ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
            if ( cudaMemcpy( (void *)queues[qid].rec_data , data , sizeof(int) * qsize , cudaMemcpyHostToDevice ) != cudaSuccess )
                return cuda_error(engine_err_cuda);

            /* Set some other values. */
            queues[qid].first = 0;
            queues[qid].last = queues[qid].count;
            queues[qid].rec_count = 0;

        }

        /* Copy the queue structures to the device. */
        if ( cudaMemcpyToSymbol( cuda_queues , queues , sizeof(struct queue_cuda) * nr_queues , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Wait so that we can re-use the local memory. */            
        if ( cudaDeviceSynchronize() != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Clean up. */
        free( data );
        
    }
        
    /* Fade to grey. */
    return engine_err_ok;

}


__global__ void engine_cuda_queues_finalize_device(int nr_queues) {
    if(threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for(int qid = 0; qid < nr_queues; qid++) {

        if(cudaFree(&cuda_queues[qid].data) != cudaSuccess) {
            printf("%s\n", "engine_cuda_queues_finalize_device failed (data)!");
            return;
        }

        if(cudaFree(&cuda_queues[qid].rec_data) != cudaSuccess) {
            printf("%s\n", "engine_cuda_queues_finalize_device failed (rec_data)!");
            return;
        }

    }

    if(cudaFree(cuda_queues) != cudaSuccess) {
        printf("%s\n", "engine_cuda_queues_finalize_device failed (cuda_queues)!");
        return;
    }
}

/**
 * @brief Close the run configuration on the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_cuda_queues_finalize ( struct engine *e ) {

    /* Loop over the devices. */
    for(int did = 0 ; did < e->nr_devices ; did++ ) {
        
        // Set the device ID
        
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        // Free queues
        
        engine_cuda_queues_finalize_device<<<1, 1>>>(e->nr_queues_cuda);

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}

/**
 * @brief Load the potentials onto the CUDA device
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_cuda_load_pots(struct engine *e) {
    int i, j, nr_pots, nr_pots_cluster;
    int did;
    int nr_devices = e->nr_devices;
    int *pind = (int*)malloc(sizeof(int) * e->max_type * e->max_type);
    int *pind_cluster = (int*)malloc(sizeof(int) * e->max_type * e->max_type);
    struct MxPotential **pots = (MxPotential**)malloc(sizeof(MxPotential*) * e->nr_types * (e->nr_types + 1) / 2 + 1);
    struct MxPotential **pots_cluster = (MxPotential**)malloc(sizeof(MxPotential*) * e->nr_types * (e->nr_types + 1) / 2 + 1);

    /* Init the null potential. */
    if ( ( pots[0] = (struct MxPotential *)alloca( sizeof(struct MxPotential) ) ) == NULL )
        return error(engine_err_malloc);
    pots[0]->alpha[0] = pots[0]->alpha[1] = pots[0]->alpha[2] = pots[0]->alpha[3] = 0.0f;
    pots[0]->a = 0.0; pots[0]->b = FLT_MAX;
    pots[0]->flags = POTENTIAL_NONE;
    pots[0]->n = 0;
    if ( ( pots[0]->c = (FPTYPE *)alloca( sizeof(float) * potential_chunk ) ) == NULL )
        return error(engine_err_malloc);
    bzero( pots[0]->c , sizeof(float) * potential_chunk );

    if ( ( pots_cluster[0] = (struct MxPotential *)alloca( sizeof(struct MxPotential) ) ) == NULL )
        return error(engine_err_malloc);
    pots_cluster[0]->alpha[0] = pots_cluster[0]->alpha[1] = pots_cluster[0]->alpha[2] = pots_cluster[0]->alpha[3] = 0.0f;
    pots_cluster[0]->a = 0.0; pots_cluster[0]->b = FLT_MAX;
    pots_cluster[0]->flags = POTENTIAL_NONE;
    pots_cluster[0]->n = 0;
    if ( ( pots_cluster[0]->c = (FPTYPE *)alloca( sizeof(float) * potential_chunk ) ) == NULL )
        return error(engine_err_malloc);
    bzero( pots_cluster[0]->c , sizeof(float) * potential_chunk );
    
    /* Start by identifying the unique potentials in the engine. */
    nr_pots = 1;
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
    
        /* Skip if there is no potential or no parts of this type. */
        if ( e->p[i] == NULL )
            continue;
            
        /* Check this potential against previous potentials. */
        for ( j = 0 ; j < nr_pots && e->p[i] != pots[j] ; j++ );
        if ( j < nr_pots )
            continue;
            
        /* Store this potential and the number of coefficient entries it has. */
        pots[nr_pots] = e->p[i];
        nr_pots += 1;
    
    }

    nr_pots_cluster = 1;
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
    
        /* Skip if there is no potential or no parts of this type. */
        if ( e->p_cluster[i] == NULL )
            continue;
            
        /* Check this potential against previous potentials. */
        for ( j = 0 ; j < nr_pots_cluster && e->p_cluster[i] != pots_cluster[j] ; j++ );
        if ( j < nr_pots_cluster )
            continue;
            
        /* Store this potential and the number of coefficient entries it has. */
        pots_cluster[nr_pots_cluster] = e->p_cluster[i];
        nr_pots_cluster += 1;
    
    }

    /* Pack the potential matrices. */
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
        if ( e->p[i] == NULL ) {
            pind[i] = 0;
        }
        else {
            for ( j = 0 ; j < nr_pots && pots[j] != e->p[i] ; j++ );
            pind[i] = j;
        }
    }
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
        if ( e->p_cluster[i] == NULL ) {
            pind_cluster[i] = 0;
        }
        else {
            for ( j = 0 ; j < nr_pots_cluster && pots_cluster[j] != e->p_cluster[i] ; j++ );
            pind_cluster[i] = j;
        }
    }

    // Pack the potentials
    MxPotentialCUDA *pots_cuda = (MxPotentialCUDA*)malloc(sizeof(MxPotentialCUDA) * nr_pots);
    for(i = 1; i < nr_pots; i++) {
        pots_cuda[i] = MxPotentialCUDA(pots[i]);
    }

    MxPotentialCUDA *pots_cluster_cuda = (MxPotentialCUDA*)malloc(sizeof(MxPotentialCUDA) * nr_pots_cluster);
    for(i = 1; i < nr_pots_cluster; i++) {
        pots_cluster_cuda[i] = MxPotentialCUDA(pots_cluster[i]);
    }
    
    /* Store pind as a constant. */
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->pind_cuda[did] , sizeof(unsigned int) * e->max_type * e->max_type ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( e->pind_cuda[did] , pind , sizeof(unsigned int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_pind , &e->pind_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }
    free(pind);
    
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->pind_cluster_cuda[did] , sizeof(unsigned int) * e->max_type * e->max_type ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( e->pind_cluster_cuda[did] , pind_cluster , sizeof(unsigned int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_pind_cluster , &e->pind_cluster_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }
    free(pind_cluster);

    // Store the potentials
    for(did = 0; did < nr_devices; did++) {
        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMalloc(&e->pots_cuda[did], sizeof(MxPotentialCUDA) * nr_pots) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpy(e->pots_cuda[did], pots_cuda, sizeof(MxPotentialCUDA) * nr_pots, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpyToSymbol(cuda_pots, &e->pots_cuda[did], sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }
    free(pots);
    free(pots_cuda);
    e->nr_pots_cuda = nr_pots;
    
    for(did = 0; did < nr_devices; did++) {
        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMalloc(&e->pots_cluster_cuda[did], sizeof(MxPotentialCUDA) * nr_pots_cluster) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpy(e->pots_cluster_cuda[did], pots_cluster_cuda, sizeof(MxPotentialCUDA) * nr_pots_cluster, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpyToSymbol(cuda_pots_cluster, &e->pots_cluster_cuda[did], sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }
    free(pots_cluster);
    free(pots_cluster_cuda);
    e->nr_pots_cluster_cuda = nr_pots_cluster;

    return engine_err_ok;
}

/**
 * @brief Unload the potentials on the CUDA device
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_cuda_unload_pots(struct engine *e) {

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        // Free the potentials.
        
        if(cudaFree(e->pind_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        
        if(cudaFree(e->pind_cluster_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        
        if(cudaFree((MxPotentialCUDA*)e->pots_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        
        if(cudaFree((MxPotentialCUDA*)e->pots_cluster_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    e->nr_pots_cuda = 0;
    e->nr_pots_cluster_cuda = 0;

    return engine_err_ok;
}

/**
 * @brief Refresh the potentials on the CUDA device. 
 * 
 * Can be safely called while on the CUDA device to reload all potential data from the engine. 
 * 
 * @param e The #engine
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err)
 */
extern "C" int engine_cuda_refresh_pots(struct engine *e) {
    
    if(engine_cuda_unload_pots(e) < 0)
        return error(engine_err);

    if(engine_cuda_load_pots(e) < 0)
        return error(engine_err);

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}


/**
 * @brief Allocates particle buffers. Must be called before running on a CUDA device. 
 * 
 * @param e The #engine
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_cuda_allocate_particles(struct engine *e) {

    /* Allocate the particle buffer. */
    if((e->parts_cuda_local = (MxParticleCUDA*)malloc(sizeof(MxParticleCUDA) * e->s.size_parts)) == NULL)
        return error(engine_err_malloc);

    /* Allocate the particle and force data. */
    for(int did = 0; did < e->nr_devices; did++) {
        if (cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if (cudaMalloc(&e->parts_cuda[did], sizeof(MxParticleCUDA) * e->s.size_parts) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if (cudaMemcpyToSymbol(cuda_parts, &e->parts_cuda[did], sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if (cudaMalloc(&e->forces_cuda[did], sizeof(float) * 4 * e->s.size_parts) != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }

    return engine_err_ok;
}


/**
 * @brief Closes particle buffers. 
 * 
 * @param e The #engine
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_cuda_finalize_particles(struct engine *e) {

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        // Free the particle and force data

        if(cudaFree(e->parts_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaFree(e->forces_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    // Free the particle buffer

    free(e->parts_cuda_local);

    return engine_err_ok;
}


/**
 * @brief Refreshes particle buffers. Can be safely used to resize buffers while running on CUDA device. 
 * 
 * @param e The #engine
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_cuda_refresh_particles(struct engine *e) {
    
    if(engine_cuda_finalize_particles(e) < 0)
        return cuda_error(engine_err_cuda);

    if(engine_cuda_allocate_particles(e) < 0)
        return cuda_error(engine_err_cuda);

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}


/**
 * @brief Load the potentials and cell pairs onto the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_load ( struct engine *e ) {

    int i, k, nr_tasks, c1 ,c2;
    int did, *cellsorts;
    struct space *s = &e->s;
    int nr_devices = e->nr_devices;
    struct task_cuda *tasks_cuda, *tc, *ts;
    struct task *t;
    float dt = e->dt, cutoff = e->s.cutoff, cutoff2 = e->s.cutoff2, dscale; //, buff[ e->nr_types ];
    float h[3], dim[3], *corig;
    void *dummy[ engine_maxgpu ];
    unsigned int *cflags;
    float3 *cdims;

    /*Split the space over the available GPUs*/
    engine_split_gpu( e , nr_devices , engine_split_GPU  );
    
    /* Copy the cell edge lengths to the device. */
    h[0] = s->h[0]*s->span[0];
    h[1] = s->h[1]*s->span[1];
    h[2] = s->h[2]*s->span[2];
    dim[0] = s->dim[0]; dim[1] = s->dim[1]; dim[2] = s->dim[2];
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_h , h , sizeof(float) * 3 , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_dim , dim , sizeof(float) * 3 , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }
        
    /* Copy the cell origins to the device. */
    if ( ( corig = (float *)malloc( sizeof(float) * s->nr_cells * 3 ) ) == NULL )
        return error(engine_err_malloc);
    for ( i = 0 ; i < s->nr_cells ; i++ ) {
        corig[ 3*i + 0 ] = s->cells[i].origin[0];
        corig[ 3*i + 1 ] = s->cells[i].origin[1];
        corig[ 3*i + 2 ] = s->cells[i].origin[2];
    }
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(float) * s->nr_cells * 3 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , corig , sizeof(float) * s->nr_cells * 3 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_corig , &dummy[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }
    free( corig );

    // Copy the cell dimensions to devices
    if((cdims = (float3*)malloc(sizeof(float3) * s->nr_cells)) == NULL)
        return error(engine_err_malloc);
    for(i = 0; i < s->nr_cells; i++) {
        cdims[i] = make_float3(s->cells[i].dim[0], s->cells[i].dim[1], s->cells[i].dim[2]);
    }
    for(did = 0; did < nr_devices; did++) {
        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMalloc(&dummy[did], sizeof(float3) * s->nr_cells) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpy(dummy[did], cdims, sizeof(float3) * s->nr_cells, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpyToSymbol(cuda_cdims, &dummy[did], sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }
    free(cdims);

    // Copy the cell flags to devices
    if((cflags = (unsigned int*)malloc(sizeof(unsigned int) * s->nr_cells)) == NULL)
        return error(engine_err_malloc);
    for(i = 0; i < s->nr_cells; i++)
        cflags[i] = s->cells[i].flags;
    for(did = 0; did < nr_devices; did++) {
        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMalloc(&dummy[did], sizeof(unsigned int) * s->nr_cells) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpy(dummy[did], cflags, sizeof(unsigned int) * s->nr_cells, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpyToSymbol(cuda_cflags, &dummy[did], sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }
    free(cflags);
        
    /* Set the constant pointer to the null potential and other useful values. */
    dscale = ((float)SHRT_MAX) / ( 3.0 * sqrt( s->h[0]*s->h[0]*s->span[0]*s->span[0] + s->h[1]*s->h[1]*s->span[1]*s->span[1] + s->h[2]*s->h[2]*s->span[2]*s->span[2] ) );
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_dt , &dt , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_cutoff , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_cutoff2 , &cutoff2 , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_maxdist , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_maxtype , &(e->max_type) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_dscale , &dscale , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }
        
    /* Allocate and fill the task list. */
    if ( ( tasks_cuda = (struct task_cuda *)malloc( sizeof(struct task_cuda) * s->nr_tasks ) ) == NULL )
        return error(engine_err_malloc);
    if ( ( cellsorts = (int *)malloc( sizeof(int) * s->nr_tasks ) ) == NULL )
        return error(engine_err_malloc);
    for ( did = 0 ;did < nr_devices ; did++ ) {
	if( (e->cells_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL)
	    return error(engine_err_malloc);
        e->cells_cuda_nr[did]=0;
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        /* Select the tasks for each device ID. */  
        for ( nr_tasks = 0 , i = 0 ; i < s->nr_tasks ; i++ ) {
            
            /* Get local pointers. */
            t = &s->tasks[i];
            tc = &tasks_cuda[nr_tasks];
	    
            /* Skip pairs and self with wrong cid, keep all sorts. */
            if ( ( t->type == task_type_pair && e->s.cells[t->i].GPUID != did  /*t->i % nr_devices != did */) ||
                 ( t->type == task_type_self && e->s.cells[t->i].GPUID != did /*e->s.cells[t->i].loc[1] < e->s.cdim[1] / e->nr_devices * (did + 1) && e->s.cells[t->i].loc[1] >= e->s.cdim[1] / e->nr_devices * did t->i % e->nr_devices != did*/ ) )
                continue;
            
            /* Copy the data. */
            tc->type = t->type;
            tc->subtype = t->subtype;
            tc->wait = 0;
            tc->flags = t->flags;
            tc->i = t->i;
            tc->j = t->j;
            tc->nr_unlock = 0;
            
            /* Remember which task sorts which cell. */
            if ( t->type == task_type_sort ) {
                tc->flags = 0;
                cellsorts[ t->i ] = nr_tasks;
            }

            /*Add the cell to list of cells for this GPU if needed*/
            c1=1; c2=1;
            for(int i = 0; i < e->cells_cuda_nr[did] ; i++ ) {
                /* Check cell is valid */
                if(t->i < 0 || t->i == e->cells_cuda_local[did][i])
                    c1 = 0;
                if(t->j < 0 || t->j == e->cells_cuda_local[did][i])
                    c2 = 0;
            }
            if( c1 )
                e->cells_cuda_local[did][e->cells_cuda_nr[did]++] = t->i;
            if( c2 )
                e->cells_cuda_local[did][e->cells_cuda_nr[did]++] = t->j;	                
            /* Add one task. */
            nr_tasks += 1;
		
        }

        /* Link each pair task to its sorts. */
        for ( i = 0 ; i < nr_tasks ; i++ ) {
            tc = &tasks_cuda[i];
	
            if ( tc->type == task_type_pair ) {
                ts = &tasks_cuda[ cellsorts[ tc->i ] ];
                ts->flags |= (1 << tc->flags);
                ts->unlock[ ts->nr_unlock ] = i;
                ts->nr_unlock += 1;
                ts = &tasks_cuda[ cellsorts[ tc->j ] ];
                ts->flags |= (1 << tc->flags);
                ts->unlock[ ts->nr_unlock ] = i;
                ts->nr_unlock += 1;
            }
        }
        
        /* Set the waits. */
        for ( i = 0 ; i < nr_tasks ; i++ )
            for ( k = 0 ; k < tasks_cuda[i].nr_unlock ; k++ )
                tasks_cuda[ tasks_cuda[i].unlock[k] ].wait += 1;

        /* Allocate and fill the tasks list on the device. */
        if ( cudaMemcpyToSymbol( cuda_nr_tasks , &nr_tasks , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(struct task_cuda) * s->nr_tasks ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , tasks_cuda , sizeof(struct task_cuda) * s->nr_tasks , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_tasks , &dummy[did] , sizeof(struct task_cuda *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Remember the number of tasks. */
        e->nrtasks_cuda[did] = nr_tasks;
            
    }
    
	/* Clean up */
    free( tasks_cuda );
    free( cellsorts );
        
    /* Allocate the sortlists locally and on the device if needed. */
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->sortlists_cuda[did] , sizeof(unsigned int) * s->nr_parts * 13 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_sortlists , &e->sortlists_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }


    for ( did = 0 ;did < nr_devices ; did++ ) {
	    /* Allocate the cell counts and offsets. */
        if ( ( e->counts_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL ||
            ( e->ind_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL )
            return error(engine_err_malloc);
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->counts_cuda[did] , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->ind_cuda[did] , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }
        
    /* Allocate and init the taboo list on the device. */
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemset( dummy[did] , 0 , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_taboo , &dummy[did] , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }

    // Allocate random number generators
    if(engine_cuda_rand_norm_init(e) < 0)
        return error(engine_err);

    // Allocate boundary conditions
    if(engine_cuda_boundary_conditions_load(e) < 0)
        return error(engine_err);
        
    if(engine_cuda_allocate_particles(e) < 0)
        return error(engine_err);

    if(engine_cuda_load_pots(e) < 0)
        return error(engine_err);

    if(engine_cuda_load_fluxes(e) < 0)
        return error(engine_err);
    
    /* Init the pair queue on the device. */
    if ( engine_cuda_queues_load( e ) < 0 )
        return error(engine_err);
        
    /* He's done it! */
    return engine_err_ok;
    
}


/**
 * @brief Removes the potentials and cell pairs on the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_parts_finalize(struct engine *e) {

    if(engine_cuda_rand_norm_finalize(e) < 0)
        return error(engine_err);

    if(engine_cuda_boundary_conditions_finalize(e) < 0)
        return error(engine_err);

    if(engine_cuda_unload_pots(e) < 0)
        return error(engine_err);

    if(engine_cuda_finalize_particles(e) < 0)
        return error(engine_err);

    if(engine_cuda_unload_fluxes(e) < 0)
        return error(engine_err);

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        e->nrtasks_cuda[did] = 0;

        // Free the sort list, counts and indices

        if(cudaFree(e->sortlists_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaFree(e->counts_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaFree(e->ind_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}

/**
 * @brief Unload the potentials and cell pairs on the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_finalize ( struct engine *e ) {
    if(engine_parts_finalize(e) < 0)
        return error(engine_err);

    if(engine_cuda_queues_finalize(e) < 0)
        return error(engine_err);

    return engine_err_ok;
}

/**
 * @brief Refresh the engine image on the CUDA device. 
 * 
 * Can be safely called while on the CUDA device to reload all data from the engine. 
 * 
 * @param e The #engine
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err)
 */
extern "C" int engine_cuda_refresh(struct engine *e) {
    
    if(engine_cuda_finalize(e) < 0)
        return error(engine_err);

    if(engine_cuda_load(e) < 0)
        return error(engine_err);

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}
