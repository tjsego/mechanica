/**
 * @file bond_cuda.cu
 * @author T.J. Sego, Ph.D.
 * @brief Defines bond kernels on CUDA-supporting GPUs
 * @date 2021-11-29
 * 
 */

#include "bond_cuda.h"

#include <engine.h>
#include <errs.h>
#include <runner_cuda.h>
#include "../../mx_cuda.h"

#if MX_THREADING
#include "MxTaskScheduler.hpp"
#endif


__device__ MxBondCUDA *cuda_bonds;
static MxBondCUDA *cuda_bonds_device_arr;
static int cuda_bonds_size = 0;

__constant__ float3 cuda_bonds_cell_edge_lens;
__constant__ float cuda_bonds_dt;

// Random number generators
__device__ curandState *cuda_rand_unif;
void *rand_unif_cuda;

static bool cuda_rand_unif_init = false;
static int cuda_rand_unif_seed;
static unsigned int cuda_bonds_nr_threads = 0;
static unsigned int cuda_bonds_nr_blocks = 0;
static int cuda_bonds_device = 0;
static cudaStream_t *cuda_bonds_stream, cuda_bonds_stream1, cuda_bonds_stream2;
static unsigned int cuda_bonds_front_stream = 1;

static struct MxBondCUDAData *cuda_bonds_bonds_arr, *cuda_bonds_bonds_arr1, *cuda_bonds_bonds_arr2;
static float *cuda_bonds_forces_arr, *cuda_bonds_forces_arr1, *cuda_bonds_forces_arr2;
static float *cuda_bonds_potenergies_arr, *cuda_bonds_potenergies_arr1, *cuda_bonds_potenergies_arr2;
static bool *cuda_bonds_todestroy_arr, *cuda_bonds_todestroy_arr1, *cuda_bonds_todestroy_arr2;

static struct MxBondCUDAData *cuda_bonds_bonds_local, *cuda_bonds_bonds_local1, *cuda_bonds_bonds_local2;
static float *cuda_bonds_forces_local, *cuda_bonds_forces_local1, *cuda_bonds_forces_local2;
static float *cuda_bonds_potenergies_local, *cuda_bonds_potenergies_local1, *cuda_bonds_potenergies_local2;
static bool *cuda_bonds_todestroy_local, *cuda_bonds_todestroy_local1, *cuda_bonds_todestroy_local2;

#define cuda_bonds_nrparts_chunk    102400
#define cuda_bonds_nrparts_incr     100

#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )

__host__ 
MxBondCUDA::MxBondCUDA() {
    this->flags = ~BOND_ACTIVE;
}

__host__ 
MxBondCUDA::MxBondCUDA(MxBond *b) : 
    flags{b->flags}, 
    dissociation_energy{(float)b->dissociation_energy}, 
    half_life{(float)b->half_life}
{
    this->p = MxToCUDADevice(b->potential);
}

__host__ 
void MxBondCUDA::finalize() {
    if(!(this->flags & BOND_ACTIVE)) 
        return;

    this->flags = ~BOND_ACTIVE;
    Mx_cudaFree(&this->p);
}


struct MxBondCUDAData {
    uint16_t pi_flags, pj_flags;
    uint32_t id;
    int3 cix, cjx;
    float3 pix, pjx;

    MxBondCUDAData(MxBond *b) : 
        id{b->id}
    {
        MxParticle *part_i = _Engine.s.partlist[b->i];
        MxParticle *part_j = _Engine.s.partlist[b->j];

        this->pi_flags = part_i->flags;
        this->pj_flags = part_j->flags;
        this->pix = make_float3(part_i->x[0], part_i->x[1], part_i->x[2]);
        this->pjx = make_float3(part_j->x[0], part_j->x[1], part_j->x[2]);

        int *loc_i, *loc_j;
        loc_i = &_Engine.s.celllist[part_i->id]->loc[0];
        loc_j = &_Engine.s.celllist[part_j->id]->loc[0];
        this->cix = make_int3(loc_i[0], loc_i[1], loc_i[2]);
        this->cjx = make_int3(loc_j[0], loc_j[1], loc_j[2]);
    }
};


int cuda_bonds_bonds_initialize(engine *e, MxBond *bonds, int N) {
    size_t size_bonds = N * sizeof(MxBondCUDA);

    if(cudaMalloc(&cuda_bonds_device_arr, size_bonds) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    MxBondCUDA *bonds_cuda = (MxBondCUDA*)malloc(size_bonds);

    int nr_runners = e->nr_runners;
    auto func = [&bonds, &bonds_cuda, N, nr_runners](int tid) {
        for(int i = tid; i < N; i += nr_runners) 
            bonds_cuda[i] = MxBondCUDA(&bonds[i]);
    };
    #if MX_THREADING
    mx::parallel_for(nr_runners, func);
    #else
    func(0);
    #endif

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaMemcpyToSymbol(cuda_bonds, &cuda_bonds_device_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    free(bonds_cuda);

    cuda_bonds_size = N;

    return engine_err_ok;
}

int cuda_bonds_bonds_finalize(engine *e) { 
    size_t size_bonds = cuda_bonds_size * sizeof(MxBondCUDA);

    MxBondCUDA *bonds_cuda = (MxBondCUDA*)malloc(size_bonds);
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    int nr_runners = e->nr_runners;
    int N = cuda_bonds_size;
    auto func = [&bonds_cuda, N, nr_runners](int tid) {
        for(int i = tid; i < N; i += nr_runners) 
            bonds_cuda[i].finalize();
    };
    #if MX_THREADING
    mx::parallel_for(nr_runners, func);
    #else
    func(0);
    #endif

    free(bonds_cuda);

    if(cudaFree(cuda_bonds_device_arr) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    
    cuda_bonds_size = 0;

    return engine_err_ok;
}

template <typename T> __global__ 
void engine_cuda_memcpy(T *dst, T *src, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    int nr_elems = size / sizeof(T);
    for(int i = tid; i < nr_elems; i += stride) 
        dst[i] = src[i];
}

int cuda_bonds_bonds_extend() {
    int Ni = cuda_bonds_size;
    int Nf = Ni + cuda_bonds_nrparts_incr;
    
    dim3 nr_threads(cuda_bonds_nr_threads, 1, 1);
    dim3 nr_blocks(std::max((unsigned)1, Ni / nr_threads.x), 1, 1);

    MxBondCUDA *cuda_bonds_device_arr_new;
    if(cudaMalloc(&cuda_bonds_device_arr_new, Nf * sizeof(MxBondCUDA)) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    
    engine_cuda_memcpy<MxBondCUDA><<<nr_blocks, nr_threads>>>(cuda_bonds_device_arr_new, cuda_bonds_device_arr, Ni * sizeof(MxBondCUDA));
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaFree(cuda_bonds_device_arr) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaMalloc(&cuda_bonds_device_arr, Nf * sizeof(MxBondCUDA)) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    engine_cuda_memcpy<MxBondCUDA><<<nr_blocks, nr_threads>>>(cuda_bonds_device_arr, cuda_bonds_device_arr_new, Ni * sizeof(MxBondCUDA));
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaFree(cuda_bonds_device_arr_new) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaMemcpyToSymbol(cuda_bonds, &cuda_bonds_device_arr, sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    cuda_bonds_size = Nf;

    return engine_err_ok;
}

__global__ 
void engine_cuda_set_bond_device(MxBondCUDA b, unsigned int bid) {
    if(threadIdx.x > 0 || blockIdx.x > 0) 
        return;

    cuda_bonds[bid] = b;
}

int engine_cuda_add_bond(MxBond *b) {
    MxBondCUDA bc(b);
    auto bid = b->id;

    if(bid >= cuda_bonds_size) 
        if(cuda_bonds_bonds_extend() < 0) 
            return error(engine_err);

    engine_cuda_set_bond_device<<<1, 1>>>(bc, bid);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    return engine_err_ok;
}

__global__ 
void engine_cuda_set_bonds_device(MxBondCUDA *bonds, unsigned int *bids, int nr_bonds) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = tid; i < nr_bonds; i += stride) 
        cuda_bonds[i] = bonds[bids[i]];
}

int engine_cuda_add_bonds(MxBond *bonds, int nr_bonds) {
    MxBond *b;
    uint32_t bidmax = 0;

    MxBondCUDA *bcs = (MxBondCUDA*)malloc(nr_bonds * sizeof(MxBondCUDA));
    MxBondCUDA *bcs_d;
    if(cudaMalloc(&bcs_d, nr_bonds * sizeof(MxBondCUDA)) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    unsigned int *bids = (unsigned int*)malloc(nr_bonds * sizeof(unsigned int));
    unsigned int *bids_d;
    if(cudaMalloc(&bids_d, nr_bonds * sizeof(unsigned int)) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    for(int i = 0; i < nr_bonds; i++) {
        b = &bonds[i];
        bidmax = std::max(bidmax, b->id);
        bcs[i] = MxBondCUDA(b);
        bids[i] = b->id;
    }

    if(cudaMemcpyAsync(bcs_d , bcs , nr_bonds * sizeof(MxBondCUDA)  , cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaMemcpyAsync(bids_d, bids, nr_bonds * sizeof(unsigned int), cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    while(bidmax >= cuda_bonds_size)
        if(cuda_bonds_bonds_extend() < 0) 
            return error(engine_err);

    dim3 nr_threads(cuda_bonds_nr_threads, 1, 1);
    dim3 nr_blocks(std::max((unsigned)1, nr_bonds / nr_threads.x), 1, 1);

    if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    engine_cuda_set_bonds_device<<<nr_blocks, nr_threads>>>(bcs_d, bids_d, nr_bonds);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    free(bcs);
    free(bids);
    if(cudaFree(bcs_d) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaFree(bids_d) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    return engine_err_ok;
}

int engine_cuda_finalize_bond(int bind) { 
    size_t size_bonds = cuda_bonds_size * sizeof(MxBondCUDA);

    MxBondCUDA *bonds_cuda = (MxBondCUDA*)malloc(cuda_bonds_size * sizeof(MxBondCUDA));
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    bonds_cuda[bind].finalize();

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    free(bonds_cuda);

    return engine_err_ok;
}

int engine_cuda_finalize_bonds(engine *e, int *binds, int nr_bonds) { 
    size_t size_bonds = cuda_bonds_size * sizeof(MxBondCUDA);

    MxBondCUDA *bonds_cuda = (MxBondCUDA*)malloc(cuda_bonds_size * sizeof(MxBondCUDA));
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    int nr_runners = e->nr_runners;
    auto func = [&bonds_cuda, &binds, nr_bonds, nr_runners](int tid) {
        for(int i = tid; i < nr_bonds; i += nr_runners) 
            bonds_cuda[binds[i]].finalize();
    };
    #if MX_THREADING
    mx::parallel_for(nr_runners, func);
    #else
    func(0);
    #endif

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    free(bonds_cuda);

    return engine_err_ok;
}

int engine_cuda_finalize_bonds_all(engine *e) {
    size_t size_bonds = cuda_bonds_size * sizeof(MxBondCUDA);

    MxBondCUDA *bonds_cuda = (MxBondCUDA*)malloc(cuda_bonds_size * sizeof(MxBondCUDA));
    if(cudaMemcpy(bonds_cuda, cuda_bonds_device_arr, size_bonds, cudaMemcpyDeviceToHost) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    int nr_bonds = cuda_bonds_size;
    int nr_runners = e->nr_runners;
    auto func = [&bonds_cuda, nr_bonds, nr_runners](int tid) {
        for(int i = tid; i < nr_bonds; i += nr_runners) 
            bonds_cuda[i].finalize();
    };
    #if MX_THREADING
    mx::parallel_for(nr_runners, func);
    #else
    func(0);
    #endif

    if(cudaMemcpy(cuda_bonds_device_arr, bonds_cuda, size_bonds, cudaMemcpyHostToDevice) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    free(bonds_cuda);

    return engine_err_ok;
}

int engine_cuda_refresh_bond(MxBond *b) {
    if(engine_cuda_finalize_bond(b->id) < 0) 
        return error(engine_err);

    if(engine_cuda_add_bond(b) < 0) 
        return error(engine_err);

    return engine_err_ok;
}

int engine_cuda_refresh_bonds(engine *e, MxBond *bonds, int nr_bonds) { 
    int *binds = (int*)malloc(nr_bonds * sizeof(int));

    for(int i = 0; i < nr_bonds; i++) 
        binds[i] = bonds[i].id;

    if(engine_cuda_finalize_bonds(e, binds, nr_bonds) < 0) 
        return error(engine_err);

    if(engine_cuda_add_bonds(bonds, nr_bonds) < 0) 
        return error(engine_err);

    free(binds);

    return engine_err_ok;
}

__global__ void cuda_init_rand_unif_device(curandState *rand_unif, int nr_rands, unsigned long long seed) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < nr_rands) {
        curand_init(seed, tid, 0, &rand_unif[tid]);
        tid += stride;
    }
}

extern "C" int engine_cuda_rand_unif_init(struct engine *e) {

    if(cuda_rand_unif_init) 
        return engine_err_ok;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess)
        return cuda_error(engine_err_cuda);

    int nr_rands = cuda_bonds_nr_threads * cuda_bonds_nr_blocks;

    if(cudaMalloc(&rand_unif_cuda, sizeof(curandState) * nr_rands) != cudaSuccess)
        return cuda_error(engine_err_cuda);

    cuda_init_rand_unif_device<<<cuda_bonds_nr_blocks, cuda_bonds_nr_threads>>>((curandState *)rand_unif_cuda, nr_rands, cuda_rand_unif_seed);
    if(cudaPeekAtLastError() != cudaSuccess)
        return cuda_error(engine_err_cuda);

    if(cudaMemcpyToSymbol(cuda_rand_unif, &rand_unif_cuda, sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        return cuda_error(engine_err_cuda);

    cuda_rand_unif_init = true;

    return engine_err_ok;

}

int engine_cuda_rand_unif_finalize(struct engine *e) {

    if(!cuda_rand_unif_init) 
        return engine_err_ok;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess)
        return cuda_error(engine_err_cuda);

    if(cudaFree(rand_unif_cuda) != cudaSuccess)
        return cuda_error(engine_err_cuda);

    cuda_rand_unif_init = false;
    
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
extern "C" int engine_cuda_rand_unif_setSeed(struct engine *e, unsigned int seed, bool onDevice) {

    if(onDevice)
        if(engine_cuda_rand_unif_finalize(e) < 0)
            return cuda_error(engine_err_cuda);

    cuda_rand_unif_seed = seed;

    if(onDevice) {
        if(engine_cuda_rand_unif_init(e) < 0)
            return cuda_error(engine_err_cuda);

        if(cudaSetDevice(cuda_bonds_device) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }

    return engine_err_ok;

}

__device__ 
void engine_cuda_rand_uniform(float *result) { 
    *result = curand_uniform(&cuda_rand_unif[threadIdx.x + blockIdx.x * blockDim.x]);
}

__global__ 
void engine_cuda_rand_unifs_device(int nr_rands, float *result) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    auto s = &cuda_rand_unif[threadID];

    for(int tid = 0; tid < nr_rands; tid += stride) {
        result[tid] = curand_uniform(s);
    }
}

int engine_cuda_rand_unifs(int nr_rands, float *result, cudaStream_t stream) {
    int nr_blocks = std::min(cuda_bonds_nr_blocks, (unsigned int)std::ceil((float)nr_rands / cuda_bonds_nr_threads));

    engine_cuda_rand_unifs_device<<<nr_blocks, cuda_bonds_nr_threads, 0, stream>>>(nr_rands, result);
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    return engine_err_ok;
}

int cuda_bonds_arrays_malloc() {
    size_t size_bonds = cuda_bonds_nrparts_chunk * sizeof(MxBondCUDAData);
    size_t size_potenergies = cuda_bonds_nrparts_chunk * sizeof(float);
    size_t size_forces = 3 * size_potenergies;
    size_t size_todestroy = cuda_bonds_nrparts_chunk * sizeof(bool);

    if(cudaMalloc(&cuda_bonds_bonds_arr1, size_bonds) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaMalloc(&cuda_bonds_bonds_arr2, size_bonds) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if((cuda_bonds_bonds_local1 = (MxBondCUDAData*)malloc(size_bonds)) == NULL) 
        return engine_err_malloc;
    if((cuda_bonds_bonds_local2 = (MxBondCUDAData*)malloc(size_bonds)) == NULL) 
        return engine_err_malloc;

    if(cudaMalloc(&cuda_bonds_forces_arr1, size_forces) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaMalloc(&cuda_bonds_forces_arr2, size_forces) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if((cuda_bonds_forces_local1 = (float*)malloc(size_forces)) == NULL) 
        return engine_err_malloc;
    if((cuda_bonds_forces_local2 = (float*)malloc(size_forces)) == NULL) 
        return engine_err_malloc;

    if(cudaMalloc(&cuda_bonds_potenergies_arr1, size_potenergies) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaMalloc(&cuda_bonds_potenergies_arr2, size_potenergies) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if((cuda_bonds_potenergies_local1 = (float*)malloc(size_potenergies)) == NULL) 
        return engine_err_malloc;
    if((cuda_bonds_potenergies_local2 = (float*)malloc(size_potenergies)) == NULL) 
        return engine_err_malloc;

    if(cudaMalloc(&cuda_bonds_todestroy_arr1, size_todestroy) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaMalloc(&cuda_bonds_todestroy_arr2, size_todestroy) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if((cuda_bonds_todestroy_local1 = (bool*)malloc(size_todestroy)) == NULL) 
        return engine_err_malloc;
    if((cuda_bonds_todestroy_local2 = (bool*)malloc(size_todestroy)) == NULL)
        return engine_err_malloc;

    return engine_err_ok;
}

int cuda_bonds_arrays_free() {
    if(cudaFree(cuda_bonds_bonds_arr1) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaFree(cuda_bonds_bonds_arr2) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    free(cuda_bonds_bonds_local1);
    free(cuda_bonds_bonds_local2);

    if(cudaFree(cuda_bonds_forces_arr1) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaFree(cuda_bonds_forces_arr2) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    free(cuda_bonds_forces_local1);
    free(cuda_bonds_forces_local2);

    if(cudaFree(cuda_bonds_potenergies_arr1) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaFree(cuda_bonds_potenergies_arr2) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    free(cuda_bonds_potenergies_local1);
    free(cuda_bonds_potenergies_local2);

    if(cudaFree(cuda_bonds_todestroy_arr1) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(cudaFree(cuda_bonds_todestroy_arr2) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    free(cuda_bonds_todestroy_local1);
    free(cuda_bonds_todestroy_local2);

    return engine_err_ok;
}

int MxBondCUDA_setBlocks(const unsigned int &nr_blocks) {
    cuda_bonds_nr_blocks = nr_blocks;
    return engine_err_ok;
}

int MxBondCUDA_setThreads(const unsigned int &nr_threads) {
    cuda_bonds_nr_threads = nr_threads;
    return engine_err_ok;
}

__device__ 
void bond_eval_single_cuda(MxPotential *pot, float _r2, float *dx, float *force, float *epot_out) {
    float r2, ee, eff;
    int k;

    if(pot->kind == POTENTIAL_KIND_COMBINATION && pot->flags & POTENTIAL_SUM) {
        if(pot->pca != NULL) bond_eval_single_cuda(pot->pca, _r2, dx, force, epot_out);
        if(pot->pcb != NULL) bond_eval_single_cuda(pot->pcb, _r2, dx, force, epot_out);
        return;
    }

    if (_r2 < pot->a*pot->a || _r2 > pot->b*pot->b) 
        r2 = fmax(pot->a * pot->a, fmin(pot->b * pot->b, _r2));
    else 
        r2 = _r2;

    potential_eval_cuda(pot, r2, &ee, &eff);

    // Update the forces
    for (k = 0; k < 3; k++) {
        force[k] -= eff * dx[k];
    }

    // Tabulate the energy
    *epot_out += ee;
}


__global__ 
void bond_eval_cuda(MxBondCUDAData *bonds, int nr_bonds, float *forces, float *epot_out, bool *toDestroy) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, k;
    MxBondCUDA *b;
    MxBondCUDAData *bu;
    float dx[3], pix[3], r2, fix[3], epot = 0.f;
    int shift[3];
    float rn;

    for(i = tid; i < nr_bonds; i += stride) {
        bu = &bonds[i];
        b = &cuda_bonds[bu->id];

        if(!(b->flags & BOND_ACTIVE) || (bu->pi_flags & PARTICLE_GHOST && bu->pj_flags & PARTICLE_GHOST)) {
            forces[3 * i    ] = 0.f;
            forces[3 * i + 1] = 0.f;
            forces[3 * i + 2] = 0.f;
            epot_out[i] = 0.f;
            continue;
        }

        // Test for decay
        if(!isinf(b->half_life) && b->half_life > 0.f) { 
            engine_cuda_rand_uniform(&rn);
            if(1.0 - powf(2.0, -cuda_bonds_dt / b->half_life) > rn) { 
                toDestroy[i] = true;
                forces[3 * i    ] = 0.f;
                forces[3 * i + 1] = 0.f;
                forces[3 * i + 2] = 0.f;
                epot_out[i] = 0.f;
                continue;
            }
        }

        // Get the distance between the particles
        shift[0] = bu->cix.x - bu->cjx.x;
        shift[1] = bu->cix.y - bu->cjx.y;
        shift[2] = bu->cix.z - bu->cjx.z;
        for(k = 0; k < 3; k++) {
            if(shift[k] > 1) shift[k] = -1;
            else if(shift[k] < -1) shift[k] = 1;
        }
        pix[0] = bu->pix.x + cuda_bonds_cell_edge_lens.x * shift[0];
        pix[1] = bu->pix.y + cuda_bonds_cell_edge_lens.y * shift[1];
        pix[2] = bu->pix.z + cuda_bonds_cell_edge_lens.z * shift[2];

        r2 = 0.f;
        dx[0] = pix[0] - bu->pjx.x; r2 += dx[0]*dx[0];
        dx[1] = pix[1] - bu->pjx.y; r2 += dx[1]*dx[1];
        dx[2] = pix[2] - bu->pjx.z; r2 += dx[2]*dx[2];

        memset(fix, 0.f, 3 * sizeof(float));

        bond_eval_single_cuda(&b->p, r2, dx, fix, &epot);
        forces[3 * i    ] = fix[0];
        forces[3 * i + 1] = fix[1];
        forces[3 * i + 2] = fix[2];
        epot_out[i] = epot;

        // Test for dissociation
        toDestroy[i] = epot >= b->dissociation_energy;
    }
}

int engine_bond_flip_stream() {
    if(cuda_bonds_front_stream == 1) { 
        cuda_bonds_stream            = &cuda_bonds_stream2;

        cuda_bonds_bonds_arr         = cuda_bonds_bonds_arr2;
        cuda_bonds_forces_arr        = cuda_bonds_forces_arr2;
        cuda_bonds_potenergies_arr   = cuda_bonds_potenergies_arr2;
        cuda_bonds_todestroy_arr     = cuda_bonds_todestroy_arr2;

        cuda_bonds_bonds_local       = cuda_bonds_bonds_local2;
        cuda_bonds_forces_local      = cuda_bonds_forces_local2;
        cuda_bonds_potenergies_local = cuda_bonds_potenergies_local2;
        cuda_bonds_todestroy_local   = cuda_bonds_todestroy_local2;

        cuda_bonds_front_stream      = 2;
    }
    else {
        cuda_bonds_stream            = &cuda_bonds_stream1;

        cuda_bonds_bonds_arr         = cuda_bonds_bonds_arr1;
        cuda_bonds_forces_arr        = cuda_bonds_forces_arr1;
        cuda_bonds_potenergies_arr   = cuda_bonds_potenergies_arr1;
        cuda_bonds_todestroy_arr     = cuda_bonds_todestroy_arr1;

        cuda_bonds_bonds_local       = cuda_bonds_bonds_local1;
        cuda_bonds_forces_local      = cuda_bonds_forces_local1;
        cuda_bonds_potenergies_local = cuda_bonds_potenergies_local1;
        cuda_bonds_todestroy_local   = cuda_bonds_todestroy_local1;

        cuda_bonds_front_stream      = 1;
    }
    return engine_err_ok;
}

int engine_bond_cuda_load_bond_chunk(MxBond *bonds, int loc, int N) { 
    size_t size_bonds = N * sizeof(MxBondCUDAData);
    size_t size_potenergies = N * sizeof(float);
    size_t size_forces = 3 * size_potenergies;
    size_t size_todestroy = N * sizeof(bool);
    MxBond *buff = &bonds[loc];

    int nr_runners = _Engine.nr_runners;
    auto bl = cuda_bonds_bonds_local;
    auto func = [&bl, &buff, N, nr_runners](int tid) -> void {
        for(int j = tid; j < N; j += nr_runners) 
            bl[j] = MxBondCUDAData(&buff[j]);
    };
    #if MX_THREADING
    mx::parallel_for(nr_runners, func);
    #else
    func(0);
    #endif

    if(cudaMemcpyAsync(cuda_bonds_bonds_arr, cuda_bonds_bonds_local, size_bonds, cudaMemcpyHostToDevice, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    dim3 nr_threads(std::min((int)cuda_bonds_nr_threads, N), 1, 1);
    dim3 nr_blocks(std::min(cuda_bonds_nr_blocks, N / nr_threads.x), 1, 1);

    bond_eval_cuda<<<nr_blocks, nr_threads, 0, *cuda_bonds_stream>>>(
        cuda_bonds_bonds_arr, N, cuda_bonds_forces_arr, cuda_bonds_potenergies_arr, cuda_bonds_todestroy_arr
    );
    if(cudaPeekAtLastError() != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaMemcpyAsync(cuda_bonds_forces_local, cuda_bonds_forces_arr, size_forces, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaMemcpyAsync(cuda_bonds_potenergies_local, cuda_bonds_potenergies_arr, size_potenergies, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaMemcpyAsync(cuda_bonds_todestroy_local, cuda_bonds_todestroy_arr, size_todestroy, cudaMemcpyDeviceToHost, *cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    return engine_err_ok;
}

int engine_bond_cuda_unload_bond_chunk(MxBond *bonds, int loc, int N, struct engine *e, double *epot_out) { 
    int i, k;
    float epot = 0.f;
    float *bufff, ee;
    MxBond *buffb = &bonds[loc];

    for(i = 0; i < N; i++) {
        auto b = &buffb[i];
        
        ee = cuda_bonds_potenergies_local[i];
        epot += ee;
        b->potential_energy += ee;
        if(cuda_bonds_todestroy_local[i]) {
            MxBond_Destroy(b);
            continue;
        }

        bufff = &cuda_bonds_forces_local[3 * i];
        auto pi = e->s.partlist[b->i];
        auto pj = e->s.partlist[b->j];
        for(k = 0; k < 3; k++) {
            pi->f[k] += bufff[k];
            pj->f[k] -= bufff[k];
        }
    }
    
    // Store the potential energy.
    *epot_out += (double)epot;

    return engine_err_ok;
}

int engine_bond_eval_cuda(struct MxBond *bonds, int N, struct engine *e, double *epot_out) {
    int i, n;
    double epot = 0.0;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    engine_bond_flip_stream();
    
    if(N < cuda_bonds_nrparts_chunk) {
        if(engine_bond_cuda_load_bond_chunk(bonds, 0, N) < 0) 
            return error(engine_err);
        if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
            return cuda_error(engine_err_cuda);
        if(engine_bond_cuda_unload_bond_chunk(bonds, 0, N, e, epot_out) < 0) 
            return error(engine_err);
        return engine_err_ok;
    }

    n = cuda_bonds_nrparts_chunk;
    if(engine_bond_cuda_load_bond_chunk(bonds, 0, n) < 0) 
        return error(engine_err);
    for(i = cuda_bonds_nrparts_chunk; i < N; i += cuda_bonds_nrparts_chunk) {
        if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
            return cuda_error(engine_err_cuda);
        if(engine_bond_cuda_unload_bond_chunk(bonds, i - cuda_bonds_nrparts_chunk, n, e, &epot) < 0) 
            return error(engine_err);

        engine_bond_flip_stream();

        n = std::min(N - i, cuda_bonds_nrparts_chunk);
        if(engine_bond_cuda_load_bond_chunk(bonds, i, n) < 0) 
            return error(engine_err);
    }
    if(cudaStreamSynchronize(*cuda_bonds_stream) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    if(engine_bond_cuda_unload_bond_chunk(bonds, N - n, n, e, &epot) < 0) 
        return error(engine_err);

    *epot_out += (double)epot;
    
    return engine_err_ok;
}

int engine_bond_cuda_initialize(engine *e) {
    if(e->bonds_cuda) 
        return engine_err_ok;

    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error(engine_err_cuda);
    
    float3 cell_edge_lens_cuda = make_float3(e->s.h[0], e->s.h[1], e->s.h[2]);
    if(cudaMemcpyToSymbol(cuda_bonds_cell_edge_lens, &cell_edge_lens_cuda, sizeof(float3), 0, cudaMemcpyHostToDevice) != cudaSuccess )
        return cuda_error(engine_err_cuda);

    if(cudaMemcpyToSymbol(cuda_bonds_dt, &e->dt, sizeof(float), 0, cudaMemcpyHostToDevice) != cudaSuccess )
        return cuda_error(engine_err_cuda);

    if(cuda_bonds_nr_blocks == 0) {
        cuda_bonds_nr_blocks = MxCUDA::maxBlockDimX(0);
    }

    if(cuda_bonds_nr_threads == 0) {
        cuda_bonds_nr_threads = MxCUDA::maxThreadsPerBlock(0);
    }

    if(cudaStreamCreate(&cuda_bonds_stream1) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaStreamCreate(&cuda_bonds_stream2) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(engine_cuda_rand_unif_init(e) < 0) 
        return error(engine_err);

    if(cuda_bonds_arrays_malloc() < 0) 
        return error(engine_err);

    if(cuda_bonds_bonds_initialize(e, e->bonds, e->nr_bonds) < 0) 
        return error(engine_err);

    e->bonds_cuda = true;
    
    return engine_err_ok;
}

int engine_bond_cuda_finalize(engine *e) {
    if(cudaSetDevice(cuda_bonds_device) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cuda_bonds_bonds_finalize(e) < 0) 
        return error(engine_err);

    if(cudaStreamDestroy(cuda_bonds_stream1) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(cudaStreamDestroy(cuda_bonds_stream2) != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    if(engine_cuda_rand_unif_finalize(e) < 0) 
        return error(engine_err);

    if(cuda_bonds_arrays_free() < 0) 
        return error(engine_err);

    e->bonds_cuda = false;

    return engine_err_ok;
}

int engine_bond_cuda_refresh(engine *e) {
    if(engine_bond_cuda_finalize(e) < 0) 
        return error(engine_err);

    if(engine_bond_cuda_initialize(e) < 0) 
        return error(engine_err);

    if(cudaDeviceSynchronize() != cudaSuccess) 
        return cuda_error(engine_err_cuda);

    return engine_err_ok;
}

int MxBondCUDA_getDevice() {
    return cuda_bonds_device;
}

int MxBondCUDA_setDevice(engine *e, const int &deviceId) {
    bool refreshing = e->bonds_cuda;

    if(refreshing) 
        if(engine_bond_cuda_finalize(e) < 0) 
            return error(engine_err);

    cuda_bonds_device = deviceId;

    if(refreshing) 
        if(engine_bond_cuda_initialize(e) < 0) 
            return error(engine_err);

    return engine_err_ok;
}

int MxBondCUDA_toDevice(engine *e) {
    if(e->bonds_cuda) 
        return engine_err_ok;
    
    return engine_bond_cuda_initialize(e);
}

int MxBondCUDA_fromDevice(engine *e) {
    if(!e->bonds_cuda) 
        return engine_err_ok;
    
    return engine_bond_cuda_finalize(e);
}

int MxBondCUDA_refresh(engine *e) {
    if(!e->bonds_cuda) 
        return engine_err_ok;
    
    return engine_bond_cuda_refresh(e);
}

int MxBondCUDA_refreshBond(engine *e, MxBondHandle *b) {
    if(e->bonds_cuda) 
        return engine_err_ok;

    if(b == NULL) 
        return engine_err_null;

    return engine_cuda_refresh_bond(b->get());
}

int MxBondCUDA_refreshBonds(engine *e, MxBondHandle **bonds, int nr_bonds) {
    if(e->bonds_cuda) 
        return engine_err_ok;

    MxBond *bs = (MxBond*)malloc(nr_bonds * sizeof(MxBond));
    MxBondHandle *bh;
    for(int i = 0; i < nr_bonds; i++) { 
        bh = bonds[i];
        if(bh == NULL) 
            return engine_err_null;
        bs[i] = *(bh->get());
    }
    
    if(engine_cuda_refresh_bonds(e, bs, nr_bonds) < 0) 
        return error(engine_err);

    free(bs);

    return engine_err_ok;
}
