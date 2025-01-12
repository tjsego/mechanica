/**
 * @file flux_cuda.cu
 * @author T.J. Sego, Ph.D.
 * @brief Defines flux kernels on CUDA-supporting GPUs
 * @date 2021-11-23
 * 
 */

#include "flux_cuda.h"

#include "errs.h"
#include <engine.h>

#include <cuda.h>

// Diagonal entries and flux index lookup table
__constant__ int *cuda_fxind;

// The fluxes
__constant__ struct MxFluxesCUDA *cuda_fluxes = NULL;
__constant__ int cuda_nr_fluxes = 0;

#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_safe_call(f)       { if(f != cudaSuccess) return cuda_error(engine_err_cuda); }


#define MXFLUXCUDA_CUDAMALLOC(member, type, member_name)                                                        \
    if(cudaMalloc(&member, this->size * sizeof(type)) != cudaSuccess) {                                         \
        printf("Flux allocation failed (%s): %s\n", member_name, cudaGetErrorString(cudaPeekAtLastError()));    \
        return;                                                                                                 \
    } 

template <typename T>
__host__ 
cudaError_t MxFluxCUDA_toDevice(struct MxFluxCUDA *f, T *memPtr, T *fluxPtr) {
    T *tmpPtr = (T*)malloc(f->size * sizeof(T));
    for(int i = 0; i < f->size; i++) tmpPtr[i] = fluxPtr[i];
    cudaMemcpy(memPtr, tmpPtr, f->size * sizeof(T), cudaMemcpyHostToDevice);
    free(tmpPtr);
    return cudaPeekAtLastError();
}

#define MXFLUXCUDA_MEMCPYH2D(member, flux_member, member_name)                                                  \
    if(MxFluxCUDA_toDevice(this, member, flux_member) != cudaSuccess) {                                        \
        printf("Flux member copy failed (%s): %s\n", member_name, cudaGetErrorString(cudaPeekAtLastError()));   \
        return;                                                                                                 \
    }


#define MXFLUXCUDA_CUDAFREE(member, member_name)                                                                \
    if(cudaFree(member) != cudaSuccess) {                                                                       \
        printf("Flux member free failed (%s): %s\n", member_name, cudaGetErrorString(cudaPeekAtLastError()));   \
        return;                                                                                                 \
    }


// MxFluxCUDA


__host__ 
MxFluxCUDA::MxFluxCUDA(MxFlux f) : 
    size{f.size}
{
    MXFLUXCUDA_CUDAMALLOC(this->kinds,      int8_t,                 "kinds");
    MXFLUXCUDA_CUDAMALLOC(this->type_ids,   MxFluxTypeIdPairCUDA,   "type_ids");
    MXFLUXCUDA_CUDAMALLOC(this->indices_a,  int32_t,                "indices_a");
    MXFLUXCUDA_CUDAMALLOC(this->indices_b,  int32_t,                "indices_b");
    MXFLUXCUDA_CUDAMALLOC(this->coef,       float,                  "coef");
    MXFLUXCUDA_CUDAMALLOC(this->decay_coef, float,                  "decay_coef");
    MXFLUXCUDA_CUDAMALLOC(this->target,     float,                  "target");

    MXFLUXCUDA_MEMCPYH2D(this->kinds,       f.kinds,        "kinds");
    MXFLUXCUDA_MEMCPYH2D(this->indices_a,   f.indices_a,    "indices_a");
    MXFLUXCUDA_MEMCPYH2D(this->indices_b,   f.indices_b,    "indices_b");
    MXFLUXCUDA_MEMCPYH2D(this->coef,        f.coef,         "coef");
    MXFLUXCUDA_MEMCPYH2D(this->decay_coef,  f.decay_coef,   "decay_coef");
    MXFLUXCUDA_MEMCPYH2D(this->target,      f.target,       "target");

    MxFluxTypeIdPairCUDA *_type_ids = (MxFluxTypeIdPairCUDA*)malloc(sizeof(MxFluxTypeIdPairCUDA) * this->size);
    for(int i = 0; i < this->size; i++) _type_ids[i] = MxFluxTypeIdPairCUDA(f.type_ids[i]);
    cudaMemcpy(this->type_ids, _type_ids, sizeof(MxFluxTypeIdPairCUDA) * this->size, cudaMemcpyHostToDevice);
    free(_type_ids);
}

__device__ 
void MxFluxCUDA::finalize() {
    MXFLUXCUDA_CUDAFREE(this->kinds,        "kinds");
    MXFLUXCUDA_CUDAFREE(this->indices_a,    "indices_a");
    MXFLUXCUDA_CUDAFREE(this->indices_b,    "indices_b");
    MXFLUXCUDA_CUDAFREE(this->coef,         "coef");
    MXFLUXCUDA_CUDAFREE(this->decay_coef,   "decay_coef");
    MXFLUXCUDA_CUDAFREE(this->target,       "target");
}


// MxFluxesCUDA


__host__ 
MxFluxesCUDA::MxFluxesCUDA(MxFluxes *f) : 
    size{f->size}
{
    if(cudaMalloc(&this->fluxes, sizeof(MxFluxCUDA) * this->size) != cudaSuccess) {
        printf("Fluxes allocation failed (fluxes): %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        return;
    }

    MxFluxCUDA *_fluxes = (MxFluxCUDA *)malloc(sizeof(MxFluxCUDA) * this->size);
    for(int i = 0; i < this->size; i++) _fluxes[i] = MxFluxCUDA(f->fluxes[i]);

    if(cudaMemcpy(this->fluxes, _fluxes, sizeof(MxFluxCUDA) * this->size, cudaMemcpyHostToDevice) != cudaSuccess) 
        printf("Fluxes member free failed (fluxes): %s\n", cudaGetErrorString(cudaPeekAtLastError()));

    free(_fluxes);
}

__device__ 
void MxFluxesCUDA::finalize() {
    for(int i = 0; i < this->size; i++)
        this->fluxes[i].finalize();
    
    if(cudaFree(this->fluxes) != cudaSuccess)
        printf("Fluxes member free failed (fluxes): %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}

__device__ 
void MxFluxCUDA_getFluxes(int **fxind_cuda, MxFluxesCUDA **fluxes_cuda) {
    *fxind_cuda = cuda_fxind;
    *fluxes_cuda = cuda_fluxes;
}

__device__ 
void MxFluxCUDA_getNrFluxes(unsigned int *nr_fluxes) {
    *nr_fluxes = cuda_nr_fluxes;
}

__device__ 
void MxFluxCUDA_getNrStates(unsigned int *nr_states) {
    *nr_states = cuda_nr_fluxes - 1;
}


/**
 * @brief Load the fluxes onto the CUDA device
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_cuda_load_fluxes(struct engine *e) {

    int i, j, nr_fluxes;
    int did;
    int nr_devices = e->nr_devices;
    int *fxind = (int*)malloc(sizeof(int) * e->max_type * e->max_type);
    struct MxFluxes **fluxes = (MxFluxes**)malloc(sizeof(MxFluxes*) * e->nr_types * (e->nr_types + 1) / 2 + 1);
    float cutoff = e->s.cutoff;
    
    // Start by identifying the unique fluxes in the engine
    nr_fluxes = 1;
    for(i = 0 ; i < e->max_type * e->max_type ; i++) {
    
        /* Skip if there is no flux or no parts of this type. */
        if ( e->fluxes[i] == NULL )
            continue;

        /* Check this flux against previous fluxes. */
        for ( j = 0 ; j < nr_fluxes && e->fluxes[i] != fluxes[j] ; j++ );
        if ( j < nr_fluxes )
            continue;

        /* Store this flux and the number of coefficient entries it has. */
        fluxes[nr_fluxes] = e->fluxes[i];
        nr_fluxes += 1;
    
    }

    /* Pack the flux matrix. */
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
        if ( e->fluxes[i] == NULL ) {
            fxind[i] = 0;
        }
        else {
            for ( j = 0 ; j < nr_fluxes && fluxes[j] != e->fluxes[i] ; j++ );
            fxind[i] = j;
        }
    }

    // Pack the fluxes
    MxFluxesCUDA *fluxes_cuda = (MxFluxesCUDA*)malloc(sizeof(MxFluxCUDA) * nr_fluxes);
    for(i = 1; i < nr_fluxes; i++) {
        fluxes_cuda[i] = MxFluxesCUDA(fluxes[i]);
    }
    
    /* Store find and other stuff as constant. */
    for(did = 0; did < nr_devices; did++) {
        cuda_safe_call(cudaSetDevice(e->devices[did]));
        cuda_safe_call(cudaMalloc(&e->fxind_cuda[did], sizeof(int) * e->max_type * e->max_type));
        cuda_safe_call(cudaMemcpy(e->fxind_cuda[did], fxind, sizeof(int) * e->max_type * e->max_type, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpyToSymbol(cuda_fxind, &e->fxind_cuda[did], sizeof(void *), 0, cudaMemcpyHostToDevice));
    }
    free(fxind);

    // Store the fluxes
    for(did = 0; did < nr_devices; did++) {
        cuda_safe_call(cudaSetDevice(e->devices[did]));
        cuda_safe_call(cudaMalloc(&e->fluxes_cuda[did], sizeof(MxFluxesCUDA) * nr_fluxes));
        cuda_safe_call(cudaMemcpy(e->fluxes_cuda[did], fluxes_cuda, sizeof(MxFluxesCUDA) * nr_fluxes, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpyToSymbol(cuda_fluxes, &e->fluxes_cuda[did], sizeof(void*), 0, cudaMemcpyHostToDevice));
        cuda_safe_call(cudaMemcpyToSymbol(cuda_nr_fluxes, &nr_fluxes , sizeof(int) , 0 , cudaMemcpyHostToDevice ) );
    }
    free(fluxes);
    free(fluxes_cuda);

    int nr_states_current = e->nr_fluxes_cuda - 1;
    int nr_states_next = nr_fluxes - 1;

    if(nr_states_current > 0 && nr_states_next == 0 && engine_cuda_finalize_particle_states(e) < 0) 
        return error(engine_err_cuda);

    e->nr_fluxes_cuda = nr_fluxes;

    if(nr_states_next > 0 && nr_states_current == 0 && engine_cuda_allocate_particle_states(e) < 0) 
        return error(engine_err_cuda);
    else if(nr_states_current != nr_states_next && engine_cuda_refresh_particle_states(e) < 0) 
        return error(engine_err_cuda);

    // Allocate the flux buffer
    if(nr_states_next > 0) {
        for(did = 0; did < nr_devices; did++) {
            cuda_safe_call(cudaSetDevice(e->devices[did]));
            cuda_safe_call(cudaMalloc(&e->fluxes_next_cuda[did], sizeof(float) * nr_states_next * e->s.size_parts));
        }
    }

    return engine_err_ok;
}


__global__ 
void engine_cuda_unload_fluxes_device(int nr_fluxes) {
    if(nr_fluxes < 1)
        return;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    if(tid == 0) 
        tid += stride;

    while(tid < nr_fluxes) {
        cuda_fluxes[tid].finalize();

        tid += stride;
    }
}


/**
 * @brief Unload the fluxes on the CUDA device
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
extern "C" int engine_cuda_unload_fluxes(struct engine *e) {

    int nr_states = e->nr_fluxes_cuda - 1;

    for(int did = 0; did < e->nr_devices; did++) {

        cuda_safe_call(cudaSetDevice(e->devices[did]));

        // Free the fluxes.
        
        cuda_safe_call(cudaFree(e->fxind_cuda[did]));
        
        engine_cuda_unload_fluxes_device<<<8, 512>>>(nr_states);

        cuda_safe_call(cudaPeekAtLastError());
        
        cuda_safe_call(cudaFree((MxFluxesCUDA*)e->fluxes_cuda[did]));

        cuda_safe_call(cudaFree(e->fluxes_next_cuda[did]));

    }

    e->nr_fluxes_cuda = 0;

    return engine_err_ok;
}

/**
 * @brief Refresh the fluxes on the CUDA device. 
 * 
 * Can be safely called while on the CUDA device to reload all flux data from the engine. 
 * 
 * @param e The #engine
 * 
 * @return #engine_err_ok or < 0 on error (see #engine_err)
 */
extern "C" int engine_cuda_refresh_fluxes(struct engine *e) {
    
    if(engine_cuda_unload_fluxes(e) < 0)
        return error(engine_err);

    if(engine_cuda_load_fluxes(e) < 0)
        return error(engine_err);

    for(int did = 0; did < e->nr_devices; did++) {

        cuda_safe_call(cudaSetDevice(e->devices[did]));

        cuda_safe_call(cudaDeviceSynchronize());

    }

    return engine_err_ok;
}
