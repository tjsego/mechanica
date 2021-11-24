/**
 * @file flux_cuda.cu
 * @author T.J. Sego, Ph.D.
 * @brief Defines flux kernels on CUDA-supporting GPUs
 * @date 2021-11-23
 * 
 */

#include "flux_cuda.h"
#include <mdcore_config.h>

#include <Flux.hpp>
#include "errs.h"
#include <engine.h>

#include <cuda.h>

// Diagonal entries and flux index lookup table
__constant__ unsigned int *cuda_fxind;

// The part states
__constant__ float *cuda_part_states;

__constant__ float cuda_cutoff_flx = 0.f;

// The fluxes
__constant__ struct MxFluxesCUDA *cuda_fluxes = NULL;
__constant__ unsigned int cuda_nr_fluxes = 0;

#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )


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


struct MxFluxTypeIdPairCUDA {
    int16_t a;
    int16_t b;

    MxFluxTypeIdPairCUDA(TypeIdPair tip) : a{tip.a}, b{tip.b} {}
};


// A wrap of MxFlux
struct MxFluxCUDA {
    int32_t size;
    int8_t *kinds;
    MxFluxTypeIdPairCUDA *type_ids;
    int32_t *indices_a;
    int32_t *indices_b;
    float *coef;
    float *decay_coef;
    float *target;

    __host__ 
    MxFluxCUDA(MxFlux f);

    __device__ 
    void finalize();
};

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


// A wrap of MxFluxes
struct MxFluxesCUDA {
    int32_t size;
    MxFluxCUDA *fluxes;

    __host__ 
    MxFluxesCUDA(MxFluxes *f);

    __device__ 
    void finalize();
};

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
void flux_fick_cuda(MxFluxCUDA flux, int i, float si, float sj, float *result) {
    *result *= flux.coef[i] * (si - sj);
}

__device__ 
void flux_secrete_cuda(MxFluxCUDA flux, int i, float si, float sj, float *result) {
    float q = flux.coef[i] * (si - flux.target[i]);
    float scale = q > 0.f;  // forward only, 1 if > 0, 0 if < 0.
    *result *= scale * q;
}

__device__ 
void flux_uptake_cuda(MxFluxCUDA flux, int i, float si, float sj, float *result) {
    float q = flux.coef[i] * (flux.target[i] - sj) * si;
    float scale = q > 0.f;
    *result *= scale * q;
}

__device__ 
void flux_eval_ex_cuda(unsigned int typeTableIndex, float r, float *states_i, float *states_j, int type_i, int type_j, float *qvec_i, bool *result) {
    
    // Check whether requested calculations are relevant
    
    unsigned int fluxes_index = cuda_fxind[typeTableIndex];
    if(fluxes_index == 0) {
        *result = false;
        return;
    }

    // Do calculations

    float ssi, ssj;
    float q;

    *result = true;
    
    MxFluxCUDA flux = cuda_fluxes[fluxes_index].fluxes[0];
    float term = 1. - r / cuda_cutoff_flx;
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
                assert(0);
        }

        qvec_i[qind] += q - 0.5 * flux.decay_coef[i] * states_i[qind];
    }
}

__device__ 
void flux_eval_ex_cuda(unsigned int typeTableIndex, float r, float *states_i, float *states_j, int type_i, int type_j, float *qvec_i, float *qvec_j, bool *result) {
    
    // Check whether requested calculations are relevant
    
    unsigned int fluxes_index = cuda_fxind[typeTableIndex];
    if(fluxes_index == 0) {
        *result = false;
        return;
    }

    // Do calculations

    *result = true;
    
    MxFluxCUDA flux = cuda_fluxes[fluxes_index].fluxes[0];
    float term = 1. - r / cuda_cutoff_flx;
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
                assert(0);
        }
        
        float half_decay = flux.decay_coef[i] * 0.5;
        qi[flux.indices_a[i]] -= q + half_decay * ssi;
        qj[flux.indices_b[i]] += q - half_decay * ssj;
    }
}

__device__ 
void MxFluxCUDA_getPartStates(float **result) {
    *result = cuda_part_states;
}

__device__ 
void MxFluxCUDA_getNrFluxes(unsigned int *nr_fluxes) {
    *nr_fluxes = cuda_nr_fluxes;
}

__device__ 
void MxFluxCUDA_getNrStates(unsigned int *nr_states) {
    *nr_states = cuda_nr_fluxes - 1;
}

__global__ 
void MxFluxCUDA_copy_partstates(float *states, int count, int ind, int nr_states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j, k;

    while(i < count) {
        k = i + ind;
        for(j = 0; j < nr_states; j++) 
            states[i * nr_states + j] = cuda_part_states[k * nr_states + j];
        i += blockDim.x * gridDim.x;
    }
}


/**
 * @brief Allocate the particle states on the CUDA device. 
 * 
 * All operations are performed according to engine current state. 
 * 
 * The performed operations are already applied while managing particle data. 
 * However, state data can also change with changes in fluxes, and so this allows 
 * data allocation without allocating particle data. Effectively, this exclusively 
 * allocates all state data according to the current configuration of the engine. 
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_cuda_allocate_part_states(struct engine *e) { 

    int nr_states = e->nr_fluxes_cuda - 1;

    if(nr_states < 1) 
        return engine_err_ok;

    // Allocate the particle state buffer
    if((e->part_states_cuda_local = (float*)malloc(sizeof(float) * nr_states * e->s.size_parts)) == NULL)
        return error(engine_err_malloc);

    /* Allocate the particle state data. */
    for(int did = 0; did < e->nr_devices; did++) {
        if (cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if (cudaMalloc(&e->part_states_cuda[did], sizeof(float) * nr_states * e->s.size_parts) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if (cudaMemcpyToSymbol(cuda_part_states, &e->part_states_cuda[did], sizeof(void *), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }

    return engine_err_ok;
}

/**
 * @brief Finalize the particle states on the CUDA device. 
 * 
 * All operations are performed according to engine current state. 
 * 
 * The performed operations are already applied while managing particle data. 
 * However, state data can also change with changes in fluxes, and so this allows 
 * data deallocation without deallocating particle data. Effectively, this exclusively 
 * deallocates all state data according to the current configuration of the engine. 
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
int engine_cuda_finalize_part_states(struct engine *e) { 

    int nr_states = e->nr_fluxes_cuda - 1;

    if(nr_states < 1) 
        return engine_err_ok;

    for(int did = 0; did < e->nr_devices; did++) {

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        // Free the particle state data

        if(cudaFree(e->part_states_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    // Free the particle state buffer

    free(e->part_states_cuda_local);

    return engine_err_ok;
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
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->fxind_cuda[did] , sizeof(unsigned int) * e->max_type * e->max_type ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( e->fxind_cuda[did] , fxind , sizeof(unsigned int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_fxind , &e->fxind_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if(cudaMemcpyToSymbol(cuda_cutoff_flx, &cutoff, sizeof(float), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
    }
    free(fxind);

    // Store the fluxes
    for(did = 0; did < nr_devices; did++) {
        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMalloc(&e->fluxes_cuda[did], sizeof(MxFluxesCUDA) * nr_fluxes) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpy(e->fluxes_cuda[did], fluxes_cuda, sizeof(MxFluxesCUDA) * nr_fluxes, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpyToSymbol(cuda_fluxes, &e->fluxes_cuda[did], sizeof(void*), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        if(cudaMemcpyToSymbol(cuda_nr_fluxes, &nr_fluxes , sizeof(unsigned int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    }
    free(fluxes);
    free(fluxes_cuda);

    int nr_states = e->nr_fluxes_cuda - 1;
    bool updating_states = e->nr_fluxes_cuda != nr_fluxes;

    if(updating_states && nr_states > 0) 
        if(engine_cuda_finalize_part_states(e) < 0) 
            return error(engine_err);

    e->nr_fluxes_cuda = nr_fluxes;
    nr_states = e->nr_fluxes_cuda - 1;

    if(updating_states && nr_states > 0) 
        if(engine_cuda_allocate_part_states(e) < 0) 
            return error(engine_err);

    // Allocate the flux buffer
    if(nr_states > 0) {
        for(did = 0; did < nr_devices; did++) {
            if(cudaSetDevice(e->devices[did]) != cudaSuccess)
                return cuda_error(engine_err_cuda);
            if (cudaMalloc(&e->fluxes_next_cuda[did], sizeof(float) * nr_states * e->s.size_parts) != cudaSuccess)
                return cuda_error(engine_err_cuda);
        }
    }

    return engine_err_ok;
}


__global__ 
void engine_cuda_unload_fluxes_device(int nr_fluxes) {
    if(nr_fluxes <= 1)
        return;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

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

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        // Free the states

        if(nr_states > 0)
            if(engine_cuda_finalize_part_states(e) < 0)
                return error(engine_err);

        // Free the fluxes.
        
        if(cudaFree(e->fxind_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);
        
        engine_cuda_unload_fluxes_device<<<8, 512>>>(nr_states);
        
        if(cudaFree((MxFluxesCUDA*)e->fluxes_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaFree(e->fluxes_next_cuda[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

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

        if(cudaSetDevice(e->devices[did]) != cudaSuccess)
            return cuda_error(engine_err_cuda);

        if(cudaDeviceSynchronize() != cudaSuccess)
            return cuda_error(engine_err_cuda);

    }

    return engine_err_ok;
}
