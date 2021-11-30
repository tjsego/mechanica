/**
 * @file angle_cuda.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines angle kernels on CUDA-supporting GPUs
 * @date 2021-11-30
 * 
 */
#ifndef SRC_MDCORE_SRC_ANGLE_CUDA_H_
#define SRC_MDCORE_SRC_ANGLE_CUDA_H_

#include "MxPotential_cuda.h"
#include "angle.h"

struct MxAngleCUDA { 
    uint32_t flags;
    float dissociation_energy;
    float half_life;
    int3 pids;

    MxPotential p;

    __host__ 
    MxAngleCUDA();
    
    __host__ 
    MxAngleCUDA(MxAngle *a);

    __host__ 
    void finalize();
};

int MxAngleCUDA_setThreads(const unsigned int &nr_threads);
int MxAngleCUDA_setBlocks(const unsigned int &nr_blocks);
int MxAngleCUDA_getDevice();
int MxAngleCUDA_toDevice(engine *e);
int MxAngleCUDA_fromDevice(engine *e);
int MxAngleCUDA_refresh(engine *e);
int MxAngleCUDA_refreshAngle(engine *e, MxAngleHandle *a);
int MxAngleCUDA_refreshAngles(engine *e, MxAngleHandle **angles, int nr_angles);

int engine_cuda_add_angle(MxAngleHandle *ah);
int engine_cuda_finalize_angle(int aind);
int engine_cuda_finalize_angles(engine *e, int *ainds, int nr_angles);
int engine_cuda_finalize_angles_all(engine *e);
int engine_angle_eval_cuda(struct MxAngle *angles, int N, struct engine *e, double *epot_out);

#endif // SRC_MDCORE_SRC_ANGLE_CUDA_H_
