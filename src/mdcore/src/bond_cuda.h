/**
 * @file bond_cuda.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines bond kernels on CUDA-supporting GPUs
 * @date 2021-11-29
 * 
 */
#ifndef SRC_MDCORE_SRC_BOND_CUDA_H_
#define SRC_MDCORE_SRC_BOND_CUDA_H_

#include "MxPotential_cuda.h"
#include "bond.h"

struct MxBondCUDA { 
    uint32_t flags;
    float dissociation_energy;
    float half_life;
    int2 pids;

    MxPotential p;

    __host__ 
    MxBondCUDA();
    
    __host__ 
    MxBondCUDA(MxBond *b);

    __host__ 
    void finalize();
};

int MxBondCUDA_setThreads(const unsigned int &nr_threads);
int MxBondCUDA_setBlocks(const unsigned int &nr_blocks);
int MxBondCUDA_getDevice();
int MxBondCUDA_setDevice(engine *e, const int &deviceId);
int MxBondCUDA_toDevice(engine *e);
int MxBondCUDA_fromDevice(engine *e);
int MxBondCUDA_refresh(engine *e);
int MxBondCUDA_refreshBond(engine *e, MxBondHandle *b);
int MxBondCUDA_refreshBonds(engine *e, MxBondHandle **bonds, int nr_bonds);

int engine_cuda_add_bond(MxBond *b);
int engine_cuda_finalize_bond(int bind);
int engine_cuda_finalize_bonds(engine *e, int *binds, int nr_bonds);
int engine_cuda_finalize_bonds_all(engine *e);
int engine_bond_eval_cuda(struct MxBond *bonds, int N, struct engine *e, double *epot_out);

#endif // SRC_MDCORE_SRC_BOND_CUDA_H_
