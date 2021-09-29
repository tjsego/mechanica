/**
 * @file cell_polarity.h
 * @author T.J. Sego, Ph.D.
 * @brief Implements model with additional features defined in 
 * Nielsen, Bjarke Frost, et al. "Model to link cell shape and polarity with organogenesis." Iscience 23.2 (2020): 100830.
 * @date 2021-09-20
 * 
 */
#ifndef MODELS_CENTER_CELL_POLARITY_POTENTIAL
#define MODELS_CENTER_CELL_POLARITY_POTENTIAL

#include <MxParticle.h>
#include <angle.h>
#include <bond.h>
#include <MxForce.h>
#include <MxParticleTypeList.h>
#include <MxPotential.h>
#include <rendering/MxArrowRenderer.h>

#include <string>
#include <unordered_map>

/**
 * @brief Gets the AB polarity vector of a cell
 * 
 * @param pId 
 * @param current 
 * @return MxVector3f 
 */
MxVector3f MxCellPolarity_GetVectorAB(const int &pId, const bool &current=true);

/**
 * @brief Gets the PCP polarity vector of a cell
 * 
 * @param pId 
 * @param current 
 * @return MxVector3f 
 */
MxVector3f MxCellPolarity_GetVectorPCP(const int &pId, const bool &current=true);

/**
 * @brief Sets the AB polarity vector of a cell
 * 
 * @param pId 
 * @param pVec 
 * @param current 
 */
void MxCellPolarity_SetVectorAB(const int &pId, const MxVector3f &pVec, const bool &current=true);

/**
 * @brief Sets the PCP polarity vector of a cell
 * 
 * @param pId 
 * @param pVec 
 * @param current 
 */
void MxCellPolarity_SetVectorPCP(const int &pId, const MxVector3f &pVec, const bool &current=true);

/**
 * @brief Updates all running polarity models
 * 
 */
void MxCellPolarity_update();

/**
 * @brief Registers a particle as polar. 
 * 
 * This must be called before the first integration step.
 * Otherwise, the engine will not know that the particle 
 * is polar and will be ignored. 
 * 
 * @param ph 
 */
void MxCellPolarity_register(MxParticleHandle *ph);

/**
 * @brief Unregisters a particle as polar. 
 * 
 * This must be called before destroying a registered particle. 
 * 
 * @param ph 
 */
void MxCellPolarity_unregister(MxParticleHandle *ph);

/**
 * @brief Registers a particle type as polar. 
 * 
 * This must be called on a particle type before any other type-specific operations. 
 * 
 * @param pType particle type
 * @param initMode initialization mode for particles of this type
 * @param initPolarAB initial value of AB polarity vector; only used when initMode="value"
 * @param initPolarPCP initial value of PCP polarity vector; only used when initMode="value"
 */
void MxCellPolarity_registerType(MxParticleType *pType, 
                                 const std::string &initMode="random", 
                                 const MxVector3f &initPolarAB=MxVector3f(0.0), 
                                 const MxVector3f &initPolarPCP=MxVector3f(0.0));

/**
 * @brief Gets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @return const std::string 
 */
const std::string MxCellPolarity_GetInitMode(MxParticleType *pType);

/**
 * @brief Sets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @param value initialization mode
 */
void MxCellPolarity_SetInitMode(MxParticleType *pType, const std::string &value);

/**
 * @brief Gets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @return const MxVector3f 
 */
const MxVector3f MxCellPolarity_GetInitPolarAB(MxParticleType *pType);

/**
 * @brief Sets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @param value initial AB polar vector
 */
void MxCellPolarity_SetInitPolarAB(MxParticleType *pType, const MxVector3f &value);

/**
 * @brief Gets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @return const MxVector3f 
 */
const MxVector3f MxCellPolarity_GetInitPolarPCP(MxParticleType *pType);

/**
 * @brief Sets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @param value initial PCP polar vector
 */
void MxCellPolarity_SetInitPolarPCP(MxParticleType *pType, const MxVector3f &value);

struct PolarityForcePersistent : MxForce {
    float sensAB = 0.0;
    float sensPCP = 0.0;
};

/**
 * @brief Creates a persistent polarity force. 
 * 
 * @param sensAB sensitivity to AB vector
 * @param sensPCP sensitivity to PCP vector
 * @return PolarityForcePersistent* 
 */
PolarityForcePersistent *MxCellPolarity_createForce_persistent(const float &sensAB=0.0, const float &sensPCP=0.0);

typedef enum PolarContactType {
    REGULAR = 0, 
    ISOTROPIC = 1, 
    ANISOTROPIC = 2
} PolarContactType;

struct MxPolarityArrowData : MxArrowData {
    float arrowLength = 1.0;
};

/**
 * @brief Toggles whether polarity vectors are rendered
 * 
 * @param _draw rendering flag; vectors are rendered when true
 */
void MxCellPolarity_SetDrawVectors(const bool &_draw);

/**
 * @brief Sets rendered polarity vector colors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param colorAB name of AB vector color
 * @param colorPCP name of PCP vector color
 */
void MxCellPolarity_SetArrowColors(const std::string &colorAB, const std::string &colorPCP);

/**
 * @brief Sets scale of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _scale scale of rendered vectors
 */
void MxCellPolarity_SetArrowScale(const float &_scale);

/**
 * @brief Sets length of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _length length of rendered vectors
 */
void MxCellPolarity_SetArrowLength(const float &_length);

/**
 * @brief Gets the rendering info for the AB polarity vector of a cell
 * 
 * @param pId 
 * @return MxArrowData* 
 */
MxPolarityArrowData *MxCellPolarity_GetVectorArrowAB(const int32_t &pId);

/**
 * @brief Gets the rendering info for the PCP polarity vector of a cell
 * 
 * @param pId 
 * @return MxArrowData* 
 */
MxPolarityArrowData *MxCellPolarity_GetVectorArrowPCP(const int32_t &pId);

/**
 * @brief Runs the polarity model along with a simulation. 
 * Must be called before doing any operations with this module. 
 * 
 */
void MxCellPolarity_load();

struct MxCellPolarityPotentialContact : MxPotential {
    float couplingFlat;  // lambda1
    float couplingOrtho;  // lambda2
    float couplingLateral;  // lambda3
    float distanceCoeff;  // beta
    PolarContactType cType;
    float mag;
    float rate;
    float bendingCoeff;

    MxCellPolarityPotentialContact();
};

/**
 * @brief Creates a contact-mediated polarity potential
 * 
 * @param cutoff cutoff distance
 * @param mag magnitude of force
 * @param rate rate of state vector dynamics
 * @param distanceCoeff distance coefficient
 * @param couplingFlat flat coupling coefficient
 * @param couplingOrtho orthogonal coupling coefficient
 * @param couplingLateral lateral coupling coefficient
 * @param contactType type of contact; available are regular, isotropic, anisotropic
 * @param bendingCoeff bending coefficient
 * @return MxCellPolarityPotentialContact* 
 */
MxCellPolarityPotentialContact *potential_create_cellpolarity(const float &cutoff, 
                                                              const float &mag=1.0, 
                                                              const float &rate=1.0,
                                                              const float &distanceCoeff=1.0, 
                                                              const float &couplingFlat=1.0, 
                                                              const float &couplingOrtho=0.0, 
                                                              const float &couplingLateral=0.0, 
                                                              std::string contactType="regular", 
                                                              const float &bendingCoeff=0.0);

namespace mx { namespace models { namespace center {

struct CellPolarity {

    /**
     * @brief Gets the AB polarity vector of a cell
     * 
     * @param pId 
     * @param current 
     * @return MxVector3f 
     */
    static MxVector3f getVectorAB(const int &pId, const bool &current=true);

    /**
     * @brief Gets the PCP polarity vector of a cell
     * 
     * @param pId 
     * @param current 
     * @return MxVector3f 
     */
    static MxVector3f getVectorPCP(const int &pId, const bool &current=true);

    /**
     * @brief Sets the AB polarity vector of a cell
     * 
     * @param pId 
     * @param pVec 
     * @param current 
     */
    static void setVectorAB(const int &pId, const MxVector3f &pVec, const bool &current=true);

    /**
     * @brief Sets the PCP polarity vector of a cell
     * 
     * @param pId 
     * @param pVec 
     * @param current 
     */
    static void setVectorPCP(const int &pId, const MxVector3f &pVec, const bool &current=true);

    /**
     * @brief Updates all running polarity models
     * 
     */
    static void update();

    /**
     * @brief Registers a particle as polar. 
     * 
     * This must be called before the first integration step. 
     * Otherwise, the engine will not know that the particle 
     * is polar and will be ignored. 
     * 
     * @param ph 
     */
    static void registerParticle(MxParticleHandle *ph);

    /**
     * @brief Unregisters a particle as polar. 
     * 
     * @param ph 
     */
    static void unregister(MxParticleHandle *ph);

    /**
     * @brief Registers a particle type as polar. 
     * 
     * This must be called on a particle type before any other type-specific operations. 
     * 
     * @param pType particle type
     * @param initMode initialization mode for particles of this type
     * @param initPolarAB initial value of AB polarity vector; only used when initMode="value"
     * @param initPolarPCP initial value of PCP polarity vector; only used when initMode="value"
     */
    static void registerType(MxParticleType *pType, 
                                    const std::string &initMode="random", 
                                    const MxVector3f &initPolarAB=MxVector3f(0.0), 
                                    const MxVector3f &initPolarPCP=MxVector3f(0.0));

    /**
     * @brief Gets the name of the initialization mode of a type
     * 
     * @param pType a type
     * @return const std::string 
     */
    static const std::string getInitMode(MxParticleType *pType);

    /**
     * @brief Sets the name of the initialization mode of a type
     * 
     * @param pType a type
     * @param value initialization mode
     */
    static void setInitMode(MxParticleType *pType, const std::string &value);

    /**
     * @brief Gets the initial AB polar vector of a type
     * 
     * @param pType a type
     * @return const MxVector3f 
     */
    static const MxVector3f getInitPolarAB(MxParticleType *pType);

    /**
     * @brief Sets the initial AB polar vector of a type
     * 
     * @param pType a type
     * @param value initial AB polar vector
     */
    static void setInitPolarAB(MxParticleType *pType, const MxVector3f &value);

    /**
     * @brief Gets the initial PCP polar vector of a type
     * 
     * @param pType a type
     * @return const MxVector3f 
     */
    static const MxVector3f getInitPolarPCP(MxParticleType *pType);

    /**
     * @brief Sets the initial PCP polar vector of a type
     * 
     * @param pType a type
     * @param value initial PCP polar vector
     */
    static void setInitPolarPCP(MxParticleType *pType, const MxVector3f &value);

    /**
     * @brief Creates a persistent polarity force. 
     * 
     * @param sensAB sensitivity to AB vector
     * @param sensPCP sensitivity to PCP vector
     * @return PolarityForcePersistent* 
     */
    static PolarityForcePersistent *forcePersistent(const float &sensAB=0.0, const float &sensPCP=0.0);

    /**
     * @brief Toggles whether polarity vectors are rendered
     * 
     * @param _draw rendering flag; vectors are rendered when true
     */
    static void setDrawVectors(const bool &_draw);

    /**
     * @brief Sets rendered polarity vector colors. 
     * 
     * Applies to subsequently created vectors and all current vectors. 
     * 
     * @param colorAB name of AB vector color
     * @param colorPCP name of PCP vector color
     */
    static void setArrowColors(const std::string &colorAB, const std::string &colorPCP);

    /**
     * @brief Sets scale of rendered polarity vectors. 
     * 
     * Applies to subsequently created vectors and all current vectors. 
     * 
     * @param _scale scale of rendered vectors
     */
    static void setArrowScale(const float &_scale);

    /**
     * @brief Sets length of rendered polarity vectors. 
     * 
     * Applies to subsequently created vectors and all current vectors. 
     * 
     * @param _length length of rendered vectors
     */
    static void setArrowLength(const float &_length);

    /**
     * @brief Gets the rendering info for the AB polarity vector of a cell
     * 
     * @param pId 
     * @return MxArrowData* 
     */
    static MxPolarityArrowData *getVectorArrowAB(const int32_t &pId);

    /**
     * @brief Gets the rendering info for the PCP polarity vector of a cell
     * 
     * @param pId 
     * @return MxArrowData* 
     */
    static MxPolarityArrowData *getVectorArrowPCP(const int32_t &pId);

    /**
     * @brief Runs the polarity model along with a simulation. 
     * Must be called before doing any operations with this module. 
     * 
     */
    static void load();

    /**
     * @brief Creates a contact-mediated polarity potential
     * 
     * @param cutoff cutoff distance
     * @param mag magnitude of force
     * @param rate rate of state vector dynamics
     * @param distanceCoeff distance coefficient
     * @param couplingFlat flat coupling coefficient
     * @param couplingOrtho orthogonal coupling coefficient
     * @param couplingLateral lateral coupling coefficient
     * @param contactType type of contact; available are regular, isotropic, anisotropic
     * @param bendingCoeff bending coefficient
     * @return MxCellPolarityPotentialContact* 
     */
    static MxCellPolarityPotentialContact *potentialContact(const float &cutoff, 
                                                            const float &mag=1.0, 
                                                            const float &rate=1.0,
                                                            const float &distanceCoeff=1.0, 
                                                            const float &couplingFlat=1.0, 
                                                            const float &couplingOrtho=0.0, 
                                                            const float &couplingLateral=0.0, 
                                                            std::string contactType="regular", 
                                                            const float &bendingCoeff=0.0);

};

}}}

#endif // MODELS_CENTER_CELL_POLARITY_POTENTIAL