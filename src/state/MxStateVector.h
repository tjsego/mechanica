/**
 * @file MxStateVector.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the state vector; derived from carbon CStateVector.hpp written by Andy Somogyi
 * @date 2021-07-03
 * 
 */

#ifndef SRC_STATE_MXSTATEVECTOR_H_
#define SRC_STATE_MXSTATEVECTOR_H_

#include <mx_port.h>
#include <io/mx_io.h>
#include <stdio.h>
#include <string>

#include "MxSpeciesList.h"

enum StateVectorFlags {
    STATEVECTOR_NONE            = 0,
    STATEVECTOR_OWNMEMORY       = 1 << 0,
};

/**
 * @brief A state vector of an object. 
 */
struct CAPI_EXPORT MxStateVector {
    uint32_t flags;
    uint32_t size;

    /**
     * @brief Species of the state vector
     */
    struct MxSpeciesList *species;
    
    /**
     * owner of this state vector, usually a
     * MxParticle, but we leave the door open for other
     * kinds of things.
     */
    void *owner;
    
    void* data;
    
    // vector of values
    union {
        float *fvec;
        double *dvec;
    };
    
    // vector of fluxes
    float *q;
    
    // vector of species flags
    uint32_t *species_flags;
    
    // reset the species values based on the values specified in the species.
    void reset();
    const std::string str() const;

    float *item(const int &i);
    void setItem(const int &i, const float &val);

    MxStateVector();
    MxStateVector(MxSpeciesList *species, 
                  void *owner=NULL, 
                  MxStateVector *existingStateVector=NULL,
                  uint32_t flags=STATEVECTOR_NONE, 
                  void *data=NULL);
    MxStateVector(const MxStateVector &other);
    ~MxStateVector();

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation. 
     * 
     * @param str 
     * @return MxStateVector* 
     */
    static MxStateVector *fromString(const std::string &str);
};

namespace mx { namespace io { 

template <>
HRESULT toFile(const MxStateVector &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxStateVector **dataElement);

}};

#endif /* SRC_STATE_MXSTATEVECTOR_H_ */
