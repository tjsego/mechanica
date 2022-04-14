/**
 * @file MxCEvent.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxEvent and associated features
 * @date 2022-04-05
 */

#ifndef _WRAPS_C_MXCEVENT_H_
#define _WRAPS_C_MXCEVENT_H_

#include <mx_port.h>

#include "MxCParticle.h"

typedef HRESULT (*MxEventMethodHandleFcn)(struct MxEventHandle*);
typedef HRESULT (*MxParticleEventMethodHandleFcn)(struct MxParticleEventHandle*);
typedef HRESULT (*MxTimeEventMethodHandleFcn)(struct MxTimeEventHandle*);
typedef HRESULT (*MxParticleTimeEventMethodHandleFcn)(struct MxParticleTimeEventHandle*);

// Handles

struct CAPI_EXPORT MxParticleEventParticleSelectorEnumHandle {
    unsigned int LARGEST; 
    unsigned int UNIFORM;
    unsigned int DEFAULT;
};

struct CAPI_EXPORT MxTimeEventTimeSetterEnumHandle {
    unsigned int DEFAULT;
    unsigned int DETERMINISTIC;
    unsigned int EXPONENTIAL;
};

struct CAPI_EXPORT MxParticleTimeEventParticleSelectorEnumHandle {
    unsigned int LARGEST; 
    unsigned int UNIFORM;
    unsigned int DEFAULT;
};

struct CAPI_EXPORT MxParticleTimeEventTimeSetterEnumHandle {
    unsigned int DETERMINISTIC;
    unsigned int EXPONENTIAL;
    unsigned int DEFAULT;
};

/**
 * @brief Handle to a @ref MxEvent instance
 * 
 */
struct CAPI_EXPORT MxEventHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxParticleEvent instance
 * 
 */
struct CAPI_EXPORT MxParticleEventHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxTimeEvent instance
 * 
 */
struct CAPI_EXPORT MxTimeEventHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxParticleTimeEvent instance
 * 
 */
struct CAPI_EXPORT MxParticleTimeEventHandle {
    void *MxObj;
};


/////////////////////////////////////////
// MxParticleEventParticleSelectorEnum //
/////////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleEventParticleSelectorEnum_init(struct MxParticleEventParticleSelectorEnumHandle *handle);


///////////////////////////////
// MxTimeEventTimeSetterEnum //
///////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEventTimeSetterEnum_init(struct MxTimeEventTimeSetterEnumHandle *handle);


/////////////////////////////////////////////
// MxParticleTimeEventParticleSelectorEnum //
/////////////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEventParticleSelectorEnum_init(struct MxParticleTimeEventParticleSelectorEnumHandle *handle);


///////////////////////////////////////
// MxParticleTimeEventTimeSetterEnum //
///////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEventTimeSetterEnum_init(struct MxParticleTimeEventTimeSetterEnumHandle *handle);


/////////////
// MxEvent //
/////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEvent_getLastFired(struct MxEventHandle *handle, double *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEvent_getTimesFired(struct MxEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEvent_remove(struct MxEventHandle *handle);


/////////////////////
// MxParticleEvent //
/////////////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleEvent_getLastFired(struct MxParticleEventHandle *handle, double *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleEvent_getTimesFired(struct MxParticleEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleEvent_remove(struct MxParticleEventHandle *handle);

/**
 * @brief Get the target particle type of this event
 * 
 * @param handle populated handle
 * @param targetType target particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleEvent_getTargetType(struct MxParticleEventHandle *handle, struct MxParticleTypeHandle *targetType);

/**
 * @brief Get the target particle of an event evaluation
 * 
 * @param handle populated handle
 * @param targetParticle target particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleEvent_getTargetParticle(struct MxParticleEventHandle *handle, struct MxParticleHandleHandle *targetParticle);


/////////////////
// MxTimeEvent //
/////////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEvent_getLastFired(struct MxTimeEventHandle *handle, double *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEvent_getTimesFired(struct MxTimeEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEvent_remove(struct MxTimeEventHandle *handle);

/**
 * @brief Get the next time of evaluation
 * 
 * @param handle populated handle
 * @param next_time next time of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEvent_getNextTime(struct MxTimeEventHandle *handle, double *next_time);

/**
 * @brief Get the period of evaluation
 * 
 * @param handle populated handle
 * @param period period of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEvent_getPeriod(struct MxTimeEventHandle *handle, double *period);

/**
 * @brief Get the start time of evaluations
 * 
 * @param handle populated handle
 * @param start_time start time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEvent_getStartTime(struct MxTimeEventHandle *handle, double *start_time);

/**
 * @brief Get the end time of evaluations
 * 
 * @param handle populated handle
 * @param end_time end time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCTimeEvent_getEndTime(struct MxTimeEventHandle *handle, double *end_time);


/////////////////////////
// MxParticleTimeEvent //
/////////////////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getLastFired(struct MxParticleTimeEventHandle *handle, double *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getTimesFired(struct MxParticleTimeEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_remove(struct MxParticleTimeEventHandle *handle);

/**
 * @brief Get the next time of evaluation
 * 
 * @param handle populated handle
 * @param next_time next time of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getNextTime(struct MxParticleTimeEventHandle *handle, double *next_time);

/**
 * @brief Get the period of evaluation
 * 
 * @param handle populated handle
 * @param period period of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getPeriod(struct MxParticleTimeEventHandle *handle, double *period);

/**
 * @brief Get the start time of evaluations
 * 
 * @param handle populated handle
 * @param start_time start time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getStartTime(struct MxParticleTimeEventHandle *handle, double *start_time);

/**
 * @brief Get the end time of evaluations
 * 
 * @param handle populated handle
 * @param end_time end time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getEndTime(struct MxParticleTimeEventHandle *handle, double *end_time);

/**
 * @brief Get the target particle type of this event
 * 
 * @param handle populated handle
 * @param targetType target particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getTargetType(struct MxParticleTimeEventHandle *handle, struct MxParticleTypeHandle *targetType);

/**
 * @brief Get the target particle of an event evaluation
 * 
 * @param handle populated handle
 * @param targetParticle target particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTimeEvent_getTargetParticle(struct MxParticleTimeEventHandle *handle, struct MxParticleHandleHandle *targetParticle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Creates an event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCOnEvent(struct MxEventHandle *handle, MxEventMethodHandleFcn *invokeMethod, MxEventMethodHandleFcn *predicateMethod);

/**
 * @brief Creates a particle event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param targetType target particle type
 * @param selectEnum particle selector enum
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCOnParticleEvent(struct MxParticleEventHandle *handle, 
                                      struct MxParticleTypeHandle *targetType, 
                                      unsigned int selectorEnum, 
                                      MxParticleEventMethodHandleFcn *invokeMethod, 
                                      MxParticleEventMethodHandleFcn *predicateMethod);

/**
 * @brief Creates a time-dependent event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param nextTimeSetterEnum enum selecting the function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCOnTimeEvent(struct MxTimeEventHandle *handle, 
                                  double period, 
                                  MxTimeEventMethodHandleFcn *invokeMethod, 
                                  MxTimeEventMethodHandleFcn *predicateMethod, 
                                  unsigned int nextTimeSetterEnum, 
                                  double start_time, 
                                  double end_time);

/**
 * @brief Creates a time-dependent particle event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param targetType target particle type
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param nextTimeSetterEnum enum of function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @param particleSelectorEnum enum of function that selects the next particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCOnParticleTimeEvent(struct MxParticleTimeEventHandle *handle, 
                                          struct MxParticleTypeHandle *targetType, 
                                          double period, 
                                          MxParticleTimeEventMethodHandleFcn *invokeMethod, 
                                          MxParticleTimeEventMethodHandleFcn *predicateMethod, 
                                          unsigned int nextTimeSetterEnum, 
                                          double start_time, 
                                          double end_time, 
                                          unsigned int particleSelectorEnum);

#endif // _WRAPS_C_MXCEVENT_H_