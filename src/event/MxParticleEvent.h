/**
 * @file MxParticleEvent.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic particle event
 * @date 2021-06-24
 * 
 */
#ifndef SRC_EVENT_MXPARTICLEEVENT_H_
#define SRC_EVENT_MXPARTICLEEVENT_H_

#include "MxEvent.h"
#include "MxEventList.h"

#include <unordered_map>

struct MxParticleHandle;
struct MxParticleType;

// Particle selector function template
template<typename event_t> using MxParticleEventParticleSelectorT = MxParticleHandle* (*)(const event_t&);

// MxParticleEvent

struct MxParticleEvent;

using MxParticleEventMethod = MxEventMethodT<MxParticleEvent>;

using MxParticleEventParticleSelector = MxParticleEventParticleSelectorT<MxParticleEvent>;

/**
 * @brief Selects a particle according to a uniform random distribution by event target type
 * 
 * @param typeId id of type
 * @param nr_parts number of particles of the type
 * @return MxParticleHandle* 
 */
MxParticleHandle *particleSelectorUniform(const int16_t &typeId, const int32_t &nr_parts);

/**
 * @brief Selects largest particle by event target type
 * 
 * @param typeId id of type
 * @return MxParticleHandle* 
 */
MxParticleHandle *particleSelectorLargest(const int16_t &typeId);

/**
 * @brief Selects a particle according to a uniform random distribution by event target type
 * 
 * @param event 
 * @return MxParticleHandle* 
 */
MxParticleHandle* MxParticleEventParticleSelectorUniform(const MxParticleEvent &event);

/**
 * @brief Selects largest particle by event target type
 * 
 * @param event 
 * @return MxParticleHandle* 
 */
MxParticleHandle* MxParticleEventParticleSelectorLargest(const MxParticleEvent &event);

// keys for selecting a particle selector
enum class MxParticleEventParticleSelectorEnum : unsigned int {
    LARGEST, 
    UNIFORM,
    DEFAULT
};

typedef std::unordered_map<MxParticleEventParticleSelectorEnum, MxParticleEventParticleSelector> MxParticleEventParticleSelectorMapType;
static MxParticleEventParticleSelectorMapType MxParticleEventParticleSelectorMap {
    {MxParticleEventParticleSelectorEnum::LARGEST, &MxParticleEventParticleSelectorLargest}, 
    {MxParticleEventParticleSelectorEnum::UNIFORM, &MxParticleEventParticleSelectorUniform}, 
    {MxParticleEventParticleSelectorEnum::DEFAULT, &MxParticleEventParticleSelectorUniform}
};

typedef std::unordered_map<std::string, MxParticleEventParticleSelectorEnum> MxParticleEventParticleSelectorNameMapType;
static MxParticleEventParticleSelectorNameMapType MxParticleEventParticleSelectorNameMap {
    {"largest", MxParticleEventParticleSelectorEnum::LARGEST}, 
    {"uniform", MxParticleEventParticleSelectorEnum::UNIFORM}, 
    {"default", MxParticleEventParticleSelectorEnum::DEFAULT}
};

/**
 * @brief Gets the particle selector on an event
 * 
 * @param selectorEnum selector enum
 * @return MxParticleEventParticleSelector* 
 */
CAPI_FUNC(MxParticleEventParticleSelector*) getMxParticleEventParticleSelector(MxParticleEventParticleSelectorEnum selectorEnum);

/**
 * @brief Gets the particle selector on an event
 * 
 * @param setterName name of selector
 * @return MxParticleEventParticleSelector* 
 */
CAPI_FUNC(MxParticleEventParticleSelector*) getMxParticleEventParticleSelectorN(std::string setterName);

// Particle event
struct CAPI_EXPORT MxParticleEvent : MxEventBase {

    /**
     * @brief Target particle type of this event
     */
    MxParticleType *targetType;

    /**
     * @brief Target particle of an event evaluation
     */
    MxParticleHandle *targetParticle;

    MxParticleEvent() {}
    MxParticleEvent(MxParticleType *targetType, 
                    MxParticleEventMethod *invokeMethod, 
                    MxParticleEventMethod *predicateMethod, 
                    MxParticleEventParticleSelector *particleSelector=NULL);
    virtual ~MxParticleEvent() {}

    virtual HRESULT predicate();
    virtual HRESULT invoke();
    virtual HRESULT eval(const double &time);

    HRESULT setMxParticleEventParticleSelector(MxParticleEventParticleSelector *particleSelector);
    HRESULT setMxParticleEventParticleSelector(MxParticleEventParticleSelectorEnum selectorEnum);
    HRESULT setMxParticleEventParticleSelector(std::string selectorName);

protected:

    MxParticleHandle *getNextParticle();

private:

    MxParticleEventMethod *invokeMethod;
    MxParticleEventMethod *predicateMethod;
    MxParticleEventParticleSelector *particleSelector;

};

// Event list for particle events
using MxParticleEventList = MxEventListT<MxParticleEvent>;

// Module entry points

/**
 * @brief Creates a particle event using prescribed invoke and predicate functions
 * 
 * @param targetType target particle type
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return MxParticleEvent* 
 */
CAPI_FUNC(MxParticleEvent*) MxOnParticleEvent(MxParticleType *targetType, 
                                              MxParticleEventMethod *invokeMethod, 
                                              MxParticleEventMethod *predicateMethod);

// python support

struct MxParticleEventPy;

struct MxParticleEventPyPredicatePyExecutor : MxEventPyExecutor<MxParticleEventPy> {
    HRESULT _result = 0;
};

struct MxParticleEventPyInvokePyExecutor : MxEventPyExecutor<MxParticleEventPy> {
    HRESULT _result = 0;
};

struct CAPI_EXPORT MxParticleEventPy : MxParticleEvent {

    MxParticleEventPy() {}
    MxParticleEventPy(MxParticleType *targetType, 
                      MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                      MxParticleEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                      MxParticleEventParticleSelector *particleSelector=NULL);
    virtual ~MxParticleEventPy();

    virtual HRESULT predicate();
    virtual HRESULT invoke();
    virtual HRESULT eval(const double &time);

private:
    
    MxParticleEventPyInvokePyExecutor *invokeExecutor;
    MxParticleEventPyPredicatePyExecutor *predicateExecutor;

};

/**
 * @brief Creates a particle event using prescribed invoke and predicate python function executors
 * 
 * @param targetType target particle type
 * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
 * @param predicateMethod a predicate python function executor; evaluated to determine if an event occurs
 * @param selector name of the function that selects the next particle
 * @return MxParticleEventPy* 
 */
CAPI_FUNC(MxParticleEventPy*) MxOnParticleEventPy(MxParticleType *targetType, 
                                                  MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                  MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                                  const std::string &selector="default");

#endif // SRC_EVENT_MXPARTICLEEVENT_H_