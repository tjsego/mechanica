/**
 * @file MxParticleEvent.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic particle event
 * @date 2021-06-24
 * 
 */
#ifndef SRC_EVENT_MXPARTICLEEVENT_H_
#define SRC_EVENT_MXPARTICLEEVENT_H_

#include "MxTimeEvent.h"

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

    // Target particle type of this event
    MxParticleType *targetType;

    // Target particle of an event evaluation
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

// Single particle event
struct CAPI_EXPORT MxParticleEventSingle : MxParticleEvent {

    MxParticleEventSingle(MxParticleType *targetType, 
                          MxParticleEventMethod *invokeMethod, 
                          MxParticleEventMethod *predicateMethod, 
                          MxParticleEventParticleSelector *particleSelector=NULL);

    virtual HRESULT eval(const double &time);

};

// Event list for particle events
using MxParticleEventList = MxEventListT<MxParticleEvent>;

// MxParticleTimeEvent
// todo: migrate MxParticleTimeEvent to separate scripts

struct MxParticleTimeEvent;

using MxParticleTimeEventMethod = MxEventMethodT<MxParticleTimeEvent>;

using MxParticleTimeEventNextTimeSetter = double (*)(MxParticleTimeEvent&, const double&);

/**
 * @brief Sets the next time on an event according to an exponential distribution of the event period
 * 
 * @param event 
 * @param time 
 */
double MxParticleTimeEventSetNextTimeExponential(MxParticleTimeEvent &event, const double &time);

/**
 * @brief Sets the next time on an event according to the period of the event
 * 
 * @param event 
 * @param time 
 */
double MxParticleTimeEventSetNextTimeDeterministic(MxParticleTimeEvent &event, const double &time);

using MxParticleTimeEventParticleSelector = MxParticleEventParticleSelectorT<MxParticleTimeEvent>;

/**
 * @brief Selects a particle according to a uniform random distribution by event target type
 * 
 * @param event 
 * @return MxParticleHandle* 
 */
MxParticleHandle* MxParticleTimeEventParticleSelectorUniform(const MxParticleTimeEvent &event);

/**
 * @brief Selects largest particle by event target type
 * 
 * @param event 
 * @return MxParticleHandle* 
 */
MxParticleHandle* MxParticleTimeEventParticleSelectorLargest(const MxParticleTimeEvent &event);

// keys for selecting a particle selector
enum class MxParticleTimeEventParticleSelectorEnum : unsigned int {
    LARGEST, 
    UNIFORM,
    DEFAULT
};

typedef std::unordered_map<MxParticleTimeEventParticleSelectorEnum, MxParticleTimeEventParticleSelector> MxParticleTimeEventParticleSelectorMapType;
static MxParticleTimeEventParticleSelectorMapType MxParticleTimeEventParticleSelectorMap {
    {MxParticleTimeEventParticleSelectorEnum::LARGEST, &MxParticleTimeEventParticleSelectorLargest}, 
    {MxParticleTimeEventParticleSelectorEnum::UNIFORM, &MxParticleTimeEventParticleSelectorUniform}, 
    {MxParticleTimeEventParticleSelectorEnum::DEFAULT, &MxParticleTimeEventParticleSelectorUniform}
};

typedef std::unordered_map<std::string, MxParticleTimeEventParticleSelectorEnum> MxParticleTimeEventParticleSelectorNameMapType;
static MxParticleTimeEventParticleSelectorNameMapType MxParticleTimeEventParticleSelectorNameMap {
    {"largest", MxParticleTimeEventParticleSelectorEnum::LARGEST}, 
    {"uniform", MxParticleTimeEventParticleSelectorEnum::UNIFORM}, 
    {"default", MxParticleTimeEventParticleSelectorEnum::DEFAULT}
};

/**
 * @brief Gets the particle selector on an event
 * 
 * @param selectorEnum selector enum
 * @return MxParticleTimeEventParticleSelector* 
 */
CAPI_FUNC(MxParticleTimeEventParticleSelector*) getMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum selectorEnum);

/**
 * @brief Gets the particle selector on an event
 * 
 * @param setterName name of selector
 * @return MxParticleTimeEventParticleSelector* 
 */
CAPI_FUNC(MxParticleTimeEventParticleSelector*) getMxParticleTimeEventParticleSelectorN(std::string setterName);

/**
 * @brief keys for selecting a next time setter
 * 
 */
enum class MxParticleTimeEventTimeSetterEnum : unsigned int {
    DETERMINISTIC,
    EXPONENTIAL,
    DEFAULT
};

typedef std::unordered_map<MxParticleTimeEventTimeSetterEnum, MxParticleTimeEventNextTimeSetter> MxParticleTimeEventNextTimeSetterMapType;
static MxParticleTimeEventNextTimeSetterMapType MxParticleTimeEventNextTimeSetterMap {
    {MxParticleTimeEventTimeSetterEnum::DETERMINISTIC, &MxParticleTimeEventSetNextTimeDeterministic},
    {MxParticleTimeEventTimeSetterEnum::EXPONENTIAL, &MxParticleTimeEventSetNextTimeExponential},
    {MxParticleTimeEventTimeSetterEnum::DEFAULT, &MxParticleTimeEventSetNextTimeDeterministic}
};

typedef std::unordered_map<std::string, MxParticleTimeEventTimeSetterEnum> MxParticleTimeEventNextTimeSetterNameMapType;
static MxParticleTimeEventNextTimeSetterNameMapType MxParticleTimeEventNextTimeSetterNameMap {
    {"deterministic", MxParticleTimeEventTimeSetterEnum::DETERMINISTIC},
    {"exponential", MxParticleTimeEventTimeSetterEnum::EXPONENTIAL},
    {"default", MxParticleTimeEventTimeSetterEnum::DEFAULT}
};

/**
 * @brief Gets the next time on an event according to an exponential distribution of the event period
 * 
 * @param setterEnum setter enum
 * @return MxParticleTimeEventNextTimeSetter* 
 */
CAPI_FUNC(MxParticleTimeEventNextTimeSetter*) getMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum setterEnum);

/**
 * @brief Gets the next time on an event according to an exponential distribution of the event period
 * 
 * @param setterName name of setter
 * @return MxParticleTimeEventNextTimeSetter* 
 */
CAPI_FUNC(MxParticleTimeEventNextTimeSetter*) getMxParticleTimeEventNextTimeSetterN(std::string setterName);

// Time-dependent particle event
struct CAPI_EXPORT MxParticleTimeEvent : MxEventBase {

    // Target particle type of this event
    MxParticleType *targetType;

    // Target particle of an event evaluation
    MxParticleHandle *targetParticle;

    // Next time at which an evaluation occurs
    double next_time;

    // Period of event evaluations
    double period;

    // Time at which evaluations begin
    double start_time;

    // Time at which evaluations stop
    double end_time;
    
    MxParticleTimeEvent() {}
    MxParticleTimeEvent(MxParticleType *targetType, 
                        const double &period, 
                        MxParticleTimeEventMethod *invokeMethod, 
                        MxParticleTimeEventMethod *predicateMethod, 
                        MxParticleTimeEventNextTimeSetter *nextTimeSetter=NULL, 
                        const double &start_time=0, 
                        const double &end_time=-1,
                        MxParticleTimeEventParticleSelector *particleSelector=NULL);
    virtual ~MxParticleTimeEvent() {}

    virtual HRESULT predicate();
    virtual HRESULT invoke();
    virtual HRESULT eval(const double &time);

    operator MxTimeEvent&() const;

protected:

    double getNextTime(const double &current_time);
    MxParticleHandle *getNextParticle();

    HRESULT setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventNextTimeSetter *nextTimeSetter);
    HRESULT setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum setterEnum);
    HRESULT setMxParticleTimeEventNextTimeSetter(std::string setterName);

    HRESULT setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelector *particleSelector);
    HRESULT setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum selectorEnum);
    HRESULT setMxParticleTimeEventParticleSelector(std::string selectorName);

private:

    MxParticleTimeEventMethod *invokeMethod;
    MxParticleTimeEventMethod *predicateMethod;
    MxParticleTimeEventNextTimeSetter *nextTimeSetter;
    MxParticleTimeEventParticleSelector *particleSelector;

};

// Event list for time-dependent particle events
using MxParticleTimeEventList = MxEventListT<MxParticleTimeEvent>;

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

/**
 * @brief Creates a single particle event using prescribed invoke and predicate functions
 * 
 * @param targetType target particle type
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return MxParticleEventSingle* 
 */
CAPI_FUNC(MxParticleEventSingle*) MxOnParticleEventSingle(MxParticleType *targetType, 
                                                          MxParticleEventMethod *invokeMethod, 
                                                          MxParticleEventMethod *predicateMethod);

/**
 * @brief Creates a time-dependent particle event using prescribed invoke and predicate functions
 * 
 * @param targetType target particle type
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param nextTimeSetterEnum enum of function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @param particleSelectorEnum enum of function that selects the next particle
 * @return MxParticleTimeEvent* 
 */
CAPI_FUNC(MxParticleTimeEvent*) MxOnParticleTimeEvent(MxParticleType *targetType, 
                                                      const double &period, 
                                                      MxParticleTimeEventMethod *invokeMethod, 
                                                      MxParticleTimeEventMethod *predicateMethod=NULL, 
                                                      unsigned int nextTimeSetterEnum=0, 
                                                      const double &start_time=0, 
                                                      const double &end_time=-1, 
                                                      unsigned int particleSelectorEnum=0);

/**
 * @brief Creates a time-dependent particle event using prescribed invoke and predicate functions
 * 
 * @param targetType target particle type
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param distribution name of function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @param selector name of function that selects the next particle
 * @return MxParticleTimeEvent* 
 */
CAPI_FUNC(MxParticleTimeEvent*) MxOnParticleTimeEventN(MxParticleType *targetType, 
                                                       const double &period, 
                                                       MxParticleTimeEventMethod *invokeMethod, 
                                                       MxParticleTimeEventMethod *predicateMethod=NULL, 
                                                       const std::string &distribution="default", 
                                                       const double &start_time=0, 
                                                       const double &end_time=-1, 
                                                       const std::string &selector="default");

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

// Single particle event
struct CAPI_EXPORT MxParticleSingleEventPy : MxParticleEventPy {
    MxParticleSingleEventPy(MxParticleType *targetType, 
                            MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                            MxParticleEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                            MxParticleEventParticleSelector *particleSelector=NULL);

    virtual HRESULT eval(const double &time);
};

struct MxParticleTimeEventPy;

struct MxParticleTimeEventPyPredicatePyExecutor : MxEventPyExecutor<MxParticleTimeEventPy> {
    HRESULT _result = 0;
};

struct MxParticleTimeEventPyInvokePyExecutor : MxEventPyExecutor<MxParticleTimeEventPy> {
    HRESULT _result = 0;
};

// Time-dependent particle event
struct CAPI_EXPORT MxParticleTimeEventPy : MxParticleTimeEvent {
    
    MxParticleTimeEventPy() {}
    MxParticleTimeEventPy(MxParticleType *targetType, 
                          const double &period, 
                          MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                          MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                          MxParticleTimeEventNextTimeSetter *nextTimeSetter=NULL, 
                          const double &start_time=0, 
                          const double &end_time=-1,
                          MxParticleTimeEventParticleSelector *particleSelector=NULL);
    virtual ~MxParticleTimeEventPy();

    virtual HRESULT predicate();
    virtual HRESULT invoke();
    virtual HRESULT eval(const double &time);

private:

    MxParticleTimeEventPyInvokePyExecutor *invokeExecutor;
    MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor;

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

/**
 * @brief Creates a single particle event using prescribed invoke and predicate python function executors
 * 
 * @param targetType target particle type
 * @param invokeMethod an invoke python function executor; evaluated when an event occurs
 * @param predicateMethod a predicate python function executor; evaluated to determine if an event occurs
 * @return MxParticleEventSinglePy* 
 */
CAPI_FUNC(MxParticleSingleEventPy*) MxOnParticleEventSinglePy(MxParticleType *targetType, 
                                                              MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                              MxParticleEventPyPredicatePyExecutor *predicateExecutor);

/**
 * @brief Creates a time-dependent particle event using prescribed invoke and predicate python function executors
 * 
 * @param targetType target particle type
 * @param period period of evaluations
 * @param invokeMethod an invoke python function executor; evaluated when an event occurs
 * @param predicateMethod a predicate python function executor; evaluated to determine if an event occurs
 * @param distribution name of the function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @param selector name of the function that selects the next particle
 * @return MxParticleTimeEvent* 
 */
CAPI_FUNC(MxParticleTimeEventPy*) MxOnParticleTimeEventPy(MxParticleType *targetType, 
                                                          const double &period, 
                                                          MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                                                          MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                                                          const std::string &distribution="default", 
                                                          const double &start_time=0.0, 
                                                          const double &end_time=-1.0, 
                                                          const std::string &selector="default");

#endif // SRC_EVENT_MXPARTICLEEVENT_H_