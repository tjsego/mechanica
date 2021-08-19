/**
 * @file MxParticleTimeEvent.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic time-dependent particle event
 * @date 2021-08-19
 * 
 */
#ifndef SRC_EVENT_MXPARTICLETIMEEVENT_H_
#define SRC_EVENT_MXPARTICLETIMEEVENT_H_

#include "MxTimeEvent.h"
#include "MxParticleEvent.h"

struct MxParticleType;
struct MxParticleEvent;

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

#endif // SRC_EVENT_MXPARTICLETIMEEVENT_H_