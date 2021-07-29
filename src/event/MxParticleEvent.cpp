/**
 * @file MxParticleEvent.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic particle event
 * @date 2021-06-24
 * 
 */

#include "MxParticleEvent.h"
#include <MxUtil.h>
#include "../mdcore/include/engine.h"
#include <MxLogger.h>

MxParticleHandle *particleSelectorUniform(const int16_t &typeId, const int32_t &nr_parts) {
    if(nr_parts == 0) {
        return NULL;
    }
    
    std::uniform_int_distribution<int> distribution(0, nr_parts-1);
    
    // index in the type's list of particles
    int tid = distribution(MxRandom);

    return _Engine.types[typeId].particle(tid)->py_particle();
}

MxParticleHandle *particleSelectorLargest(const int16_t &typeId) {
    auto ptype = &_Engine.types[typeId];

    if(ptype->parts.nr_parts == 0) return NULL;

    MxParticle *pLargest = ptype->particle(0);
    for(int i = 1; i < ptype->parts.nr_parts; ++i) {
        MxParticle *p = ptype->particle(i);
        if(p->nr_parts > pLargest->nr_parts) pLargest = p;
    }

    return pLargest->py_particle();
}

MxParticleHandle* MxParticleEventParticleSelectorUniform(const MxParticleEvent &event) {
    return particleSelectorUniform(event.targetType->id, event.targetType->parts.nr_parts);
}

MxParticleHandle* MxParticleEventParticleSelectorLargest(const MxParticleEvent &event) {
    return particleSelectorLargest(event.targetType->id);
}

MxParticleEventParticleSelector *getMxParticleEventParticleSelector(MxParticleEventParticleSelectorEnum selectorEnum) {
    auto x = MxParticleEventParticleSelectorMap.find(selectorEnum);
    if (x == MxParticleEventParticleSelectorMap.end()) return NULL;
    return &x->second;
}

MxParticleEventParticleSelector *getMxParticleEventParticleSelectorN(std::string setterName) {
    auto x = MxParticleEventParticleSelectorNameMap.find(setterName);
    if (x == MxParticleEventParticleSelectorNameMap.end()) return NULL;
    return getMxParticleEventParticleSelector(x->second);
}

MxParticleEvent::MxParticleEvent(MxParticleType *targetType, 
                                 MxParticleEventMethod *invokeMethod, 
                                 MxParticleEventMethod *predicateMethod, 
                                 MxParticleEventParticleSelector *particleSelector) : 
    MxEventBase(), 
    invokeMethod(invokeMethod), 
    predicateMethod(predicateMethod), 
    targetType(targetType), 
    particleSelector(particleSelector)
{
    if (particleSelector==NULL) setMxParticleEventParticleSelector(MxParticleEventParticleSelectorEnum::DEFAULT);
}

HRESULT MxParticleEvent::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);
    return 1;
}

HRESULT MxParticleEvent::invoke() {
    if(invokeMethod) (*invokeMethod)(*this);
    return 0;
}

HRESULT MxParticleEvent::eval(const double &time) {
    targetParticle = getNextParticle();
    return MxEventBase::eval(time);
}

HRESULT MxParticleEvent::setMxParticleEventParticleSelector(MxParticleEventParticleSelector *particleSelector) {
    this->particleSelector = particleSelector;
    return S_OK;
}

HRESULT MxParticleEvent::setMxParticleEventParticleSelector(MxParticleEventParticleSelectorEnum selectorEnum) {
    auto x = MxParticleEventParticleSelectorMap.find(selectorEnum);
    if (x == MxParticleEventParticleSelectorMap.end()) return 1;
    return setMxParticleEventParticleSelector(&x->second);
}

HRESULT MxParticleEvent::setMxParticleEventParticleSelector(std::string selectorName) {
    auto *particleSelector = getMxParticleEventParticleSelectorN(selectorName);
    if(!particleSelector) return E_FAIL;
    setMxParticleEventParticleSelector(particleSelector);
}

MxParticleHandle *MxParticleEvent::getNextParticle() {
    return (*particleSelector)(*this);
}

MxParticleEventSingle::MxParticleEventSingle(MxParticleType *targetType, 
                                             MxParticleEventMethod *invokeMethod, 
                                             MxParticleEventMethod *predicateMethod, 
                                             MxParticleEventParticleSelector *particleSelector) : 
    MxParticleEvent(targetType, invokeMethod, predicateMethod, particleSelector) 
{}

HRESULT MxParticleEventSingle::eval(const double &time) {
    remove();
    return MxParticleEvent::eval(time);
}

double MxParticleTimeEventSetNextTimeExponential(MxParticleTimeEventNew &event, const double &time) {
    return MxTimeEventSetNextTimeExponential(event, time);
}

double MxParticleTimeEventSetNextTimeDeterministic(MxParticleTimeEventNew &event, const double &time) {
    return MxTimeEventSetNextTimeDeterministic(event, time);
}

MxParticleHandle* MxParticleTimeEventParticleSelectorUniform(const MxParticleTimeEventNew &event) {
    return particleSelectorUniform(event.targetType->id, event.targetType->parts.nr_parts);
}

MxParticleHandle* MxParticleTimeEventParticleSelectorLargest(const MxParticleTimeEventNew &event) {
    return particleSelectorLargest(event.targetType->id);
}

MxParticleTimeEventParticleSelector *getMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum selectorEnum) {
    auto x = MxParticleTimeEventParticleSelectorMap.find(selectorEnum);
    if (x == MxParticleTimeEventParticleSelectorMap.end()) return NULL;
    return &x->second;
}

MxParticleTimeEventParticleSelector *getMxParticleTimeEventParticleSelectorN(std::string setterName) {
    auto x = MxParticleTimeEventParticleSelectorNameMap.find(setterName);
    if (x == MxParticleTimeEventParticleSelectorNameMap.end()) return NULL;
    return getMxParticleTimeEventParticleSelector(x->second);
}

MxParticleTimeEventNextTimeSetter* getMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum setterEnum) {
    auto x = MxParticleTimeEventNextTimeSetterMap.find(setterEnum);
    if (x == MxParticleTimeEventNextTimeSetterMap.end()) return NULL;
    return &x->second;
}

MxParticleTimeEventNextTimeSetter *getMxParticleTimeEventNextTimeSetterN(std::string setterName) {
    auto x = MxParticleTimeEventNextTimeSetterNameMap.find(setterName);
    if (x == MxParticleTimeEventNextTimeSetterNameMap.end()) return NULL;
    return getMxParticleTimeEventNextTimeSetter(x->second);
}

MxParticleTimeEventNew::MxParticleTimeEventNew(MxParticleType *targetType, 
                                               const double &period, 
                                               MxParticleTimeEventMethod *invokeMethod, 
                                               MxParticleTimeEventMethod *predicateMethod, 
                                               MxParticleTimeEventNextTimeSetter *nextTimeSetter, 
                                               const double &start_time, 
                                               const double &end_time,
                                               MxParticleTimeEventParticleSelector *particleSelector) : 
    MxEventBase(), 
    targetType(targetType), 
    invokeMethod(invokeMethod), 
    predicateMethod(predicateMethod), 
    period(period), 
    nextTimeSetter(nextTimeSetter),
    next_time(0),
    start_time(start_time),
    end_time(end_time > 0 ? end_time : std::numeric_limits<double>::max()), 
    particleSelector(particleSelector)
{
    if (nextTimeSetter == NULL) setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum::DEFAULT);
    if (particleSelector == NULL) setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum::DEFAULT);
}

HRESULT MxParticleTimeEventNew::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);

    return defaultMxTimeEventPredicateEval(this->next_time, this->start_time, this->end_time);
}

HRESULT MxParticleTimeEventNew::invoke() {
    if(invokeMethod) (*invokeMethod)(*this);
    return 0;
}

HRESULT MxParticleTimeEventNew::eval(const double &time) {
    targetParticle = getNextParticle();
    auto result = MxEventBase::eval(time);
    if(result) this->next_time = getNextTime(time);

    return result;
}

MxParticleTimeEventNew::operator MxTimeEvent&() const {
    MxTimeEvent e(period, NULL, NULL, NULL, start_time, end_time);
    return e;
}

double MxParticleTimeEventNew::getNextTime(const double &current_time) {
    return (*nextTimeSetter)(*this, current_time);
}

MxParticleHandle *MxParticleTimeEventNew::getNextParticle() {
    return (*particleSelector)(*this);
}

HRESULT MxParticleTimeEventNew::setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventNextTimeSetter *nextTimeSetter) {
    this->nextTimeSetter = nextTimeSetter;
    return S_OK;
}

HRESULT MxParticleTimeEventNew::setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum setterEnum) {
    auto x = MxParticleTimeEventNextTimeSetterMap.find(setterEnum);
    if (x == MxParticleTimeEventNextTimeSetterMap.end()) return 1;
    return setMxParticleTimeEventNextTimeSetter(&x->second);
}

HRESULT MxParticleTimeEventNew::setMxParticleTimeEventNextTimeSetter(std::string setterName) {
    auto *selector = getMxParticleTimeEventParticleSelectorN(setterName);
    if(!selector) return E_FAIL;
    return setMxParticleTimeEventParticleSelector(selector);
}

HRESULT MxParticleTimeEventNew::setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelector *particleSelector) {
    this->particleSelector = particleSelector;
    return S_OK;
}

HRESULT MxParticleTimeEventNew::setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum selectorEnum) {
    auto x = MxParticleTimeEventParticleSelectorMap.find(selectorEnum);
    if (x == MxParticleTimeEventParticleSelectorMap.end()) return 1;
    return setMxParticleTimeEventParticleSelector(&x->second);
}

HRESULT MxParticleTimeEventNew::setMxParticleTimeEventParticleSelector(std::string selectorName) {
    auto *selector = getMxParticleTimeEventParticleSelectorN(selectorName);
    if(!selector) return E_FAIL;
    return setMxParticleTimeEventParticleSelector(selector);
}

MxParticleEvent *MxOnParticleEvent(MxParticleType *targetType, 
                                   MxParticleEventMethod *invokeMethod, 
                                   MxParticleEventMethod *predicateMethod)
{
    MxParticleEvent *event = new MxParticleEvent(targetType, invokeMethod, predicateMethod);

    _Engine.events->addEvent(event);

    return event;
}

MxParticleEventSingle *MxOnParticleEventSingle(MxParticleType *targetType, 
                                               MxParticleEventMethod *invokeMethod, 
                                               MxParticleEventMethod *predicateMethod)
{
    MxParticleEventSingle *event = new MxParticleEventSingle(targetType, invokeMethod, predicateMethod);

    _Engine.events->addEvent(event);

    return event;
}

MxParticleTimeEventNew *MxOnParticleTimeEvent(MxParticleType *targetType, 
                                              const double &period, 
                                              MxParticleTimeEventMethod *invokeMethod, 
                                              MxParticleTimeEventMethod *predicateMethod, 
                                              unsigned int nextTimeSetterEnum, 
                                              const double &start_time, 
                                              const double &end_time, 
                                              unsigned int particleSelectorEnum)
{
    auto *nextTimeSetter = getMxParticleTimeEventNextTimeSetter((MxParticleTimeEventTimeSetterEnum)nextTimeSetterEnum);
    if(!nextTimeSetter) return NULL;

    auto *particleSelector = getMxParticleTimeEventParticleSelector((MxParticleTimeEventParticleSelectorEnum)particleSelectorEnum);
    if(!particleSelector) return NULL;
    
    MxParticleTimeEventNew *event = new MxParticleTimeEventNew(targetType, period, invokeMethod, predicateMethod, nextTimeSetter, start_time, end_time, particleSelector);

    _Engine.events->addEvent(event);

    return event;
}

MxParticleTimeEventNew *MxOnParticleTimeEventN(MxParticleType *targetType, 
                                               const double &period, 
                                               MxParticleTimeEventMethod *invokeMethod, 
                                               MxParticleTimeEventMethod *predicateMethod, 
                                               const std::string &distribution, 
                                               const double &start_time, 
                                               const double &end_time, 
                                               const std::string &selector) 
{
    auto x = MxParticleTimeEventNextTimeSetterNameMap.find(distribution);
    if (x == MxParticleTimeEventNextTimeSetterNameMap.end()) return NULL;
    unsigned int nextTimeSetterEnum = (unsigned) x->second;

    auto y = MxParticleEventParticleSelectorNameMap.find(selector);
    if (y == MxParticleEventParticleSelectorNameMap.end()) return NULL;
    unsigned int particleSelectorEnum = (unsigned) y->second;

    return MxOnParticleTimeEvent(targetType, period, invokeMethod, predicateMethod, nextTimeSetterEnum, start_time, end_time, particleSelectorEnum);
}

// python support

MxParticleEventPy::MxParticleEventPy(MxParticleType *targetType, 
                                     MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                     MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                     MxParticleEventParticleSelector *particleSelector) : 
    MxParticleEvent(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{
    this->targetType = targetType;
    if (particleSelector==NULL) setMxParticleEventParticleSelector(MxParticleEventParticleSelectorEnum::DEFAULT);
    else setMxParticleEventParticleSelector(particleSelector);
}

MxParticleEventPy::~MxParticleEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT MxParticleEventPy::predicate() {
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) return 1;
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT MxParticleEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT MxParticleEventPy::eval(const double &time) {
    targetParticle = getNextParticle();
    return MxEventBase::eval(time);
}

MxParticleSingleEventPy::MxParticleSingleEventPy(MxParticleType *targetType, 
                                                 MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                 MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                                 MxParticleEventParticleSelector *particleSelector) : 
    MxParticleEventPy(targetType, invokeExecutor, predicateExecutor, particleSelector)
{}

HRESULT MxParticleSingleEventPy::eval(const double &time) {
    remove();
    return MxParticleEventPy::eval(time);
}

MxParticleTimeEventPy::MxParticleTimeEventPy(MxParticleType *targetType, 
                                             const double &period, 
                                             MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                                             MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor, 
                                             MxParticleTimeEventNextTimeSetter *nextTimeSetter, 
                                             const double &start_time, 
                                             const double &end_time,
                                             MxParticleTimeEventParticleSelector *particleSelector) : 
    MxParticleTimeEventNew(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{
    this->targetType = targetType;
    this->period = period;
    this->next_time = 0;
    this->start_time = start_time;
    this->end_time = end_time > 0 ? end_time : std::numeric_limits<double>::max();

    if (nextTimeSetter == NULL) setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum::DEFAULT);
    else setMxParticleTimeEventNextTimeSetter(nextTimeSetter);

    if (particleSelector == NULL) setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum::DEFAULT);
    else setMxParticleTimeEventParticleSelector(particleSelector);
}

MxParticleTimeEventPy::~MxParticleTimeEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT MxParticleTimeEventPy::predicate() {
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) {
        return defaultMxTimeEventPredicateEval(this->next_time, this->start_time, this->end_time);
    }
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT MxParticleTimeEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT MxParticleTimeEventPy::eval(const double &time) {
    targetParticle = getNextParticle();
    auto result = MxEventBase::eval(time);
    if(result) this->next_time = getNextTime(time);

    return result;
}

MxParticleEventPy *MxOnParticleEventPy(MxParticleType *targetType, 
                                       MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                       MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                       const std::string &selector)
{
    Log(LOG_TRACE) << targetType->id;

    MxParticleEventParticleSelector *particleSelector = getMxParticleEventParticleSelectorN(selector);
    if (!particleSelector) return NULL;

    MxParticleEventPy *event = new MxParticleEventPy(targetType, invokeExecutor, predicateExecutor, particleSelector);

    _Engine.events->addEvent(event);

    return event;
}

MxParticleSingleEventPy *MxOnParticleEventSinglePy(MxParticleType *targetType, 
                                                   MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                   MxParticleEventPyPredicatePyExecutor *predicateExecutor)
{
    Log(LOG_TRACE) << targetType->id;

    MxParticleSingleEventPy *event = new MxParticleSingleEventPy(targetType, invokeExecutor, predicateExecutor);

    _Engine.events->addEvent(event);

    return event;
}

MxParticleTimeEventPy *MxOnParticleTimeEventPy(MxParticleType *targetType, 
                                               const double &period, 
                                               MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                                               MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor, 
                                               const std::string &distribution, 
                                               const double &start_time, 
                                               const double &end_time, 
                                               const std::string &selector)
{
    Log(LOG_TRACE) << targetType->id;
    
    MxParticleTimeEventNextTimeSetter *nextTimeSetter = getMxParticleTimeEventNextTimeSetterN(distribution);
    if (!nextTimeSetter) return NULL;

    MxParticleTimeEventParticleSelector *particleSelector = getMxParticleTimeEventParticleSelectorN(selector);
    if (!particleSelector) return NULL;
    
    MxParticleTimeEventPy *event = new MxParticleTimeEventPy(targetType, period, invokeExecutor, predicateExecutor, nextTimeSetter, start_time, end_time, particleSelector);

    _Engine.events->addEvent(event);

    return event;
}
