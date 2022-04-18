/**
 * @file MxParticleTimeEvent.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic time-dependent particle event
 * @date 2021-08-19
 * 
 */
#include "MxParticleTimeEvent.h"

#include <MxUtil.h>
#include <MxUniverse.h>
#include <MxLogger.h>

double MxParticleTimeEventSetNextTimeExponential(MxParticleTimeEvent &event, const double &time) {
    return MxTimeEventSetNextTimeExponential(event, time);
}

double MxParticleTimeEventSetNextTimeDeterministic(MxParticleTimeEvent &event, const double &time) {
    return MxTimeEventSetNextTimeDeterministic(event, time);
}

MxParticleHandle* MxParticleTimeEventParticleSelectorUniform(const MxParticleTimeEvent &event) {
    return particleSelectorUniform(event.targetType->id, event.targetType->parts.nr_parts);
}

MxParticleHandle* MxParticleTimeEventParticleSelectorLargest(const MxParticleTimeEvent &event) {
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

MxParticleTimeEvent::MxParticleTimeEvent(MxParticleType *targetType, 
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

HRESULT MxParticleTimeEvent::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);

    return defaultMxTimeEventPredicateEval(this->next_time, this->start_time, this->end_time);
}

HRESULT MxParticleTimeEvent::invoke() {
    if(invokeMethod) (*invokeMethod)(*this);
    return 0;
}

HRESULT MxParticleTimeEvent::eval(const double &time) {
    targetParticle = getNextParticle();
    auto result = MxEventBase::eval(time);
    if(result) this->next_time = getNextTime(time);

    return result;
}

MxParticleTimeEvent::operator MxTimeEvent&() const {
    MxTimeEvent *e = new MxTimeEvent(period, NULL, NULL, NULL, start_time, end_time);
    return *e;
}

double MxParticleTimeEvent::getNextTime(const double &current_time) {
    return (*nextTimeSetter)(*this, current_time);
}

MxParticleHandle *MxParticleTimeEvent::getNextParticle() {
    return (*particleSelector)(*this);
}

HRESULT MxParticleTimeEvent::setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventNextTimeSetter *nextTimeSetter) {
    this->nextTimeSetter = nextTimeSetter;
    return S_OK;
}

HRESULT MxParticleTimeEvent::setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum setterEnum) {
    auto x = MxParticleTimeEventNextTimeSetterMap.find(setterEnum);
    if (x == MxParticleTimeEventNextTimeSetterMap.end()) return 1;
    return setMxParticleTimeEventNextTimeSetter(&x->second);
}

HRESULT MxParticleTimeEvent::setMxParticleTimeEventNextTimeSetter(std::string setterName) {
    auto *selector = getMxParticleTimeEventParticleSelectorN(setterName);
    if(!selector) return E_FAIL;
    return setMxParticleTimeEventParticleSelector(selector);
}

HRESULT MxParticleTimeEvent::setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelector *particleSelector) {
    this->particleSelector = particleSelector;
    return S_OK;
}

HRESULT MxParticleTimeEvent::setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum selectorEnum) {
    auto x = MxParticleTimeEventParticleSelectorMap.find(selectorEnum);
    if (x == MxParticleTimeEventParticleSelectorMap.end()) return 1;
    return setMxParticleTimeEventParticleSelector(&x->second);
}

HRESULT MxParticleTimeEvent::setMxParticleTimeEventParticleSelector(std::string selectorName) {
    auto *selector = getMxParticleTimeEventParticleSelectorN(selectorName);
    if(!selector) return E_FAIL;
    return setMxParticleTimeEventParticleSelector(selector);
}

MxParticleTimeEvent *MxOnParticleTimeEvent(MxParticleType *targetType, 
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
    
    MxParticleTimeEvent *event = new MxParticleTimeEvent(targetType, period, invokeMethod, predicateMethod, nextTimeSetter, start_time, end_time, particleSelector);

    MxUniverse::get()->events->addEvent(event);

    return event;
}

MxParticleTimeEvent *MxOnParticleTimeEventN(MxParticleType *targetType, 
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
