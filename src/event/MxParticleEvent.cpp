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
#include <MxUniverse.h>

MxParticleHandle *particleSelectorUniform(const int16_t &typeId, const int32_t &nr_parts) {
    if(nr_parts == 0) {
        return NULL;
    }
    
    std::uniform_int_distribution<int> distribution(0, nr_parts-1);
    auto MxRandom = MxRandomEngine();
    
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
    return setMxParticleEventParticleSelector(particleSelector);
}

MxParticleHandle *MxParticleEvent::getNextParticle() {
    return (*particleSelector)(*this);
}

MxParticleEvent *MxOnParticleEvent(MxParticleType *targetType, 
                                   MxParticleEventMethod *invokeMethod, 
                                   MxParticleEventMethod *predicateMethod)
{
    MxParticleEvent *event = new MxParticleEvent(targetType, invokeMethod, predicateMethod);

    MxUniverse::get()->events->addEvent(event);

    return event;
}
