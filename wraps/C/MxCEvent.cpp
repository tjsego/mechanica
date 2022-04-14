/**
 * @file MxCEvent.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxEvent and associated features
 * @date 2022-04-05
 */

#include "MxCEvent.h"

#include "mechanica_c_private.h"

#include <event/MxEvent.h>
#include <event/MxParticleEvent.h>
#include <event/MxTimeEvent.h>
#include <event/MxParticleTimeEvent.h>


////////////////////////
// Function factories //
////////////////////////


// MxEventMethodHandleFcn

static MxEventMethodHandleFcn _MxEventMethod_factory_evalFcn;

HRESULT MxEventMethod_eval(const MxEvent &e) {
    MxEventHandle eHandle {(void*)&e};
    
    return (*_MxEventMethod_factory_evalFcn)(&eHandle);
}

MxEventMethod *MxEventMethod_factory(MxEventMethodHandleFcn &fcn) {
    _MxEventMethod_factory_evalFcn = fcn;
    return new MxEventMethod(MxEventMethod_eval);
}

// MxParticleEventMethodHandleFcn

static MxParticleEventMethodHandleFcn _MxParticleEventMethod_factory_evalFcn;

HRESULT MxParticleEventMethod_eval(const MxParticleEvent &e) {
    MxParticleEventHandle eHandle {(void*)&e};
    
    return (*_MxParticleEventMethod_factory_evalFcn)(&eHandle);
}

MxParticleEventMethod *MxParticleEventMethod_factory(MxParticleEventMethodHandleFcn fcn) {
    _MxParticleEventMethod_factory_evalFcn = fcn;
    return new MxParticleEventMethod(MxParticleEventMethod_eval);
}

// MxTimeEventMethodHandleFcn

static MxTimeEventMethodHandleFcn _MxTimeEventMethod_factory_evalFcn;

HRESULT MxTimeEventMethod_eval(const MxTimeEvent &e) {
    MxTimeEventHandle eHandle {(void*)&e};
    
    return (*_MxTimeEventMethod_factory_evalFcn)(&eHandle);
}

MxTimeEventMethod *MxTimeEventMethod_factory(MxTimeEventMethodHandleFcn fcn) {
    _MxTimeEventMethod_factory_evalFcn = fcn;
    return new MxTimeEventMethod(MxTimeEventMethod_eval);
}

// MxParticleTimeEventMethodHandleFcn

static MxParticleTimeEventMethodHandleFcn _MxParticleTimeEventMethod_factory_evalFcn;

HRESULT MxParticleTimeEventMethod_eval(const MxParticleTimeEvent &e) {
    MxParticleTimeEventHandle eHandle {(void*)&e};
    
    return (*_MxParticleTimeEventMethod_factory_evalFcn)(&eHandle);
}

MxParticleTimeEventMethod *MxParticleTimeEventMethod_factory(MxParticleTimeEventMethodHandleFcn fcn) {
    _MxParticleTimeEventMethod_factory_evalFcn = fcn;
    return new MxParticleTimeEventMethod(MxParticleTimeEventMethod_eval);
}


//////////////////
// Module casts //
//////////////////


namespace mx { 

MxEvent *castC(struct MxEventHandle *handle) {
    return castC<MxEvent, MxEventHandle>(handle);
}

MxParticleEvent *castC(struct MxParticleEventHandle *handle) {
    return castC<MxParticleEvent, MxParticleEventHandle>(handle);
}

MxTimeEvent *castC(struct MxTimeEventHandle *handle) {
    return castC<MxTimeEvent, MxTimeEventHandle>(handle);
}

MxParticleTimeEvent *castC(struct MxParticleTimeEventHandle *handle) {
    return castC<MxParticleTimeEvent, MxParticleTimeEventHandle>(handle);
}

}

#define MXEVENTHANDLE_GET(handle, varname) \
    MxEvent *varname = mx::castC<MxEvent, MxEventHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXPARTICLEEVENTHANDLE_GET(handle, varname) \
    MxParticleEvent *varname = mx::castC<MxParticleEvent, MxParticleEventHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXTIMEEVENTHANDLE_GET(handle, varname) \
    MxTimeEvent *varname = mx::castC<MxTimeEvent, MxTimeEventHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXPARTICLETIMEEVENTHANDLE_GET(handle, varname) \
    MxParticleTimeEvent *varname = mx::castC<MxParticleTimeEvent, MxParticleTimeEventHandle>(handle); \
    MXCPTRCHECK(varname);


/////////////////////////////////////////
// MxParticleEventParticleSelectorEnum //
/////////////////////////////////////////


HRESULT MxCParticleEventParticleSelectorEnum_init(struct MxParticleEventParticleSelectorEnumHandle *handle) {
    handle->LARGEST = (unsigned int)MxParticleEventParticleSelectorEnum::LARGEST; 
    handle->UNIFORM = (unsigned int)MxParticleEventParticleSelectorEnum::UNIFORM;
    handle->DEFAULT = (unsigned int)MxParticleEventParticleSelectorEnum::DEFAULT;
    return S_OK;
}


///////////////////////////////
// MxTimeEventTimeSetterEnum //
///////////////////////////////


HRESULT MxCTimeEventTimeSetterEnum_init(struct MxTimeEventTimeSetterEnumHandle *handle) { 
    handle->DEFAULT = (unsigned int)MxTimeEventTimeSetterEnum::DEFAULT;
    handle->DETERMINISTIC = (unsigned int)MxTimeEventTimeSetterEnum::DETERMINISTIC;
    handle->EXPONENTIAL = (unsigned int)MxTimeEventTimeSetterEnum::EXPONENTIAL;
    return S_OK;
}


/////////////////////////////////////////////
// MxParticleTimeEventParticleSelectorEnum //
/////////////////////////////////////////////


HRESULT MxCParticleTimeEventParticleSelectorEnum_init(struct MxParticleTimeEventParticleSelectorEnumHandle *handle) {
    handle->LARGEST = (unsigned int)MxParticleTimeEventParticleSelectorEnum::LARGEST; 
    handle->UNIFORM = (unsigned int)MxParticleTimeEventParticleSelectorEnum::UNIFORM;
    handle->DEFAULT = (unsigned int)MxParticleTimeEventParticleSelectorEnum::DEFAULT;
    return S_OK;
}


///////////////////////////////////////
// MxParticleTimeEventTimeSetterEnum //
///////////////////////////////////////


HRESULT MxCParticleTimeEventTimeSetterEnum_init(struct MxParticleTimeEventTimeSetterEnumHandle *handle) {
    handle->DETERMINISTIC = (unsigned int)MxParticleTimeEventTimeSetterEnum::DETERMINISTIC;
    handle->EXPONENTIAL = (unsigned int)MxParticleTimeEventTimeSetterEnum::EXPONENTIAL;
    handle->DEFAULT = (unsigned int)MxParticleTimeEventTimeSetterEnum::DEFAULT;
    return S_OK;
}


/////////////
// MxEvent //
/////////////


HRESULT MxCEvent_getLastFired(struct MxEventHandle *handle, double *last_fired) {
    MXEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT MxCEvent_getTimesFired(struct MxEventHandle *handle, unsigned int *times_fired) {
    MXEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT MxCEvent_remove(struct MxEventHandle *handle) {
    MXEVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}


/////////////////////
// MxParticleEvent //
/////////////////////


HRESULT MxCParticleEvent_getLastFired(struct MxParticleEventHandle *handle, double *last_fired) {
    MXPARTICLEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT MxCParticleEvent_getTimesFired(struct MxParticleEventHandle *handle, unsigned int *times_fired) {
    MXPARTICLEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT MxCParticleEvent_remove(struct MxParticleEventHandle *handle) {
    MXPARTICLEEVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}

HRESULT MxCParticleEvent_getTargetType(struct MxParticleEventHandle *handle, struct MxParticleTypeHandle *targetType) {
    MXPARTICLEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(targetType);
    MXCPTRCHECK(ev->targetType);
    targetType->MxObj = (void*)ev->targetType;
    return S_OK;
}

HRESULT MxCParticleEvent_getTargetParticle(struct MxParticleEventHandle *handle, struct MxParticleHandleHandle *targetParticle) {
    MXPARTICLEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(targetParticle);
    MXCPTRCHECK(ev->targetParticle);
    targetParticle->MxObj = (void*)ev->targetParticle;
    return S_OK;
}


/////////////////
// MxTimeEvent //
/////////////////


HRESULT MxCTimeEvent_getLastFired(struct MxTimeEventHandle *handle, double *last_fired) {
    MXTIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT MxCTimeEvent_getTimesFired(struct MxTimeEventHandle *handle, unsigned int *times_fired) {
    MXTIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT MxCTimeEvent_remove(struct MxTimeEventHandle *handle) {
    MXTIMEEVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}

HRESULT MxCTimeEvent_getNextTime(struct MxTimeEventHandle *handle, double *next_time) {
    MXTIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(next_time);
    *next_time = ev->next_time;
    return S_OK;
}

HRESULT MxCTimeEvent_getPeriod(struct MxTimeEventHandle *handle, double *period) {
    MXTIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(period);
    *period = ev->period;
    return S_OK;
}

HRESULT MxCTimeEvent_getStartTime(struct MxTimeEventHandle *handle, double *start_time) {
    MXTIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(start_time);
    *start_time = ev->start_time;
    return S_OK;
}

HRESULT MxCTimeEvent_getEndTime(struct MxTimeEventHandle *handle, double *end_time) {
    MXTIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(end_time);
    *end_time = ev->end_time;
    return S_OK;
}


/////////////////////////
// MxParticleTimeEvent //
/////////////////////////


HRESULT MxCParticleTimeEvent_getLastFired(struct MxParticleTimeEventHandle *handle, double *last_fired) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT MxCParticleTimeEvent_getTimesFired(struct MxParticleTimeEventHandle *handle, unsigned int *times_fired) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT MxCParticleTimeEvent_remove(struct MxParticleTimeEventHandle *handle) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}

HRESULT MxCParticleTimeEvent_getNextTime(struct MxParticleTimeEventHandle *handle, double *next_time) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(next_time);
    *next_time = ev->next_time;
    return S_OK;
}

HRESULT MxCParticleTimeEvent_getPeriod(struct MxParticleTimeEventHandle *handle, double *period) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(period);
    *period = ev->period;
    return S_OK;
}

HRESULT MxCParticleTimeEvent_getStartTime(struct MxParticleTimeEventHandle *handle, double *start_time) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(start_time);
    *start_time = ev->start_time;
    return S_OK;
}

HRESULT MxCParticleTimeEvent_getEndTime(struct MxParticleTimeEventHandle *handle, double *end_time) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(end_time);
    *end_time = ev->end_time;
    return S_OK;
}

HRESULT MxCParticleTimeEvent_getTargetType(struct MxParticleTimeEventHandle *handle, struct MxParticleTypeHandle *targetType) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(targetType);
    MXCPTRCHECK(ev->targetType);
    targetType->MxObj = (void*)ev->targetType;
    return S_OK;
}

HRESULT MxCParticleTimeEvent_getTargetParticle(struct MxParticleTimeEventHandle *handle, struct MxParticleHandleHandle *targetParticle) {
    MXPARTICLETIMEEVENTHANDLE_GET(handle, ev);
    MXCPTRCHECK(targetParticle);
    MXCPTRCHECK(ev->targetParticle);
    targetParticle->MxObj = (void*)ev->targetParticle;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCOnEvent(struct MxEventHandle *handle, MxEventMethodHandleFcn *invokeMethod, MxEventMethodHandleFcn *predicateMethod) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(invokeMethod);
    MXCPTRCHECK(predicateMethod);
    MxEvent *ev = MxOnEvent(MxEventMethod_factory(*invokeMethod), MxEventMethod_factory(*predicateMethod));
    MXCPTRCHECK(ev);
    handle->MxObj = (void*)ev;
    return S_OK;
}

HRESULT MxCOnParticleEvent(struct MxParticleEventHandle *handle, 
                           struct MxParticleTypeHandle *targetType, 
                           unsigned int selectorEnum, 
                           MxParticleEventMethodHandleFcn *invokeMethod, 
                           MxParticleEventMethodHandleFcn *predicateMethod) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(targetType); MXCPTRCHECK(targetType->MxObj);
    MXCPTRCHECK(invokeMethod);
    MXCPTRCHECK(predicateMethod);
    MxParticleEvent *ev = MxOnParticleEvent(
        (MxParticleType*)targetType->MxObj, MxParticleEventMethod_factory(*invokeMethod), MxParticleEventMethod_factory(*predicateMethod)
    );
    MXCPTRCHECK(ev);
    if(ev->setMxParticleEventParticleSelector((MxParticleEventParticleSelectorEnum)selectorEnum) != S_OK) 
        return E_FAIL;
    handle->MxObj = (void*)ev;
    return S_OK;
}

HRESULT MxCOnTimeEvent(struct MxTimeEventHandle *handle, 
                       double period, 
                       MxTimeEventMethodHandleFcn *invokeMethod, 
                       MxTimeEventMethodHandleFcn *predicateMethod, 
                       unsigned int nextTimeSetterEnum, 
                       double start_time, 
                       double end_time) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(invokeMethod);
    MXCPTRCHECK(predicateMethod);
    MxTimeEvent *ev = MxOnTimeEvent(
        period, MxTimeEventMethod_factory(*invokeMethod), MxTimeEventMethod_factory(*predicateMethod), nextTimeSetterEnum, start_time, end_time
    );
    MXCPTRCHECK(ev);
    handle->MxObj = (void*)ev;
    return S_OK;
}

HRESULT MxCOnParticleTimeEvent(struct MxParticleTimeEventHandle *handle, 
                               struct MxParticleTypeHandle *targetType, 
                               double period, 
                               MxParticleTimeEventMethodHandleFcn *invokeMethod, 
                               MxParticleTimeEventMethodHandleFcn *predicateMethod, 
                               unsigned int nextTimeSetterEnum, 
                               double start_time, 
                               double end_time, 
                               unsigned int particleSelectorEnum) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(targetType); MXCPTRCHECK(targetType->MxObj);
    MXCPTRCHECK(invokeMethod);
    MXCPTRCHECK(predicateMethod);
    MxParticleTimeEvent *ev = MxOnParticleTimeEvent(
        (MxParticleType*)targetType->MxObj, period, 
        MxParticleTimeEventMethod_factory(*invokeMethod), MxParticleTimeEventMethod_factory(*predicateMethod), 
        nextTimeSetterEnum, start_time, end_time, particleSelectorEnum
    );
    MXCPTRCHECK(ev);
    handle->MxObj = (void*)ev;
    return S_OK;
}
