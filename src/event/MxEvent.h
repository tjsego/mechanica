/**
 * @file MxEvent.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic event; all data relevant to an event will be stored in the data of the event
 * @date 2021-06-23
 * 
 */
#ifndef SRC_EVENT_MXEVENT_H_
#define SRC_EVENT_MXEVENT_H_

#include "../../mechanica_private.h"
#include "MxEventPyExecutor.h"

#include <forward_list>
#include <iostream>

template<typename event_t> using MxEventMethodT = HRESULT (*)(const event_t&);

// Flags for providing feedback during predicate and invoke evaluations
enum class MxEventFlag : unsigned int {
    REMOVE
};

// Base class of all events
struct MxEventBase {

    // Flags set by invoke and predicate to provide feedback
    std::forward_list<MxEventFlag> flags;

    // Record of last time fired
    double last_fired;

    // Record of how many times fired
    int times_fired;

    /**
     * Evaluates an event predicate,
     * returns 0 if the event should not fire, 1 if the event should, and a
     * negative value on error.
     * A predicate without a defined predicate method always returns 0
     */
    virtual HRESULT predicate() = 0;

    /**
     * What occurs during an event. 
     * Typically, this invokes an underlying specialized method
     * returns 0 if OK and 1 on error.
     */
    virtual HRESULT invoke() = 0;

    MxEventBase() : 
        last_fired(0.0), 
        times_fired(0)
    {}
    virtual ~MxEventBase() {}

    // Tests the predicate and evaluates invoke accordingly
    // Returns 1 if the event was invoked, 0 if not, and a negative value on error
    virtual HRESULT eval(const double &time) {
        // check predicate
        if(!predicate()) return 0;

        // invoke
        if(invoke() == 1) return -1;

        // Update internal data
        times_fired += 1;
        last_fired = time;

        return 1;
    }

    // Designates event for removal
    void remove() { flags.push_front(MxEventFlag::REMOVE); }

};

struct MxEvent;

using MxEventMethod = MxEventMethodT<MxEvent>;

// Simple event
struct CAPI_EXPORT MxEvent : MxEventBase {
    
    MxEvent();

    /**
     * @brief Construct a new Mx Event object using functions
     * 
     * @param invokeMethod an invoke function
     * @param predicateMethod a predicate function
     */
    MxEvent(MxEventMethod *invokeMethod, MxEventMethod *predicateMethod);
    virtual ~MxEvent();

    HRESULT predicate();
    HRESULT invoke();
    HRESULT eval(const double &time);

private:

    MxEventMethod *invokeMethod;
    MxEventMethod *predicateMethod;

};

/**
 * @brief Creates an event using prescribed invoke and predicate functions
 * 
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return MxEvent* 
 */
CAPI_FUNC(MxEvent*) MxOnEvent(MxEventMethod *invokeMethod, MxEventMethod *predicateMethod);


// python support

struct MxEventPy;

struct MxEventPyPredicatePyExecutor : MxEventPyExecutor<MxEventPy> {
    HRESULT _result = 0;
};

struct MxEventPyInvokePyExecutor : MxEventPyExecutor<MxEventPy> {
    HRESULT _result = 0;
};

struct CAPI_EXPORT MxEventPy : MxEventBase {
    MxEventPy(MxEventPyInvokePyExecutor *invokeExecutor, MxEventPyPredicatePyExecutor *predicateExecutor=NULL);
    ~MxEventPy();

    HRESULT predicate();
    HRESULT invoke();
    HRESULT eval(const double &time);

private:
    MxEventPyInvokePyExecutor *invokeExecutor; 
    MxEventPyPredicatePyExecutor *predicateExecutor;
};

/**
 * @brief Creates an event using prescribed invoke and predicate python function executors
 * 
 * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
 * @param predicateExecutor a predicate python function executor; evaluated to determine if an event occurs
 * @return MxEventPy* 
 */
CAPI_FUNC(MxEventPy*) MxOnEventPy(MxEventPyInvokePyExecutor *invokeExecutor, MxEventPyPredicatePyExecutor *predicateExecutor=NULL);


#endif // SRC_EVENT_MXEVENT_H_