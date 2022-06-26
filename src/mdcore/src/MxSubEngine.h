/**
 * @file MxSubEngine.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines interface for solvers that can be injected into the Mechanica engine
 * @date 2022-03-15
 * 
 */
#ifndef SRC_MDCORE_SRC_MXSUBENGINE_H_
#define SRC_MDCORE_SRC_MXSUBENGINE_H_

#include <mx_port.h>

/**
 * @brief A MxSubEngine is a singleton object that injects dynamics into the Mechanica engine. 
 * It does not necessarily integrate any object in time, but can also 
 * simply add to the dynamics of existing Mechanica objects. 
 * Mechanica supports an arbitrary number of subengines with multi-threading and GPU support. 
 * 
 */
struct MxSubEngine {

    /** Unique name of the engine. No two registered engines can have the same name. */
    const char* name;

    /**
     * @brief Register with the Mechanica engine.
     * 
     * @return HRESULT 
     */
    HRESULT registerEngine();

    /**
     * @brief First call before forces are calculated for a step. 
     * 
     * @return HRESULT 
     */
    virtual HRESULT preStepStart() { return S_OK; };

    /**
     * @brief Last call before forces are calculated for a step.
     * 
     * @return HRESULT 
     */
    virtual HRESULT preStepJoin() { return S_OK; };

    /**
     * @brief First call after forces are calculated, and before events are executed, for a step.
     * 
     * @return HRESULT 
     */
    virtual HRESULT postStepStart() { return S_OK; };

    /**
     * @brief Last call after forces are calculated, and before events are executed, for a step.
     * 
     * @return HRESULT 
     */
    virtual HRESULT postStepJoin() { return S_OK; };

    /**
     * @brief Called during termination of a simulation, just before shutdown of Mechanica engine.
     * 
     * @return HRESULT 
     */
    virtual HRESULT finalize() { return S_OK; };

};

#endif // SRC_MDCORE_SRC_MXSUBENGINE_H_