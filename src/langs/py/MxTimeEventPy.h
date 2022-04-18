/**
 * @file MxTimeEventPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxTimeEvent
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXTIMEEVENTPY_H_
#define _SRC_LANGS_PY_MXTIMEEVENTPY_H_

#include "MxPy.h"

#include "MxEventPyExecutor.h"

#include <event/MxTimeEvent.h>


struct MxTimeEventPy;

struct MxTimeEventPyPredicatePyExecutor : MxEventPyExecutor<MxTimeEventPy> {
    HRESULT _result = 0;
};

struct MxTimeEventPyInvokePyExecutor : MxEventPyExecutor<MxTimeEventPy> {
    HRESULT _result = 0;
};

struct CAPI_EXPORT MxTimeEventPy : MxEventBase {
    /**
     * @brief Next time of evaluation
     */
    double next_time;

    /**
     * @brief Period of evaluation
     */
    double period;

    /**
     * @brief Start time of evaluations
     */
    double start_time;

    /**
     * @brief End time of evaluations
     */
    double end_time;

    MxTimeEventPy(const double &period, 
                  MxTimeEventPyInvokePyExecutor *invokeExecutor, 
                  MxTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                  MxTimeEventNextTimeSetter *nextTimeSetter=NULL, 
                  const double &start_time=0, 
                  const double &end_time=-1) : 
        MxEventBase(), 
        period(period), 
        invokeExecutor(invokeExecutor), 
        predicateExecutor(predicateExecutor), 
        nextTimeSetter(nextTimeSetter), 
        next_time(start_time), 
        start_time(start_time), 
        end_time(end_time > 0 ? end_time : std::numeric_limits<double>::max())
    {}
    ~MxTimeEventPy();

    HRESULT predicate();
    HRESULT invoke();
    HRESULT eval(const double &time);

protected:

    double getNextTime(const double &current_time);

private:
    MxTimeEventPyInvokePyExecutor *invokeExecutor;
    MxTimeEventPyPredicatePyExecutor *predicateExecutor;
    MxTimeEventNextTimeSetter *nextTimeSetter;
};

using MxTimeEventPyList = MxEventListT<MxTimeEventPy>;

/**
 * @brief Creates a time-dependent event using prescribed invoke and predicate python function executors
 * 
 * @param period period of evaluations
 * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
 * @param predicateExecutor a predicate python function executor; evaluated to determine if an event occurs
 * @param distribution name of the function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @return MxTimeEvent* 
 */
CAPI_FUNC(MxTimeEventPy*) MxOnTimeEventPy(const double &period, 
                                          MxTimeEventPyInvokePyExecutor *invokeExecutor, 
                                          MxTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                                          const std::string &distribution="default", 
                                          const double &start_time=0, 
                                          const double &end_time=-1);

#endif // _SRC_LANGS_PY_MXTIMEEVENTPY_H_