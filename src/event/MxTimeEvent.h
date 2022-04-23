/**
 * @file MxTimeEvent.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines time-dependent event
 * @date 2021-06-23
 * 
 */
#ifndef SRC_EVENT_MXTIMEEVENT_H_
#define SRC_EVENT_MXTIMEEVENT_H_

#include "MxEventList.h"

#include <limits>
#include <unordered_map>

struct MxTimeEvent;

using MxTimeEventMethod = MxEventMethodT<MxTimeEvent>;
using MxTimeEventNextTimeSetter = double (*)(MxTimeEvent&, const double&);

HRESULT defaultMxTimeEventPredicateEval(const double &next_time, const double &start_time=-1, const double &end_time=-1);

double MxTimeEventSetNextTimeExponential(MxTimeEvent &event, const double &time);
double MxTimeEventSetNextTimeDeterministic(MxTimeEvent &event, const double &time);

enum class MxTimeEventTimeSetterEnum : unsigned int {
    DEFAULT = 0, 
    DETERMINISTIC,
    EXPONENTIAL
};

typedef std::unordered_map<MxTimeEventTimeSetterEnum, MxTimeEventNextTimeSetter> MxTimeEventNextTimeSetterMapType;
static MxTimeEventNextTimeSetterMapType MxTimeEventNextTimeSetterMap {
    {MxTimeEventTimeSetterEnum::DETERMINISTIC, &MxTimeEventSetNextTimeDeterministic},
    {MxTimeEventTimeSetterEnum::EXPONENTIAL, &MxTimeEventSetNextTimeExponential},
    {MxTimeEventTimeSetterEnum::DEFAULT, &MxTimeEventSetNextTimeDeterministic}
};

typedef std::unordered_map<std::string, MxTimeEventTimeSetterEnum> MxTimeEventNextTimeSetterNameMapType;
static MxTimeEventNextTimeSetterNameMapType MxTimeEventNextTimeSetterNameMap {
    {"deterministic", MxTimeEventTimeSetterEnum::DETERMINISTIC},
    {"exponential", MxTimeEventTimeSetterEnum::EXPONENTIAL},
    {"default", MxTimeEventTimeSetterEnum::DEFAULT}
};

CAPI_FUNC(MxTimeEventNextTimeSetter*) getMxTimeEventNextTimeSetter(MxTimeEventTimeSetterEnum setterEnum);
CAPI_FUNC(MxTimeEventNextTimeSetter*) getMxTimeEventNextTimeSetterN(std::string setterName);

struct CAPI_EXPORT MxTimeEvent : MxEventBase {
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

    MxTimeEvent(const double &period, 
                MxTimeEventMethod *invokeMethod, 
                MxTimeEventMethod *predicateMethod=NULL, 
                MxTimeEventNextTimeSetter *nextTimeSetter=NULL, 
                const double &start_time=0, 
                const double &end_time=-1) : 
        MxEventBase(), 
        period(period), 
        invokeMethod(invokeMethod), 
        predicateMethod(predicateMethod), 
        nextTimeSetter(nextTimeSetter),
        next_time(start_time),
        start_time(start_time),
        end_time(end_time > 0 ? end_time : std::numeric_limits<double>::max())
    {
        if (nextTimeSetter == NULL) setMxTimeEventNextTimeSetter(MxTimeEventTimeSetterEnum::DEFAULT);
    }
    ~MxTimeEvent();

    HRESULT predicate();
    HRESULT invoke();
    HRESULT eval(const double &time);

protected:

    double getNextTime(const double &current_time);
    HRESULT setMxTimeEventNextTimeSetter(MxTimeEventTimeSetterEnum setterEnum);

private:

    MxTimeEventMethod *invokeMethod;
    MxTimeEventMethod *predicateMethod;
    MxTimeEventNextTimeSetter *nextTimeSetter;
};

using MxTimeEventList = MxEventListT<MxTimeEvent>;


// Module entry points

/**
 * @brief Creates a time-dependent event using prescribed invoke and predicate functions
 * 
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param nextTimeSetterEnum enum selecting the function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @return MxTimeEvent* 
 */
CAPI_FUNC(MxTimeEvent*) MxOnTimeEvent(const double &period, 
                                      MxTimeEventMethod *invokeMethod, 
                                      MxTimeEventMethod *predicateMethod=NULL, 
                                      const unsigned int &nextTimeSetterEnum=0, 
                                      const double &start_time=0, 
                                      const double &end_time=-1);

/**
 * @brief Creates a time-dependent event using prescribed invoke and predicate functions
 * 
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param distribution name of the function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @return MxTimeEvent* 
 */
CAPI_FUNC(MxTimeEvent*) MxOnTimeEventN(const double &period, 
                                       MxTimeEventMethod *invokeMethod, 
                                       MxTimeEventMethod *predicateMethod=NULL, 
                                       const std::string &distribution="default", 
                                       const double &start_time=0, 
                                       const double &end_time=-1);

#endif // SRC_EVENT_MXTIMEEVENT_H_