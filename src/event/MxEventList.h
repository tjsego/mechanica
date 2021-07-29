/**
 * @file MxEventList.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic event list operations
 * @date 2021-06-24
 * 
 */
#ifndef SRC_EVENT_MXEVENTLIST_H_
#define SRC_EVENT_MXEVENTLIST_H_

#include "MxEvent.h"

HRESULT event_func_invoke(MxEventBase &event, const double &time);

struct CAPI_EXPORT MxEventBaseList {

private:

    inline std::vector<MxEventBase*>::iterator findEventIterator(MxEventBase *event);
    std::vector<MxEventBase*> toRemove;

public:

    std::vector<MxEventBase*> events;

    ~MxEventBaseList();

    inline void addEvent(MxEventBase *event);
    inline HRESULT removeEvent(MxEventBase *event);
    HRESULT eval(const double &time);

};

inline HRESULT MxEventListEval(MxEventBaseList *eventList, const double &time);

// Basic event list
template<typename event_t> 
struct MxEventListT : MxEventBaseList {};

using MxEventList = MxEventListT<MxEvent>;

#endif // SRC_EVENT_MXEVENTLIST_H_