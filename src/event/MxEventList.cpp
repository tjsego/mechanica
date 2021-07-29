/**
 * @file MxEventList.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic event list operations
 * @date 2021-06-24
 * 
 */

#include "MxEventList.h"

HRESULT event_func_invoke(MxEventBase &event, const double &time) {
    return event.eval(time);
}

std::vector<MxEventBase*>::iterator MxEventBaseList::findEventIterator(MxEventBase *event) {
    for(std::vector<MxEventBase*>::iterator itr = events.begin(); itr != events.end(); ++itr)
        if(*itr == event) 
            return itr;
    return events.end();
}

MxEventBaseList::~MxEventBaseList() {
    events.clear();
    toRemove.clear();
}

void MxEventBaseList::addEvent(MxEventBase *event) { events.push_back(event); }

HRESULT MxEventBaseList::removeEvent(MxEventBase *event) {
    auto itr = findEventIterator(event);
    if (itr == events.end()) return 1;
    delete *itr;
    events.erase(itr);
    return S_OK;
}

HRESULT MxEventBaseList::eval(const double &time) {
    for (auto &e : events) {
        auto result = e->eval(time);
        if (result != 0) return result;

        for (auto flag : e->flags) {

            switch (flag)
            {
            case MxEventFlag::REMOVE:
                toRemove.push_back(e);
                break;
            
            default:
                break;
            }

        }
    }

    for (auto e : toRemove)
        removeEvent(e);

    toRemove.clear();

    return S_OK;
}

HRESULT MxEventListEval(MxEventBaseList *eventList, const double &time) {
    return eventList->eval(time);
}
