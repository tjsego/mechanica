/**
 * @file mx_parse.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines string parsing.
 * @date 2021-08-10
 * 
 */
#include "mx_parse.h"

#include <MxLogger.h>
#include <mx_error.h>

namespace mx { namespace parse {

void logMessageNoKwarg() {
    Log(LOG_ERROR) << "No kwarg found";
}

void logMessageNoKwargFound(const std::string &s) {
    Log(LOG_ERROR) << "kwarg found: " + s;
}

bool has_kwarg(const std::string &arg) {
    return arg.find("=") != std::string::npos;
}

bool has_kwarg(const std::string &arg, const std::string &kwarg) {
    return arg.find(kwarg + "=") != std::string::npos;
}

bool has_kwarg(const std::vector<std::string> &args, const std::string &kwarg) {
    for(auto &a : args)
        if(has_kwarg(a, kwarg)) 
            return true;

    return false;
}

std::string get_kwarg(const std::string &arg) {
    size_t idx = arg.find("=");
    if(idx == std::string::npos) {
        Log(LOG_ERROR) << "No kwarg found";
        return "";
    }
    
    std::string result(arg);
    result.replace(result.begin() + idx, result.begin() + arg.size(), "");
    return result;
}

std::string kwarg_findStr(const std::vector<std::string> &args, const std::string &kwarg) {
    for(auto &a : args)
        if(has_kwarg(a, kwarg))
            return a;

    return "";
}

std::string kwarg_strVal(const std::string &arg, const std::string &kwarg) {
    if(!has_kwarg(arg, kwarg)) {
        Log(LOG_ERROR) << kwarg + " not found.";
        return "";
    }

    std::string val(arg);
    val.replace(val.begin(), val.begin() + kwarg.size() + 1, "");
    return val;
}

std::pair<std::string, std::string> kwarg_getNameVal(const std::string &arg) {
    if(!has_kwarg(arg)) {
        Log(LOG_ERROR) << "No kwarg";
        return std::make_pair(std::string(), std::string());
    }
    std::string name = get_kwarg(arg);
    std::string val = kwarg_strVal(arg, name);
    return std::make_pair(name, val);
}

std::string kwargVal(const std::string &kwarg, const std::string &keyword) {
    if(!has_kwarg(kwarg, keyword)) {
        Log(LOG_ERROR) << keyword + " not found: " + kwarg;
        return "";
    }

    return kwarg_strVal(kwarg, keyword);
}

std::string kwargVal(const std::vector<std::string> &kwargs, const std::string &keyword) {
    for(auto &s : kwargs) {
        if(has_kwarg(s, keyword))
            return kwargVal(s, keyword);
    }

    Log(LOG_ERROR) << keyword + " not found.";
    return "";
}

bool has_mapKwarg(const std::string &arg, const std::string &kwarg) {
    return arg.find(kwarg + "={") != std::string::npos;
}

bool has_mapKwarg(const std::vector<std::string> &args, const std::string &kwarg) {
    for(auto &a : args)
        if(has_mapKwarg(a, kwarg)) 
            return true;

    return false;
}

std::string kwarg_findMapStr(const std::vector<std::string> &args, const std::string &kwarg) {
    for(auto &a : args)
        if(has_mapKwarg(a, kwarg))
            return a;

    return "";
}

std::string mapStrip(const std::string &arg) {
    std::string val(arg);
    val.replace(val.begin(), val.begin() + 1, "");
    return val.substr(0, val.size()-1);
}

std::string mapStrip(const std::string &arg, const std::string &kwarg) {
    std::string val(arg);
    val.replace(val.begin(), val.begin() + kwarg.size() + 2, "");
    return val.substr(0, val.size()-1);
}

std::vector<std::string> mapStrToStrVec(const std::string &s) {
    std::vector<std::string> result;

    // Handle embedded maps: every element without a terminal bracket absorbs its following element
    std::stringstream ss(s);
    std::string str;
    std::string storageStr = "";
    while(getline(ss, str, ';')) {
        storageStr += str;

        std::string name = get_kwarg(storageStr);
        if(has_mapKwarg(storageStr, name)) {
            if(*(storageStr.rbegin()) == *std::string("}").begin()) {
                result.push_back(storageStr);
                storageStr = "";
            }
            else storageStr += ";";
        }
        else {
            result.push_back(storageStr);
            storageStr = "";
        }
    }

    return result;
}

std::string mapVecToStr(const std::vector<std::string> &v) {
    std::string result;
    if(v.size() == 0) return result;

    result = "{" + v[0];
    if(v.size() == 1) return result + "}";

    for(int i = 1; i < v.size(); ++i) result += ";" + v[i];

    return result + "}";
}

std::string kwarg_strMapVal(const std::string &arg, const std::string &kwarg) {
    if(!has_mapKwarg(arg, kwarg)) {
        Log(LOG_ERROR) << kwarg + " not found.";
        return "";
    }

    std::string val(arg);
    val.replace(val.begin(), val.begin() + kwarg.size() + 2, "");
    return val.substr(0, val.size()-1);
}

std::string kwarg_strMapVal(const std::vector<std::string> &args, const std::string &kwarg) {
    return kwarg_strMapVal(mapVecToStr(args), kwarg);
}

std::pair<std::string, std::vector<std::string>> kwarg_getNameMapVals(const std::string &arg) {
    if(!has_kwarg(arg)) {
        Log(LOG_ERROR) << "No kwarg";
        return std::make_pair(std::string(), std::vector<std::string>());
    }
    std::string name = get_kwarg(arg);
    std::string val = kwarg_strMapVal(arg, name);
    return std::make_pair(name, mapStrToStrVec(val));
}

}}
