/**
 * @file mx_cast.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines cast for basic types. cast may also be defined elsewhere for other types. 
 * cast is not intended to cover all possible type conversions, 
 * but for conversions relevant to the scope of various Mechanica functionality. 
 * @date 2021-08-09
 * 
 */
#include "mx_cast.h"

namespace mx {

template<>
std::string cast(const int &t) {
    return std::to_string(t);
}

template<>
std::string cast(const long &t) {
    return std::to_string(t);
}

template<>
std::string cast(const long long &t) {
    return std::to_string(t);
}

template<>
std::string cast(const unsigned int &t) {
    return std::to_string(t);
}

template<>
std::string cast(const unsigned long &t) {
    return std::to_string(t);
}

template<>
std::string cast(const unsigned long long &t) {
    return std::to_string(t);
}

template<> 
std::string cast(const bool &t) {
    return std::to_string(t);
}

template<>
std::string cast(const float &t) {
    return std::to_string(t);
}

template<>
std::string cast(const double &t) {
    return std::to_string(t);
}

template<>
std::string cast(const long double &t) {
    return std::to_string(t);
}

template<>
int cast(const std::string &s) {
    return std::stoi(s);
}

template<>
long cast(const std::string &s) {
    return std::stol(s);
}

template<>
long long cast(const std::string &s) {
    return std::stoll(s);
}

template<>
unsigned int cast(const std::string &s) {
    return std::stoul(s);
}

template<>
unsigned long cast(const std::string &s) {
    return std::stoul(s);
}

template<>
unsigned long long cast(const std::string &s) {
    return std::stoull(s);
}

template<>
bool cast(const std::string &s) {
    return (bool)cast<std::string, int>(s);
}

template<>
float cast(const std::string &s) {
    return std::stof(s);
}

template<>
double cast(const std::string &s) {
    return std::stod(s);
}

template<>
long double cast(const std::string &s) {
    return std::stold(s);
}

template<>
bool check<int>(const std::string &s) {
    try {
        cast<std::string, int>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<long>(const std::string &s) {
    try {
        cast<std::string, long>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<long long>(const std::string &s) {
    try {
        cast<std::string, long long>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<unsigned int>(const std::string &s) {
    try {
        cast<std::string, unsigned int>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<unsigned long>(const std::string &s) {
    try {
        cast<std::string, unsigned long>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<unsigned long long>(const std::string &s) {
    try {
        cast<std::string, unsigned long long>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<bool>(const std::string &s) {
    try {
        cast<std::string, bool>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<float>(const std::string &s) {
    try {
        cast<std::string, float>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<double>(const std::string &s) {
    try {
        cast<std::string, double>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

template<>
bool check<long double>(const std::string &s) {
    try {
        cast<std::string, long double>(s);
        return true;
    }
    catch (const std::invalid_argument &) {
        return false;
    }
}

}