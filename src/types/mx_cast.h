/**
 * @file mx_cast.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines cast for basic types. cast may also be defined elsewhere for other types. 
 * cast is not intended to cover all possible type conversions, 
 * but for conversions relevant to the scope of various Mechanica functionality. 
 * @date 2021-08-09
 * 
 */
#ifndef _SRC_TYPES_MX_CAST_H_
#define _SRC_TYPES_MX_CAST_H_

#include <stdexcept>
#include <string>


namespace mx {

template<typename T, typename S> 
S cast(const T&);
template<typename T, typename S> 
S cast(T*);
// template<typename T, typename S> 
// S *cast(const T&);
// template<typename T, typename S> 
// S *cast(T*);

// template<typename T>
// std::string cast(const T &t) {
//     return cast<T, std::string>(t);
// }

// template<typename T> 
// T cast(const std::string &);

template<> std::string cast(const int &t);

template<> std::string cast(const long &t);

template<> std::string cast(const long long &t);

template<> std::string cast(const unsigned int &t);

template<> std::string cast(const unsigned long &t);

template<> std::string cast(const unsigned long long &t);

template<> std::string cast(const bool &t);

template<> std::string cast(const float &t);

template<> std::string cast(const double &t);

template<> std::string cast(const long double &t);

template<> int cast(const std::string &s);

template<> long cast(const std::string &s);

template<> long long cast(const std::string &s);

template<> unsigned int cast(const std::string &s);

template<> unsigned long cast(const std::string &s);

template<> unsigned long long cast(const std::string &s);

template<> bool cast(const std::string &s);

template<> float cast(const std::string &s);

template<> double cast(const std::string &s);

template<> long double cast(const std::string &s);

template<typename T, typename S>
bool check(const T&);

template<typename T>
bool check(const std::string&);

template<> bool check<int>(const std::string &s);

template<> bool check<long>(const std::string &s);

template<> bool check<long long>(const std::string &s);

template<> bool check<unsigned int>(const std::string &s);

template<> bool check<unsigned long>(const std::string &s);

template<> bool check<unsigned long long>(const std::string &s);

template<> bool check<bool>(const std::string &s);

template<> bool check<float>(const std::string &s);

template<> bool check<double>(const std::string &s);

template<> bool check<long double>(const std::string &s);

}

#endif // _SRC_TYPES_MX_CAST_H_