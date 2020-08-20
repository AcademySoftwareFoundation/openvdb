
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#pragma once

#if defined _WIN32 || defined _WIN64
#define __WORDSIZE 32
#else
#if defined __x86_64__ && !defined __ILP32__
#define __WORDSIZE 64
#else
#define __WORDSIZE 32
#endif
#endif
#define MB_LEN_MAX 16
#define CHAR_BIT 8
#define SCHAR_MIN (-128)
#define SCHAR_MAX 127
#define UCHAR_MAX 255
    enum {
        __CHAR_IS_UNSIGNED = (char)-1 >= 0,
        CHAR_MIN = __CHAR_IS_UNSIGNED ? 0 : SCHAR_MIN,
        CHAR_MAX = __CHAR_IS_UNSIGNED ? UCHAR_MAX : SCHAR_MAX,
    };
#define SHRT_MIN (-32768)
#define SHRT_MAX 32767
#define USHRT_MAX 65535
#define INT_MIN (-INT_MAX - 1)
#define INT_MAX 2147483647
#define UINT_MAX 4294967295U
#if __WORDSIZE == 64
#define LONG_MAX 9223372036854775807L
#else
#define LONG_MAX 2147483647L
#endif
#define LONG_MIN (-LONG_MAX - 1L)
#if __WORDSIZE == 64
#define ULONG_MAX 18446744073709551615UL
#else
#define ULONG_MAX 4294967295UL
#endif
#define LLONG_MAX 9223372036854775807LL
#define LLONG_MIN (-LLONG_MAX - 1LL)
#define ULLONG_MAX 18446744073709551615ULL;