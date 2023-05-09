/*************************  dispatch_example1.cpp   ***************************
Author:        Agner Fog
Date created:  2012-05-30
Last modified: 2020-02-25
Version:       2.02.00
Project:       vector class library

Description:   Example of automatic CPU dispatching.
               This shows how to compile vector code in multiple versions, each
               optimized for a different instruction set. The optimal version is
               selected by a dispatcher at run time.

There are two examples of automatic dispatching:

dispatch_example1.cpp: Uses separate function names for each version.
                       This is useful for simple cases with one or a few functions.

dispatch_example2.cpp: Uses separate namespaces for each version.
                       This is the recommended method for cases with multiple functions,
                       classes, objects, etc.

The code has two sections: 

Dispatched code: This code is compiled multiple times to generate multiple instances
of the compiled code, each one optimized for a different instruction set. The
dispatched code section contains the speed-critical part of the program.

Common code: This code is compiled only once, using the lowest instruction set.
The common code section contains the dispatcher, startup code, user interface, and 
other parts of the program that do not need advanced optimization.

To compile this code, do as in this example:

# Example of compiling dispatch example with Gnu or Clang compiler:
# Compile dispatch_example1.cpp four times for different instruction sets:

# Compile for AVX
clang++ -O2 -m64 -mavx -std=c++17 -c dispatch_example1.cpp -od7.o

# Compile for AVX2
clang++ -O2 -m64 -mavx2 -mfma -std=c++17 -c dispatch_example1.cpp -od8.o

# Compile for AVX512
clang++ -O2 -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -std=c++17 -c dispatch_example1.cpp -od10.o

# The last compilation uses the lowest supported instruction set (SSE2)
# This includes the main program, and links all versions together:
# (Change test.exe to test in Linux and Mac)
clang++ -O2 -m64 -msse2 -std=c++17 dispatch_example1.cpp instrset_detect.cpp d7.o d8.o d10.o -otest.exe

# Run the program
./test.exe

(c) Copyright 2012-2022 Agner Fog.
Apache License version 2.0 or later.
******************************************************************************/

/* The different instruction sets are defined in instrset_detect.cpp:
2:  SSE2
3:  SSE3
4:  SSSE3 (Supplementary SSE3)
5:  SSE4.1
6:  SSE4.2
7:  AVX
8:  AVX2
9:  AVX512F
10: AVX512VL + AVX512BW + AVX512DQ
*/


#include <stdio.h>
#include "vectorclass.h"

// Define function type
// Change this to fit the entry function. Should not contain vector types:
typedef float MyFuncType(float const []);

// function prototypes for each version
MyFuncType  myfunc_SSE2, myfunc_AVX, myfunc_AVX2, myfunc_AVX512;

// function prototypes for common entry point and dispatcher
MyFuncType  myfunc, myfunc_dispatch;

// Define name of entry function depending on which instruction set we compile for
#if   INSTRSET >= 10                   // AVX512VL
#define FUNCNAME myfunc_AVX512
#elif INSTRSET >= 8                    // AVX2
#define FUNCNAME myfunc_AVX2
#elif INSTRSET >= 7                    // AVX
#define FUNCNAME myfunc_AVX
#elif INSTRSET == 2
#define FUNCNAME myfunc_SSE2           // SSE2
#else
#error Unsupported instruction set
#endif

/******************************************************************************
                             Dispatched code

Everything in this section is compiled multiple times, with one version for
each instruction set. Speed-critical vector code belongs here.
******************************************************************************/

// This is the dispatched function that is compiled in multiple versions with different names.
// Make sure this function is static to prevent clash with other versions having the same name.
// The function cannot be member of a class.
static float sum (float const f[]) {
    // This example adds 16 floats
    Vec16f a;                          // vector of 16 floats
    a.load(f);                         // load array into vector
    return horizontal_add(a);          // return sum of 16 elements
}

// -----------------------------------------------------------------------------
//                       Entry function
// -----------------------------------------------------------------------------
// This is the entry function that is accessed through the dispatcher.
// This serves as the interface between the common code and the dispatched code.
// The entry function cannot be member of a class.
// The entry function must use arrays rather than vectors for input and output.
float FUNCNAME (float const f[]) {
    return sum(f);
}


/**********************************************************************************
                             Common code

Everything in this section is compiled only once, using the lowest instruction set. 

The dispatcher must be placed here. Program main(), user interface, and other
less critical parts of the code are also placed in the common code section.
**********************************************************************************/

#if INSTRSET == 2
// The common code is only included in the lowest of the compiled versions


// ---------------------------------------------------------------------------------
//                       Dispacther
// ---------------------------------------------------------------------------------
// This function pointer initially points to the dispatcher.
// After the first call, it points to the selected version of the entry function
MyFuncType * myfunc_pointer = &myfunc_dispatch;            // function pointer

// Dispatcher
float myfunc_dispatch(float const f[]) {
    int iset = instrset_detect();                          // Detect supported instruction set
    // Choose which version of the entry function we want to point to:
    if      (iset >= 10) myfunc_pointer = &myfunc_AVX512;  // AVX512 version
    else if (iset >=  8) myfunc_pointer = &myfunc_AVX2;    // AVX2 version
    else if (iset >=  7) myfunc_pointer = &myfunc_AVX;     // AVX version
    else if (iset >=  2) myfunc_pointer = &myfunc_SSE2;    // SSE2 version
    else {
        // Error: lowest instruction set not supported.
        // Put any appropriate error handler here
        fprintf(stderr, "\nError: Instruction set SSE2 not supported on this computer");
        return 0.f;
    }
    // continue in dispatched version of the function
    return (*myfunc_pointer)(f);
}


// Call the entry function through the function pointer.
// The first time this function is called, it goes through the dispatcher.
// The dispatcher will change the function pointer so that all subsequent
// calls go directly to the optimal version of the entry function
inline float myfunc(float const f[]) {
    return (*myfunc_pointer)(f);                 // go to dispatched version
}


// ---------------------------------------------------------------------------------
//                       Program main
// ---------------------------------------------------------------------------------
int main() {

    // array of 16 floats
    float const a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    float sum = myfunc(a);                       // call function with dispatching

    printf("\nsum = %8.2f \n", sum);             // print result (= 136.00)

    return 0;
}

#endif  // INSTRSET == 2
