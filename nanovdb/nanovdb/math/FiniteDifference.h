// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file FiniteDifference.h
///
/// @brief Scheme selectors for finite-difference level-set operators.
///
/// @details These enums mirror their OpenVDB counterparts in
///          openvdb/math/FiniteDifference.h.  Only the schemes NanoVDB
///          implements are declared, with their enumerator values pinned to the
///          OpenVDB precedent.

#ifndef NANOVDB_MATH_FINITEDIFFERENCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_FINITEDIFFERENCE_H_HAS_BEEN_INCLUDED

namespace nanovdb {

namespace math {

/// @brief Biased (upwind) gradient scheme used by the level-set operators.
enum BiasedGradientScheme {
    UNKNOWN_BIAS = -1,
    HJWENO5_BIAS =  4   ///< fifth-order Hamilton-Jacobi WENO; see math::WenoStencil
};

/// @brief Total-variation-diminishing Runge-Kutta scheme used to integrate the
///        level-set operators in (pseudo-)time.
enum TemporalIntegrationScheme {
    UNKNOWN_TIS = -1,
    TVD_RK2     =  1    ///< two-stage TVD Runge-Kutta
};

} // namespace math

} // namespace nanovdb

#endif // NANOVDB_MATH_FINITEDIFFERENCE_H_HAS_BEEN_INCLUDED
