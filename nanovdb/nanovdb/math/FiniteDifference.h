// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @file FiniteDifference.h
///
/// @brief Scheme selectors for finite-difference level-set operators.
///
/// @details These enums mirror their OpenVDB counterparts in
///          openvdb/math/FiniteDifference.h.  NanoVDB cannot reuse those
///          directly: OpenVDB is an optional dependency here, pulled in only by
///          the conversion utilities (tools/CreateNanoGrid.h,
///          tools/NanoToOpenVDB.h), so a core header must not include it.
///
///          Only the schemes NanoVDB actually implements are declared, plus the
///          UNKNOWN sentinels.  Enumerator values are therefore pinned
///          explicitly to the values their OpenVDB namesakes receive
///          implicitly, so that the two enumerations agree numerically and a
///          translation across the boundary is a static_cast rather than a
///          mapping table.  Adding a scheme later means adding its label at its
///          OpenVDB value -- never renumbering an existing one.

#ifndef NANOVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED
#define NANOVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED

namespace nanovdb {

namespace math {

// ---------------------------- Spatial schemes ----------------------------

/// @brief Biased (upwind) gradient scheme used by the level-set operators.
///
/// @note Values match openvdb::math::BiasedGradientScheme, in which the full
///       sequence is UNKNOWN_BIAS = -1, FIRST_BIAS = 0, SECOND_BIAS = 1,
///       THIRD_BIAS = 2, WENO5_BIAS = 3, HJWENO5_BIAS = 4.  The unimplemented
///       labels are deliberately omitted rather than declared-and-rejected, so
///       an unsupported scheme cannot be named at all; the gap in the values is
///       intentional.
///
/// @warning HJWENO5_BIAS (4) is *not* WENO5_BIAS (3).  They are different
///          operators, not spellings of one scheme: WENO5_BIAS reconstructs the
///          function and then differences it, whereas HJWENO5_BIAS applies the
///          WENO weights to the divided differences (the Hamilton-Jacobi form).
///          math::WenoStencil implements the latter, which is why it is the one
///          declared here.
enum BiasedGradientScheme {
    UNKNOWN_BIAS = -1,
    HJWENO5_BIAS =  4   ///< fifth-order Hamilton-Jacobi WENO; see math::WenoStencil
};

// ---------------------------- Temporal schemes ----------------------------

/// @brief Total-variation-diminishing Runge-Kutta scheme used to integrate the
///        level-set operators in (pseudo-)time.
///
/// @note Values match openvdb::math::TemporalIntegrationScheme, in which the
///       full sequence is UNKNOWN_TIS = -1, TVD_RK1 = 0, TVD_RK2 = 1,
///       TVD_RK3 = 2.  As above, unimplemented labels are omitted and the gap
///       in the values is intentional.
enum TemporalIntegrationScheme {
    UNKNOWN_TIS = -1,
    TVD_RK2     =  1    ///< two-stage TVD Runge-Kutta
};

} // namespace math

} // namespace nanovdb

#endif // NANOVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED
