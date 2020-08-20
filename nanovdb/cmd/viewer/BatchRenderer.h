// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file BatchRenderer.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Class definition for a minimal, render-agnostic nanovdb Grid renderer.
*/

#pragma once

#include "Renderer.h"

class BatchRenderer : public RendererBase
{
public:
    BatchRenderer(const RendererParams& params);

    void open() override;
    void run() override;
};