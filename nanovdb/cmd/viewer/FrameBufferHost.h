// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
	\file FrameBufferHost.h

	\author Wil Braithwaite

	\date May 10, 2020

	\brief Class definition for an image renderer using host memory.
*/

#pragma once
#include "FrameBuffer.h"

class FrameBufferHost : public FrameBufferBase
{
public:
    bool  setup(int w, int h, InternalFormat format) override;
    bool  cleanup() override;
    bool  render(int x, int y, int w, int h) override;
    void* map(AccessType access) override;
    void  unmap() override;
    void* cudaMap(AccessType access, void* streamCUDA = nullptr) override;
    void  cudaUnmap(void* streamCUDA = nullptr) override;
    void* clMap(AccessType access, void* commandQueueCL) override;
    void  clUnmap(void* commandQueueCL) override;

private:
    void* mBuffer;
};
