// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

const char* getABI();
const char* getNamespace();
void* createGrid();
void cleanupGrid(void*);
int validateGrid(void*);
