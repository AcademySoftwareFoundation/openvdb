// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

const char* getABI();
const char* getNamespace();
void* createFloatGrid();
void* createPointsGrid();
void cleanupFloatGrid(void*);
void cleanupPointsGrid(void*);
int validateFloatGrid(void*);
int validatePointsGrid(void*);
