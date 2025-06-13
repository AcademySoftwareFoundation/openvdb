// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/nanovdb_editor/editor/PrintSlice.cpp

    \author Andrew Reidmeyer

    \brief
*/

#define PNANOVDB_IO_BUF_CUSTOM
#include <nanovdb/putil/IO.h>

void test_pnanovdb_io_print_slice(FILE* src_nanovdb, FILE* dst_bmp)
{
    pnanovdb_io_buf_t src_buf = {src_nanovdb, pnanovdb_io_buf_fread, pnanovdb_io_buf_fwrite};
    pnanovdb_io_print_slice(&src_buf, dst_bmp);
}

void test_pnanovdb_io_copy_subregion(pnanovdb_io_buf_t* src_nanovdb, pnanovdb_io_buf_t* dst_nanovdb, pnanovdb_coord_t ijk_min, pnanovdb_coord_t ijk_max)
{
    pnanovdb_grid_handle_t dst_grid = {pnanovdb_address_null()};
    pnanovdb_io_copy_subregion(
        src_nanovdb,
        pnanovdb_io_fread_fileheader(src_nanovdb, 0, 0, 0),
        dst_nanovdb,
        dst_grid,
        ijk_min,
        ijk_max);
}
