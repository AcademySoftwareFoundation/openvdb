// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Summary.h
///
/// @brief The rows of a run, saved to and loaded from a CSV file, and the
/// summary written at the end of a run.

#pragma once

#include "Harness.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace bench {

/// What a summary needs of a row, which is also all a CSV file keeps of it
struct Record
{
    std::string group;
    std::string tool;
    std::string name;
    double minimum = 0.0; // milliseconds
    std::string hash;
};

std::vector<Record> toRecords(const std::vector<Row>& rows);

/// Read a CSV file written by writeCsv() or by the option --csv. Files can be joined, and
/// a row that is in the file more than once counts as one more run of that row.
/// @throw std::runtime_error if the file cannot be read
std::vector<Record> readCsv(const std::string& path);

/// @throw std::runtime_error if the file cannot be written
void writeCsv(const std::string& path, const std::vector<Row>& rows);

/// @brief Write the summary as markdown: the time of each tool, or the speedup over the
/// baseline if there is one, the notes on noise and on a result that differs from the
/// baseline, and the comparison of the accessors in the group accessor.
/// @return false if a result differs from the baseline
bool writeSummary(std::ostream& out, const Options& options, const std::vector<Record>& records,
    const std::vector<Record>& baseline);

} // namespace bench
