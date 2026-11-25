// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file Summary.cc
///
/// @brief The summary at the end of a run. It is in its own file, so that code
/// added to it cannot change what the compiler inlines in the groups.

#include "Summary.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace bench {

namespace {

// A speedup closer to 1.0 than this is never reported as different
constexpr double kMinimumNoise = 0.02;

// A row that moves more than this between two runs is listed as one to run again
constexpr double kRerunSpread = 0.05;

// Rows with this text in the name are expected to give a different result
const char* const kExpectedDifferent = "active tiles";

// A row that only exists with a NeighborAccessor is compared with the row of the tool it replaces
std::string
baselineTool(const std::string& tool)
{
    if (tool == "DiscreteField NeighborAccessor") return "LevelSetAdvection DiscreteField";
    return tool;
}

using Key = std::tuple<std::string, std::string, std::string>; // group, tool, name

/// The runs of one row
struct Measurement
{
    std::vector<double> times;
    std::set<std::string> hashes;

    double minimum() const { return *std::min_element(times.begin(), times.end()); }

    /// The difference between the slowest and the fastest run, relative to the fastest
    double spread() const
    {
        if (times.size() < 2) return 0.0;
        const auto limits = std::minmax_element(times.begin(), times.end());
        return (*limits.second - *limits.first) / *limits.first;
    }
};

using Measurements = std::map<Key, Measurement>;

Measurements
collect(const std::vector<Record>& records)
{
    Measurements measurements;
    for (const Record& record : records) {
        Measurement& measurement = measurements[Key(record.group, record.tool, record.name)];
        measurement.times.push_back(record.minimum);
        measurement.hashes.insert(record.hash);
    }
    return measurements;
}

double
geometricMean(const std::vector<double>& values)
{
    double sum = 0.0;
    for (double value : values) sum += std::log(value);
    return std::exp(sum / double(values.size()));
}

double
median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

/// One decimal place, and 1.0x for anything within the noise
std::string
speedupText(double value, double noise)
{
    char text[32];
    if (std::abs(value - 1.0) <= std::max(noise, kMinimumNoise)) return "1.0x";
    std::snprintf(text, sizeof(text), "%.1fx", value);
    return text;
}

std::string
number(double value, const char* format)
{
    char text[32];
    std::snprintf(text, sizeof(text), format, value);
    return text;
}

bool
isExpectedDifferent(const std::string& name)
{
    return name.find(kExpectedDifferent) != std::string::npos;
}

/// One row of this run against the same row of the baseline
struct Comparison
{
    Key key;
    double baselineMs;
    double timeMs;
    double speedup() const { return baselineMs / timeMs; }
};

using ToolKey = std::pair<std::string, std::string>; // group, tool

std::string
machineText(const Options& options)
{
    std::string cpu = "unknown cpu";
    std::ifstream file("/proc/cpuinfo");
    for (std::string line; std::getline(file, line);) {
        if (line.compare(0, 10, "model name") == 0 && line.find(':') != std::string::npos) {
            cpu = line.substr(line.find(':') + 2);
            break;
        }
    }
    std::ostringstream text;
    text << cpu << ", " << std::thread::hardware_concurrency() << " hardware threads, running "
        << options.threads << " threads";
    return text.str();
}

/// The times of the accessor rows relative to the ValueAccessor row of the same tool. The ratio
/// does not depend on the machine, as the speedup of a build over a baseline does not.
void
writeAccessorComparison(std::ostream& out, const Measurements& measurements)
{
    std::map<std::string, std::vector<std::pair<std::string, double>>> tools;
    for (const auto& entry : measurements) {
        if (std::get<0>(entry.first) != "accessor") continue;
        tools[std::get<1>(entry.first)].push_back({std::get<2>(entry.first), entry.second.minimum()});
    }
    if (tools.empty()) return;
    out << "## Accessor reads\n\n"
        << "Time of each row relative to the ValueAccessor row of the same tool. Below 1.00x is faster.\n\n"
        << "| Tool | Row | ms | Relative |\n|---|---|---|---|\n";
    for (const auto& tool : tools) {
        double reference = 0.0;
        for (const auto& row : tool.second) {
            if (row.first.compare(0, 13, "ValueAccessor") == 0) reference = row.second;
        }
        for (const auto& row : tool.second) {
            out << "| " << tool.first << " | " << row.first << " | " << number(row.second, "%.1f") << " | "
                << (reference > 0.0 ? number(row.second / reference, "%.2fx") : std::string()) << " |\n";
        }
    }
    out << "\n";
}

} // anonymous namespace


std::vector<Record>
toRecords(const std::vector<Row>& rows)
{
    std::vector<Record> records;
    for (const Row& row : rows) {
        records.push_back(Record{row.group, row.tool, row.name, row.timing.minimum, row.hashText()});
    }
    return records;
}

std::vector<Record>
readCsv(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path);
    std::vector<Record> records;
    std::map<std::string, size_t> columns;
    for (std::string line; std::getline(file, line);) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::istringstream stream(line);
        for (std::string field; std::getline(stream, field, ',');) fields.push_back(field);
        if (fields.size() > 1 && fields[0] == "group") { // a header, which also starts a joined file
            columns.clear();
            for (size_t i = 0; i < fields.size(); ++i) columns[fields[i]] = i;
            continue;
        }
        for (const char* needed : {"group", "tool", "name", "min_ms", "hash"}) {
            if (!columns.count(needed)) throw std::runtime_error(path + " has no column " + needed);
        }
        if (fields.size() <= columns["hash"]) throw std::runtime_error(path + " has a short line: " + line);
        records.push_back(Record{fields[columns["group"]], fields[columns["tool"]], fields[columns["name"]],
            std::stod(fields[columns["min_ms"]]), fields[columns["hash"]]});
    }
    return records;
}

void
writeCsv(const std::string& path, const std::vector<Row>& rows)
{
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot write " + path);
    file << Row::csvHeader() << "\n";
    for (const Row& row : rows) file << row.csvLine() << "\n";
}

bool
writeSummary(std::ostream& out, const Options& options, const std::vector<Record>& records,
    const std::vector<Record>& baseline)
{
    const Measurements measurements = collect(records);
    const Measurements reference = collect(baseline);

    out << "# vdb_bench summary\n\n" << machineText(options) << ". Repeat " << options.repeat << ".\n\n";

    // The noise of a group: how much its rows move between runs, and never less than the minimum
    std::map<std::string, std::vector<double>> spreads;
    for (const Measurements* source : {&measurements, &reference}) {
        for (const auto& entry : *source) {
            if (entry.second.times.size() > 1) spreads[std::get<0>(entry.first)].push_back(entry.second.spread());
        }
    }
    auto noiseOf = [&](const std::string& group) { return median(spreads[group]); };

    std::vector<std::string> notes;
    std::vector<std::string> mismatches;
    std::map<ToolKey, std::vector<Comparison>> tools;
    std::map<ToolKey, double> totals;
    for (const auto& entry : measurements) {
        const Key& key = entry.first;
        const ToolKey toolKey(std::get<0>(key), std::get<1>(key));
        totals[toolKey] += entry.second.minimum();
        const std::string label = std::get<0>(key) + " / " + std::get<1>(key) + " / " + std::get<2>(key);
        if (entry.second.spread() > kRerunSpread) {
            notes.push_back("run again: " + label + " moved " + number(100.0 * entry.second.spread(), "%.0f") +
                "% between runs");
        }
        if (entry.second.hashes.size() > 1) notes.push_back("the result varies between runs: " + label);
        if (reference.empty()) continue;
        const auto found = reference.find(Key(std::get<0>(key), baselineTool(std::get<1>(key)), std::get<2>(key)));
        if (found == reference.end()) continue;
        if (found->second.hashes != entry.second.hashes) {
            if (isExpectedDifferent(std::get<2>(key))) continue; // left out of the speedups
            mismatches.push_back(label);
            continue;
        }
        tools[toolKey].push_back(Comparison{key, found->second.minimum(), entry.second.minimum()});
    }

    if (reference.empty()) {
        out << "| Tool | Rows | Total ms |\n|---|---|---|\n";
        std::map<ToolKey, int> counts;
        for (const auto& entry : measurements) {
            ++counts[ToolKey(std::get<0>(entry.first), std::get<1>(entry.first))];
        }
        for (const auto& tool : totals) {
            out << "| " << tool.first.second << " (" << tool.first.first << ") | " << counts[tool.first] << " | "
                << number(tool.second, "%.1f") << " |\n";
        }
        out << "\n";
    } else {
        out << "Speedup over the baseline, which is the time of the baseline divided by the time of this run."
            << " More than 1.0x is faster.\n\n| Tool | Speedup |\n|---|---|\n";
        for (const auto& tool : tools) {
            std::vector<double> speedups;
            for (const Comparison& comparison : tool.second) speedups.push_back(comparison.speedup());
            out << "| " << tool.first.second << " | "
                << speedupText(geometricMean(speedups), noiseOf(tool.first.first)) << " |\n";
        }
        out << "\n## Rows\n\n";
        for (const auto& tool : tools) {
            std::vector<double> speedups;
            for (const Comparison& comparison : tool.second) speedups.push_back(comparison.speedup());
            const auto limits = std::minmax_element(speedups.begin(), speedups.end());
            out << "### " << tool.first.second << " (" << tool.first.first << ")\n\n"
                << "Lowest " << number(*limits.first, "%.2fx") << ", highest " << number(*limits.second, "%.2fx")
                << ", geometric mean " << number(geometricMean(speedups), "%.2fx") << ".\n\n"
                << "| Row | Baseline ms | ms | Speedup |\n|---|---|---|---|\n";
            for (const Comparison& comparison : tool.second) {
                out << "| " << std::get<2>(comparison.key) << " | " << number(comparison.baselineMs, "%.1f") << " | "
                    << number(comparison.timeMs, "%.1f") << " | " << number(comparison.speedup(), "%.2fx") << " |\n";
            }
            out << "\n";
        }
    }

    writeAccessorComparison(out, measurements);

    if (!mismatches.empty()) {
        out << "## A result differs from the baseline\n\n"
            << "These rows are left out of the speedups. A change that is not meant to change a result"
            << " has a bug, or the tool is not deterministic.\n\n";
        for (const std::string& label : mismatches) out << "- " << label << "\n";
        out << "\n";
    }
    if (!notes.empty()) {
        out << "## Notes\n\n";
        for (const std::string& note : notes) out << "- " << note << "\n";
        out << "\n";
    }
    return mismatches.empty();
}

} // namespace bench
