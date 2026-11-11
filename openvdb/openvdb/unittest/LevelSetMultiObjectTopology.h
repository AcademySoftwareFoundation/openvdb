// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file unittest/LevelSetMultiObjectTopology.h
///
/// @brief Test-only topology utilities for multi-label level sets:
/// - digitalTopology(): foreground components, enclosed background cavities
///   and the Euler characteristic of one label of a label grid, under a
///   digital connectivity pair;
/// - meshTopology(): components, Euler characteristic, closedness and
///   manifoldness of a triangle mesh, with the positions of the defects;
/// - verifyLabelTopology(): both for one label of a multi-label level set,
///   meshed under the same connectivity pair, checked against each other
///   (and an optional expectation), with a debug dump on failure;
/// - connectivityOf(): the connectivity pair an evolver protects topology
///   under, so that tests verify with the rule they evolve with.
///
/// Debug dumps go to <dir>/<test suite>.<test name>/<tag>/ with
/// dir = $OPENVDB_MOGAC_DEBUG_DIR, or <temp>/openvdb_mogac_debug if it is not
/// set. They are written on every failed verification and, when
/// OPENVDB_MOGAC_DEBUG_DIR is set, on every verification: pair.vdb (grids
/// "phi" and "labels", the labels as an Int32 copy), label<L>.ply (the mesh)
/// and report.txt (per-component voxel counts and bounding boxes, cavity
/// bounding boxes, boundary and non-manifold edge positions).
///
/// Header-only; shared by TestLevelSetMultiObject.cc and
/// TestLevelSetMultiObjectMesh.cc.

#ifndef OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_TOPOLOGY_HAS_BEEN_INCLUDED
#define OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_TOPOLOGY_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/LevelSetMultiObject.h>
#include <openvdb/tools/LevelSetMultiObjectMesh.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif

namespace lsmo_test {

using Connectivity = openvdb::tools::MultiObjectConnectivity;

inline std::string
connectivityName(Connectivity c)
{
    switch (c) {
        case Connectivity::FG26_BG6: return "FG26_BG6";
        case Connectivity::FG6_BG26: return "FG6_BG26";
        case Connectivity::FG18_BG6: return "FG18_BG6";
        case Connectivity::FG6_BG18: return "FG6_BG18";
    }
    return "?";
}

/// The connectivity pair the evolver preserves topology under
/// (LevelSetMultiObject::setConnectivity).
template<typename EvolverT>
inline Connectivity
connectivityOf(const EvolverT& evolver) { return evolver.connectivity(); }

/// The phi floor (world units) the evolver keeps at protected interfaces
/// (LevelSetMultiObject::phiFloor), to mesh its state as it sees it.
template<typename EvolverT>
inline double
phiFloorOf(const EvolverT& evolver) { return evolver.phiFloor(); }


////////////////////////////////////////
// Digital topology

struct DigitalTopology
{
    int components = 0; ///< foreground components (fg adjacency)
    int cavities = 0;   ///< background components (bg adjacency) not touching the domain boundary
    int euler = 0;      ///< Euler characteristic of the region (ball 1, hollow shell 2, solid torus 0)
};

inline bool operator==(const DigitalTopology& a, const DigitalTopology& b)
{
    return a.components == b.components && a.cavities == b.cavities && a.euler == b.euler;
}
inline bool operator!=(const DigitalTopology& a, const DigitalTopology& b) { return !(a == b); }

inline std::ostream& operator<<(std::ostream& os, const DigitalTopology& t)
{
    return os << "(components " << t.components << ", cavities " << t.cavities << ", euler "
              << t.euler << ")";
}

/// One connected component of a digital region (or of its complement).
struct RegionPart
{
    openvdb::Index64 voxels = 0;
    openvdb::CoordBBox bbox;
};

/// Per-component details of digitalTopology().
struct RegionDetails
{
    std::vector<RegionPart> components; ///< foreground components
    std::vector<RegionPart> cavities;   ///< enclosed background components
};

namespace detail {

/// Squared-length bound of an adjacency: 6 -> 1, 18 -> 2, 26 -> 3.
inline int adjBound(int n) { return n == 6 ? 1 : (n == 18 ? 2 : 3); }

inline void connectivityPair(Connectivity c, int& fg, int& bg)
{
    switch (c) {
        case Connectivity::FG26_BG6: fg = 26; bg = 6; break;
        case Connectivity::FG6_BG26: fg = 6; bg = 26; break;
        case Connectivity::FG18_BG6: fg = 18; bg = 6; break;
        case Connectivity::FG6_BG18: fg = 6; bg = 18; break;
    }
}

/// Dense boolean volume over a bounding box; outside reads false.
struct Volume
{
    openvdb::Coord lo;
    int nx, ny, nz;
    std::vector<char> v;
    explicit Volume(const openvdb::CoordBBox& b)
        : lo(b.min()), nx(b.dim().x()), ny(b.dim().y()), nz(b.dim().z())
        , v(size_t(nx) * ny * nz, 0) {}
    size_t idx(int i, int j, int k) const { return (size_t(k) * ny + j) * nx + i; }
    bool in(int i, int j, int k) const
    {
        return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz;
    }
    bool at(int i, int j, int k) const { return in(i, j, k) && v[idx(i, j, k)]; }
};

/// A component found by components(): its size, bounding box (index space)
/// and whether it touches the box boundary.
struct Component
{
    RegionPart part;
    bool touching = false;
};

/// Components of the voxels with value @a want under adjacency @a n.
inline std::vector<Component> components(const Volume& vol, bool want, int n)
{
    const int bound = adjBound(n);
    std::vector<char> seen(vol.v.size(), 0);
    std::vector<size_t> stack;
    std::vector<Component> out;
    for (int k = 0; k < vol.nz; ++k)
    for (int j = 0; j < vol.ny; ++j)
    for (int i = 0; i < vol.nx; ++i) {
        const size_t s = vol.idx(i, j, k);
        if (bool(vol.v[s]) != want || seen[s]) continue;
        Component comp;
        seen[s] = 1;
        stack.assign(1, s);
        while (!stack.empty()) {
            const size_t c = stack.back();
            stack.pop_back();
            const int ci = int(c % vol.nx), cj = int((c / vol.nx) % vol.ny),
                      ck = int(c / (size_t(vol.nx) * vol.ny));
            ++comp.part.voxels;
            comp.part.bbox.expand(vol.lo.offsetBy(ci, cj, ck));
            if (ci == 0 || cj == 0 || ck == 0 || ci == vol.nx - 1 || cj == vol.ny - 1
                || ck == vol.nz - 1) comp.touching = true;
            for (int c2 = -1; c2 <= 1; ++c2)
            for (int b = -1; b <= 1; ++b)
            for (int a = -1; a <= 1; ++a) {
                const int d2 = a * a + b * b + c2 * c2;
                if (d2 == 0 || d2 > bound) continue;
                const int qi = ci + a, qj = cj + b, qk = ck + c2;
                if (!vol.in(qi, qj, qk)) continue;
                const size_t q = vol.idx(qi, qj, qk);
                if (bool(vol.v[q]) != want || seen[q]) continue;
                seen[q] = 1;
                stack.push_back(q);
            }
        }
        out.push_back(comp);
    }
    return out;
}

/// Euler characteristic of the union of closed unit cubes (26-adjacency).
inline long eulerClosedCubes(const Volume& vol)
{
    long V = 0, E = 0, F = 0, C = 0;
    // lattice point (i,j,k) is the low corner of voxel (i,j,k)
    for (int k = 0; k <= vol.nz; ++k)
    for (int j = 0; j <= vol.ny; ++j)
    for (int i = 0; i <= vol.nx; ++i) {
        bool vert = false;
        for (int c = 0; c < 2; ++c) for (int b = 0; b < 2; ++b) for (int a = 0; a < 2; ++a)
            vert |= vol.at(i - a, j - b, k - c);
        V += vert;
        // edges from (i,j,k) along x, y, z
        E += vol.at(i, j, k) || vol.at(i, j - 1, k) || vol.at(i, j, k - 1) || vol.at(i, j - 1, k - 1);
        E += vol.at(i, j, k) || vol.at(i - 1, j, k) || vol.at(i, j, k - 1) || vol.at(i - 1, j, k - 1);
        E += vol.at(i, j, k) || vol.at(i - 1, j, k) || vol.at(i, j - 1, k) || vol.at(i - 1, j - 1, k);
        // faces with low corner (i,j,k), normal x, y, z
        F += vol.at(i, j, k) || vol.at(i - 1, j, k);
        F += vol.at(i, j, k) || vol.at(i, j - 1, k);
        F += vol.at(i, j, k) || vol.at(i, j, k - 1);
        C += vol.at(i, j, k);
    }
    return V - E + F - C;
}

/// Euler characteristic of the face-connected complex (6-adjacency):
/// voxels, face-adjacent pairs, full 2x2 squares, full 2x2x2 blocks.
inline long eulerFaceComplex(const Volume& vol)
{
    long V = 0, E = 0, F = 0, C = 0;
    for (int k = 0; k < vol.nz; ++k)
    for (int j = 0; j < vol.ny; ++j)
    for (int i = 0; i < vol.nx; ++i) {
        if (!vol.at(i, j, k)) continue;
        ++V;
        const bool x = vol.at(i + 1, j, k), y = vol.at(i, j + 1, k), z = vol.at(i, j, k + 1);
        E += x + y + z;
        const bool xy = x && y && vol.at(i + 1, j + 1, k);
        const bool xz = x && z && vol.at(i + 1, j, k + 1);
        const bool yz = y && z && vol.at(i, j + 1, k + 1);
        F += xy + xz + yz;
        C += xy && z && vol.at(i, j + 1, k + 1) && vol.at(i + 1, j, k + 1)
             && vol.at(i + 1, j + 1, k + 1);
    }
    return V - E + F - C;
}

/// Number of 2x2x2 blocks in which exactly two voxels of value @a want lie
/// at opposite corners. With want = true (18-adjacency) the lattice point
/// they share is split in two; with want = false (6-adjacency paired with
/// 18) the six foreground voxels around the unconnected background pair
/// enclose no tunnel, so their hexagon is filled.
inline long antipodalBlocks(const Volume& vol, bool want)
{
    long n = 0;
    for (int k = -1; k < vol.nz; ++k)
    for (int j = -1; j < vol.ny; ++j)
    for (int i = -1; i < vol.nx; ++i) {
        int count = 0, first = -1, second = -1;
        for (int c = 0; c < 8; ++c) {
            if (vol.at(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)) == want) {
                ++count;
                (first < 0 ? first : second) = c;
            }
        }
        if (count == 2 && (first ^ second) == 7) ++n;
    }
    return n;
}

} // namespace detail

/// @brief Digital topology of the region {labels == @a label} inside
/// @a domain (voxels outside @a domain are ignored), under the connectivity
/// pair @a conn.
/// @details components: foreground components under the foreground
/// adjacency; cavities: background components under the background adjacency
/// that do not touch the domain boundary (pad the domain by at least one
/// voxel); euler: Euler characteristic of the region's cell complex: union
/// of closed cubes for 26-adjacency; the same with lattice points shared
/// only by two opposite voxels of a 2x2x2 block split for 18-adjacency; the
/// face-connected complex (voxels, face pairs, full 2x2 squares, full 2x2x2
/// blocks) for 6-adjacency paired with 26; and that plus one filled hexagon
/// per 2x2x2 block missing only two opposite voxels for 6 paired with 18.
/// If @a details is given it receives every component and cavity.
template<typename LabelGridT>
inline DigitalTopology
digitalTopology(const LabelGridT& labels, int32_t label, const openvdb::CoordBBox& domain,
    Connectivity conn, RegionDetails* details = nullptr)
{
    int fg = 26, bg = 6;
    detail::connectivityPair(conn, fg, bg);
    detail::Volume vol(domain);
    openvdb::tree::ValueAccessor<const typename LabelGridT::TreeType> acc(labels.tree());
    for (int k = 0; k < vol.nz; ++k)
    for (int j = 0; j < vol.ny; ++j)
    for (int i = 0; i < vol.nx; ++i) {
        vol.v[vol.idx(i, j, k)] = static_cast<int32_t>(acc.getValue(vol.lo.offsetBy(i, j, k)))
                                  == label;
    }
    DigitalTopology t;
    const auto fgComps = detail::components(vol, true, fg);
    const auto bgComps = detail::components(vol, false, bg);
    t.components = int(fgComps.size());
    for (const auto& c : bgComps) t.cavities += !c.touching;
    if (details) {
        details->components.clear();
        details->cavities.clear();
        for (const auto& c : fgComps) details->components.push_back(c.part);
        for (const auto& c : bgComps) if (!c.touching) details->cavities.push_back(c.part);
    }
    if (fg == 6 && bg == 26) t.euler = int(detail::eulerFaceComplex(vol));
    else if (fg == 6) t.euler = int(detail::eulerFaceComplex(vol) + detail::antipodalBlocks(vol, false));
    else if (fg == 18) t.euler = int(detail::eulerClosedCubes(vol) + detail::antipodalBlocks(vol, true));
    else t.euler = int(detail::eulerClosedCubes(vol));
    return t;
}


////////////////////////////////////////
// Mesh topology

/// Topology of a triangle mesh.
struct MeshTopology
{
    int components = 0; ///< edge-connected components of the triangles
    int euler = 0;      ///< V - E + F over the vertices used by the triangles
    bool closed = true; ///< every edge has a triangle on both sides
    /// Consistently oriented 2-manifold (with boundary if not closed): each
    /// directed edge used once, at most two triangles per edge, each interior
    /// vertex's link a single cycle, every vertex used, no degenerate or
    /// zero-area triangle.
    bool manifold = true;
    std::vector<openvdb::Vec3s> boundaryEdgeMidpoints;    ///< edges with one triangle
    /// Edges with more than two triangles or two triangles of the same winding.
    std::vector<openvdb::Vec3s> nonManifoldEdgeMidpoints;
    std::vector<openvdb::Vec3s> nonManifoldVertices;      ///< link not a single cycle
    std::string defect; ///< first manifold defect found (empty if manifold)
};

/// V - E + F over the vertices used by @a tris.
inline int
eulerCharacteristic(const std::vector<openvdb::Vec3I>& tris)
{
    std::set<std::pair<uint32_t, uint32_t>> edges;
    std::set<uint32_t> verts;
    for (const openvdb::Vec3I& t : tris)
        for (int k = 0; k < 3; ++k) {
            verts.insert(t[k]);
            edges.insert({std::min(t[k], t[(k + 1) % 3]), std::max(t[k], t[(k + 1) % 3])});
        }
    return int(verts.size()) - int(edges.size()) + int(tris.size());
}

/// @brief Topology of @a m; @a dx scales the zero-area threshold (doubled
/// triangle area below 1e-10*dx^2 counts as zero).
inline MeshTopology
meshTopology(const openvdb::tools::LabelMesh& m, double dx = 1.0)
{
    using namespace openvdb;
    MeshTopology r;
    auto defect = [&](const std::string& s) {
        r.manifold = false;
        if (r.defect.empty()) r.defect = s;
    };
    const size_t nv = m.points.size();
    std::vector<Vec3I> tris;
    tris.reserve(m.triangles.size());
    for (size_t i = 0; i < m.triangles.size(); ++i) {
        const Vec3I& t = m.triangles[i];
        if (t[0] >= nv || t[1] >= nv || t[2] >= nv) {
            defect("triangle " + std::to_string(i) + ": vertex index out of range");
            continue;
        }
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) {
            defect("triangle " + std::to_string(i) + ": repeated vertex");
            continue;
        }
        const Vec3d a(m.points[t[0]]), b(m.points[t[1]]), c(m.points[t[2]]);
        if (!((b - a).cross(c - a).length() > 1e-10 * dx * dx)) {
            defect("zero-area triangle " + std::to_string(i));
        }
        tris.push_back(t);
    }
    // undirected edge -> (triangles using a->b, triangles using b->a), a < b
    std::map<std::pair<uint32_t, uint32_t>, std::pair<int, int>> edges;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> link(nv);
    for (const Vec3I& t : tris) {
        for (int k = 0; k < 3; ++k) {
            const uint32_t a = t[k], b = t[(k + 1) % 3];
            auto& e = edges[{std::min(a, b), std::max(a, b)}];
            (a < b ? e.first : e.second) += 1;
            link[a].emplace_back(b, t[(k + 2) % 3]);
        }
    }
    auto mid = [&](const std::pair<uint32_t, uint32_t>& e) {
        return (m.points[e.first] + m.points[e.second]) * 0.5f;
    };
    std::vector<char> onBoundary(nv, 0);
    for (const auto& kv : edges) {
        const int fwd = kv.second.first, bwd = kv.second.second;
        if (fwd + bwd == 1) {
            r.boundaryEdgeMidpoints.push_back(mid(kv.first));
            onBoundary[kv.first.first] = onBoundary[kv.first.second] = 1;
        } else if (fwd != 1 || bwd != 1) {
            r.nonManifoldEdgeMidpoints.push_back(mid(kv.first));
            onBoundary[kv.first.first] = onBoundary[kv.first.second] = 1;
        }
    }
    r.closed = r.boundaryEdgeMidpoints.empty();
    if (!r.nonManifoldEdgeMidpoints.empty()) {
        defect(std::to_string(r.nonManifoldEdgeMidpoints.size())
            + " edges with more than two triangles or inconsistent winding");
    }
    for (size_t v = 0; v < nv; ++v) {
        const auto& L = link[v];
        if (L.empty()) {
            defect("unused vertex " + std::to_string(v));
            continue;
        }
        if (onBoundary[v]) continue; // the link of a boundary vertex is a path
        std::map<uint32_t, uint32_t> next;
        for (const auto& e : L) next[e.first] = e.second;
        size_t steps = 0;
        uint32_t cur = L[0].first;
        do { cur = next[cur]; ++steps; } while (cur != L[0].first && steps <= L.size());
        if (steps != L.size()) {
            r.nonManifoldVertices.push_back(m.points[v]);
            defect("vertex " + std::to_string(v) + " link is not one cycle");
        }
    }
    // components (union-find over vertices of the triangles)
    std::vector<uint32_t> rep(nv);
    for (size_t i = 0; i < nv; ++i) rep[i] = uint32_t(i);
    auto find = [&](uint32_t x) {
        while (rep[x] != x) x = rep[x] = rep[rep[x]];
        return x;
    };
    for (const Vec3I& t : tris) {
        rep[find(t[0])] = find(t[1]);
        rep[find(t[1])] = find(t[2]);
    }
    std::set<uint32_t> roots;
    for (const Vec3I& t : tris) roots.insert(find(t[0]));
    r.components = int(roots.size());
    r.euler = eulerCharacteristic(tris);
    return r;
}


////////////////////////////////////////
// Verification of one label

struct TopologyReport
{
    bool ok = false;
    std::string message; ///< summary, problems, and the dump directory if written
    DigitalTopology region;
    MeshTopology mesh;
};

namespace detail {

inline void setEnv(const char* name, const std::string& value)
{
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    ::setenv(name, value.c_str(), 1);
#endif
}

inline void unsetEnv(const char* name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

inline int processId()
{
#ifdef _WIN32
    return _getpid();
#else
    return int(::getpid());
#endif
}

inline std::string
sanitizePath(const std::string& s, bool keepSlash)
{
    std::string out = s;
    for (char& c : out) {
        const bool okChar = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
            || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' || c == '='
            || (keepSlash && c == '/');
        if (!okChar) c = '_';
    }
    return out;
}

/// <dir>/<suite>.<test>/<tag>, dir from OPENVDB_MOGAC_DEBUG_DIR or the
/// system temporary directory.
inline std::filesystem::path
dumpDirectory(const std::string& tag)
{
    namespace fs = std::filesystem;
    const char* env = std::getenv("OPENVDB_MOGAC_DEBUG_DIR");
    fs::path dir = (env && *env) ? fs::path(env)
                                 : fs::temp_directory_path() / "openvdb_mogac_debug";
    std::string test = "no_test";
    if (const auto* info = ::testing::UnitTest::GetInstance()->current_test_info()) {
        test = std::string(info->test_suite_name()) + "." + info->name();
    }
    return dir / sanitizePath(test, false) / sanitizePath(tag, true);
}

inline void
writePly(const std::filesystem::path& file, const openvdb::tools::LabelMesh& m)
{
    std::ofstream os(file);
    os << "ply\nformat ascii 1.0\nelement vertex " << m.points.size()
       << "\nproperty float x\nproperty float y\nproperty float z\nelement face "
       << m.triangles.size() << "\nproperty list uchar int vertex_indices\nend_header\n";
    for (const auto& p : m.points) os << p[0] << " " << p[1] << " " << p[2] << "\n";
    for (const auto& t : m.triangles) os << "3 " << t[0] << " " << t[1] << " " << t[2] << "\n";
}

template<typename MLS>
inline void
writePair(const std::filesystem::path& file, const MLS& mls)
{
    using namespace openvdb;
    FloatGrid::Ptr phi = mls.phi()->deepCopy();
    phi->setName("phi");
    // labels as Int32 (registered for every label tree type)
    Int32Grid::Ptr labels = Int32Grid::create(
        Int32Tree::Ptr(new Int32Tree(mls.labels()->tree())));
    labels->setTransform(mls.labels()->transform().copy());
    labels->setName("labels");
    GridPtrVec grids{phi, labels};
    io::File(file.string()).write(grids);
}

inline void
writeReport(const std::filesystem::path& file, const TopologyReport& r,
    const RegionDetails& details, double dx)
{
    std::ofstream os(file);
    const size_t cap = 2000;
    auto bbox = [](const openvdb::CoordBBox& b) {
        std::ostringstream s;
        s << "[" << b.min() << " .. " << b.max() << "]";
        return s.str();
    };
    os << r.message << "\n\nvoxel size " << dx << " (positions below: world space)\n";
    os << "\nregion " << r.region << "\n";
    for (size_t i = 0; i < details.components.size(); ++i) {
        os << "  component " << i << ": " << details.components[i].voxels << " voxels, bbox "
           << bbox(details.components[i].bbox) << "\n";
    }
    for (size_t i = 0; i < details.cavities.size(); ++i) {
        os << "  cavity " << i << ": " << details.cavities[i].voxels << " voxels, bbox "
           << bbox(details.cavities[i].bbox) << "\n";
    }
    os << "\nmesh: components " << r.mesh.components << ", euler " << r.mesh.euler
       << ", closed " << r.mesh.closed << ", manifold " << r.mesh.manifold;
    if (!r.mesh.defect.empty()) os << " (" << r.mesh.defect << ")";
    os << "\n";
    auto list = [&](const char* what, const std::vector<openvdb::Vec3s>& pts) {
        os << what << ": " << pts.size() << "\n";
        for (size_t i = 0; i < pts.size() && i < cap; ++i) os << "  " << pts[i] << "\n";
        if (pts.size() > cap) os << "  ... (" << pts.size() - cap << " more)\n";
    };
    list("boundary edge midpoints", r.mesh.boundaryEdgeMidpoints);
    list("non-manifold edge midpoints", r.mesh.nonManifoldEdgeMidpoints);
    list("non-manifold vertices", r.mesh.nonManifoldVertices);
}

} // namespace detail

/// @brief Topology of label @a label of @a mls under the connectivity pair
/// @a conn, from the digital region (inside @a domain, padded by at least
/// one voxel around the label) and from its multiObjectToMesh mesh under
/// the same pair.
/// @details ok iff mesh components == region components + cavities,
/// chi(mesh) == 2 chi(region), the mesh is closed and manifold, and the
/// region equals @a expected if given. The message names the tag, label,
/// rule and every mismatch. A debug dump (see the file documentation) is
/// written to <dir>/<test>/<tag>/ when the check fails, or always when
/// OPENVDB_MOGAC_DEBUG_DIR is set; the message then gives its directory.
/// @a minimumPhi (world units) is passed to the mesher
/// (MultiObjectMeshSettings::minimumPhi); give an evolver's phiFloorOf().
template<typename MLS>
inline TopologyReport
verifyLabelTopology(const MLS& mls, int32_t label, const openvdb::CoordBBox& domain,
    Connectivity conn, const std::string& tag, const DigitalTopology* expected = nullptr,
    double minimumPhi = 0.0)
{
    TopologyReport r;
    RegionDetails details;
    r.region = digitalTopology(*mls.labels(), label, domain, conn, &details);
    openvdb::tools::MultiObjectMeshSettings settings;
    settings.connectivity = conn;
    settings.minimumPhi = minimumPhi;
    const auto meshes = openvdb::tools::multiObjectToMesh(mls, {label}, settings);
    const auto it = meshes.find(label);
    const openvdb::tools::LabelMesh mesh = it != meshes.end() ? it->second
                                                              : openvdb::tools::LabelMesh();
    r.mesh = meshTopology(mesh, mls.voxelSize());

    const std::string rule = connectivityName(conn);
    std::ostringstream msg, problems;
    msg << tag << ": label " << label << ", rule " << rule << ": region " << r.region
        << ", mesh (components " << r.mesh.components << ", euler " << r.mesh.euler << ")";
    if (r.mesh.components != r.region.components + r.region.cavities) {
        problems << "\n  mesh components " << r.mesh.components
                 << " != region components + cavities " << r.region.components + r.region.cavities
                 << " under " << rule;
    }
    if (r.mesh.euler != 2 * r.region.euler) {
        problems << "\n  mesh euler " << r.mesh.euler << " != 2 * region euler "
                 << 2 * r.region.euler << " under " << rule;
    }
    if (!r.mesh.closed) {
        problems << "\n  mesh not closed: " << r.mesh.boundaryEdgeMidpoints.size()
                 << " boundary edges";
    }
    if (!r.mesh.manifold) problems << "\n  mesh not manifold: " << r.mesh.defect;
    if (expected && r.region != *expected) {
        problems << "\n  region " << r.region << " != expected " << *expected << " under rule "
                 << rule;
    }
    const std::string p = problems.str();
    r.ok = p.empty();
    msg << (r.ok ? ": ok" : ":") << p;

    const char* env = std::getenv("OPENVDB_MOGAC_DEBUG_DIR");
    if (!r.ok || (env && *env)) {
        const std::filesystem::path dir = detail::dumpDirectory(tag);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            msg << "\n  debug dump failed: cannot create " << dir.string();
        } else {
            msg << "\n  debug dump: " << dir.string();
            r.message = msg.str();
            detail::writePair(dir / "pair.vdb", mls);
            detail::writePly(dir / ("label" + std::to_string(label) + ".ply"), mesh);
            detail::writeReport(dir / "report.txt", r, details, mls.voxelSize());
        }
    }
    r.message = msg.str();
    return r;
}

/// @brief Points OPENVDB_MOGAC_DEBUG_DIR at a fresh temporary directory for
/// its lifetime (restoring the previous value and removing the directory
/// afterwards), so that tests exercising failed verifications do not write
/// into the default dump directory.
class ScopedDebugDir
{
public:
    ScopedDebugDir()
    {
        namespace fs = std::filesystem;
        const char* old = std::getenv("OPENVDB_MOGAC_DEBUG_DIR");
        mHadOld = old != nullptr;
        if (old) mOld = old;
        static int counter = 0;
        mPath = fs::temp_directory_path() / ("openvdb_mogac_test_"
            + std::to_string(detail::processId()) + "_" + std::to_string(counter++));
        fs::remove_all(mPath);
        fs::create_directories(mPath);
        detail::setEnv("OPENVDB_MOGAC_DEBUG_DIR", mPath.string());
    }
    ~ScopedDebugDir()
    {
        if (mHadOld) detail::setEnv("OPENVDB_MOGAC_DEBUG_DIR", mOld);
        else detail::unsetEnv("OPENVDB_MOGAC_DEBUG_DIR");
        std::error_code ec;
        std::filesystem::remove_all(mPath, ec);
    }
    ScopedDebugDir(const ScopedDebugDir&) = delete;
    ScopedDebugDir& operator=(const ScopedDebugDir&) = delete;
    const std::filesystem::path& path() const { return mPath; }

private:
    std::filesystem::path mPath;
    std::string mOld;
    bool mHadOld = false;
};

} // namespace lsmo_test

/// EXPECT that a TopologyReport is ok, printing its message otherwise.
#define EXPECT_TOPOLOGY_CONSISTENT(report) \
    do { \
        const ::lsmo_test::TopologyReport& lsmo_topology_report_ = (report); \
        EXPECT_TRUE(lsmo_topology_report_.ok) << lsmo_topology_report_.message; \
    } while (0)

#endif // OPENVDB_UNITTEST_LEVEL_SET_MULTI_OBJECT_TOPOLOGY_HAS_BEEN_INCLUDED
