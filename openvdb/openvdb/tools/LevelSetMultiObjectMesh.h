// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file tools/LevelSetMultiObjectMesh.h
///
/// @brief Topology-constrained contouring of multi-object level sets
/// (MultiLevelSet, see LevelSetMultiObject.h): every label is meshed into a
/// closed, consistently wound 2-manifold whose topology matches the label's
/// digital voxel region under a chosen (foreground, background) adjacency
/// pair, MultiObjectConnectivity.
///
/// @par Usage
/// @code
/// auto mls = tools::MultiLevelSet<>::fromSignedGrids(grids);
/// tools::MultiObjectMeshSettings settings;
/// settings.connectivity = tools::MultiObjectConnectivity::FG26_BG6;
/// std::map<int32_t, tools::LabelMesh> meshes = tools::multiObjectToMesh(*mls, {}, settings);
/// for (const auto& [label, mesh] : meshes) { /* mesh.points, mesh.triangles */ }
/// @endcode
///
/// @par Algorithm
/// - Cells are enumerated in blocks of LeafNodeType::DIM^3 minimum corners:
///   every leaf of the label tree and its seven lower neighbor blocks (a
///   cell whose minimum corner lies one voxel below a leaf has corners in
///   it). A block reads its labels plus a one-voxel upper halo once; cells
///   whose eight labels agree are skipped, so the work is proportional to the
///   number of leaves near interfaces. Interfaces inside tiles (which valid
///   MultiLevelSets never have) are not seen.
/// - Classification is by labels, never by the sign of phi: for every
///   non-zero label l among a mixed cell's corners the case index has bit i
///   set when corner i has label l (inactive labels count: they hold the
///   label of the region they lie in), so the meshes follow exactly the
///   digital regions R_l.
/// - The case table of the connectivity pair gives the cell's surface
///   patches (disks, and tubes around body diagonals for 26-adjacency).
/// - A vertex lies on the cell edge from p (in l) to q (not in l) at
///   theta = phi_p / (phi_p + psi_q) from p, the linear zero crossing of
///   psi_l, with theta clamped to [1e-3, 1 - 1e-3]. psi_q = phi_q, except
///   with junctionCorrection on edges of a cell with three or more distinct
///   labels (label 0 included), where psi_q is re-estimated within its proven
///   bounds (see lsmo_internal::MeshContext::correctedOutside). Vertices are
///   functions of (edge, label) only, so cells sharing an edge agree.
/// - Re-triangulation: the table's triangulation is designed for vertices
///   at edge midpoints, and interpolated vertices can fold it. A
///   patch whose triangulation, with the interpolated vertices, has a
///   triangle of signed quality below 0.01 (folded against the patch's
///   vector area, or near-degenerate; for a tube, against the radial
///   direction of its diagonal) is re-triangulated: disks by the max-min
///   quality polygon triangulation, tubes by the max-min quality strip over
///   all start chords. Chords between two vertices on a common cube face are
///   never used, so neighboring cells cannot share an interior edge. Only the
///   triangulation of a loop changes, never which vertices and loops exist,
///   so the topology is the table's.
/// - Vertices are deduplicated per block with a hash map keyed by (edge,
///   label); vertices on edges shared with other blocks are merged in a final
///   serial pass in block order, so the output is identical for every grain
///   size.
/// - Normals (optional) are the normalized gradient of psi_l (central
///   differences) interpolated along the edge.
///
/// @par Guarantees
/// Each label's mesh is a closed, consistently wound (counter-clockwise seen
/// from outside) 2-manifold without zero-area triangles, with one component
/// per boundary of the region (outer boundaries plus enclosed cavities) and
/// Euler characteristic twice that of the digital region under the chosen
/// connectivity pair. Interfaces between labels are not shared: two touching
/// labels have coincident but separate surfaces.
///
/// Vertex positions, unlike the topology, depend on phi: when phi is not a
/// distance field near the interface (e.g. an unredistanced or noisy grid),
/// interpolated vertices can be placed inconsistently and a triangle can
/// fold (face against its patch), even after re-triangulation. The topology
/// guarantees above still hold.
///
/// @par References
/// - J.-O. Lachaud, A. Montanvert, "Continuous analogs of digital
///   boundaries: a topological approach to iso-surfaces", Graphical Models
///   62(3), 2000.
/// - W. E. Lorensen, H. E. Cline, "Marching cubes", SIGGRAPH 1987.
///
/// @par Cube numbering (independent of VolumeToMesh.h)
/// A cell is the 2x2x2 block of voxel centers with minimum corner ijk.
/// - Corner bit i (0..7) is the voxel ijk + (i&1, (i>>1)&1, (i>>2)&1); a case
///   index has bit i set when corner i lies in the label ("in-corner").
/// - Edges 0-3 run along x, 4-7 along y, 8-11 along z. Edge 4a+k (axis a) has
///   its two free coordinates, in ascending axis order, equal to
///   (k&1, k>>1); kCubeEdgeCorners lists (lower corner, upper corner):
///   @code
///   x: e0 (0,1)  e1 (2,3)  e2 (4,5)  e3 (6,7)
///   y: e4 (0,2)  e5 (1,3)  e6 (4,6)  e7 (5,7)
///   z: e8 (0,4)  e9 (1,5)  e10 (2,6) e11 (3,7)
///   @endcode
/// - Face f = 2a + s holds the corners whose coordinate a equals s; its
///   outward normal is (2s-1) along axis a.
///
/// @par Case-table rule
/// Triangles are triples of edge ids, counter-clockwise seen from outside
/// the label (normal from in-corners towards out-corners).
/// - Face rule: on a face with exactly two diagonal in-corners the in-pair is
///   joined (boundary segments cut off the two out-corners) when the
///   foreground adjacency is 18 or 26, and separated (segments cut off the
///   in-corners) when it is 6. Faces with two crossing edges have the single
///   segment between them. Since the segments depend only on the face's four
///   corner bits, neighboring cells always agree.
/// - Interior rule: two corners on a body diagonal that are not otherwise
///   connected are joined by a tube when their side uses 26-adjacency
///   (in-pairs for FG26_BG6, out-pairs for FG6_BG26), separated otherwise.
/// Within a cell the in-corners (out-corners) joined by the local surface
/// are exactly the components of the in-corners (out-corners) under the
/// cube-restricted foreground (background) adjacency, where 6 = cube edges,
/// 18 = + face diagonals and 26 = + body diagonals.

#ifndef OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_MESH_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_MESH_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>
#include <openvdb/tools/LevelSetMultiObject.h>
#include <openvdb/thread/Threading.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/util/NullInterrupter.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Settings of the multi-label mesher.
struct MultiObjectMeshSettings
{
    /// Digital (foreground, background) adjacency the meshes are consistent with.
    MultiObjectConnectivity connectivity = MultiObjectConnectivity::FG26_BG6;
    /// Re-estimate vertices at junctions of three or more labels, where phi
    /// may be the distance to a third label's boundary (see
    /// lsmo_internal::MeshContext::correctedOutside).
    bool junctionCorrection = true;
    /// With junctionCorrection, estimate those vertices from the closest
    /// point on a local planar reconstruction of the label's boundary
    /// (lsmo_internal::closestPointPsi); false uses one local Godunov step.
    bool closestPointJunctions = true;
    /// Also output per-vertex normals.
    bool computeNormals = false;
    /// TBB grain size in blocks of cells; zero or negative runs serially.
    /// The output does not depend on it.
    int grainSize = 1;
    /// phi is read as max(phi, minimumPhi) (world units; default 0). Pass
    /// the floor of a topology-preserving evolver
    /// (LevelSetMultiObject::phiFloor()) to mesh its state as it sees it.
    double minimumPhi = 0.0;
};


/// @brief Triangle mesh of one label.
struct LabelMesh
{
    std::vector<Vec3s> points;     ///< world space
    std::vector<Vec3I> triangles;  ///< counter-clockwise seen from outside the label
    std::vector<Vec3s> normals;    ///< per point, unit, outward; empty unless requested
};


namespace lsmo_internal {

/// Edge e of the cell joins corners kCubeEdgeCorners[e][0] < [e][1] (see the
/// file documentation for the numbering).
inline constexpr uint8_t kCubeEdgeCorners[12][2] = {
    {0, 1}, {2, 3}, {4, 5}, {6, 7},   // along x
    {0, 2}, {1, 3}, {4, 6}, {5, 7},   // along y
    {0, 4}, {1, 5}, {2, 6}, {3, 7}};  // along z

/// Triangulation of one case: @c count triangles of edge ids (0..11),
/// counter-clockwise seen from outside the label.
struct CaseTriangulation
{
    uint8_t count = 0;
    std::array<std::array<uint8_t, 3>, 16> tris{};
};

using CaseTable = std::array<CaseTriangulation, 256>;


/// @brief Builds the 256-case table for adjacency pair (fg, bg), each in
/// {6, 18, 26}, from the face and interior rules of the file documentation.
///
/// Per case: (1) directed boundary segments on each face (face rule),
/// oriented so that faceNormal x direction points from in to out;
/// (2) segments chained into closed loops (each crossing edge lies on two
/// faces); (3) the corners are grouped into regions connected along the cube
/// surface, each loop separating one in-region from one out-region; regions
/// of one side that must be joined under that side's adjacency (only body
/// diagonals under 26) are connected by replacing their two loop disks,
/// which bound a common region of the other side, with a triangle-strip
/// tube; (4) every other loop becomes a disk, a fan from a vertex chosen so
/// that no interior edge lies in a cube face. Throws ValueError if a case
/// cannot be resolved or needs more than 16 triangles.
inline CaseTable
generateCaseTable(int fg, int bg)
{
    // integer geometry: coordinates doubled so edge midpoints are integral
    struct V { int x[3]; };
    auto cornerPos = [](int c) { return V{{2 * (c & 1), 2 * ((c >> 1) & 1), 2 * ((c >> 2) & 1)}}; };
    auto midPos = [&](int e) {
        const V a = cornerPos(kCubeEdgeCorners[e][0]), b = cornerPos(kCubeEdgeCorners[e][1]);
        return V{{(a.x[0] + b.x[0]) / 2, (a.x[1] + b.x[1]) / 2, (a.x[2] + b.x[2]) / 2}};
    };
    auto edgeBetween = [](int a, int b) {
        for (int e = 0; e < 12; ++e)
            if ((kCubeEdgeCorners[e][0] == a && kCubeEdgeCorners[e][1] == b) ||
                (kCubeEdgeCorners[e][0] == b && kCubeEdgeCorners[e][1] == a)) return e;
        return -1;
    };
    // faces: f = 2a + s, corners in cyclic order
    int faceCorners[6][4];
    int edgeFaces[12] = {0};
    for (int a = 0; a < 3; ++a) {
        const int u = a == 0 ? 1 : 0, v = a == 2 ? 1 : 2;
        for (int s = 0; s < 2; ++s) {
            const int f = 2 * a + s, base = s << a;
            faceCorners[f][0] = base;
            faceCorners[f][1] = base | (1 << u);
            faceCorners[f][2] = base | (1 << u) | (1 << v);
            faceCorners[f][3] = base | (1 << v);
            for (int j = 0; j < 4; ++j)
                edgeFaces[edgeBetween(faceCorners[f][j], faceCorners[f][(j + 1) % 4])] |= 1 << f;
        }
    }
    auto adjacent = [](int a, int b, int n) {
        const int d = ((a ^ b) & 1) + (((a ^ b) >> 1) & 1) + (((a ^ b) >> 2) & 1);
        return d == 1 || (d == 2 && n >= 18) || (d == 3 && n == 26);
    };
    auto fail = [&](int caseIdx, const char* what) {
        std::ostringstream os;
        os << "multi-object case table (" << fg << "," << bg << "), case " << caseIdx
           << ": " << what;
        OPENVDB_THROW(ValueError, os.str());
    };

    CaseTable table;
    for (int caseIdx = 1; caseIdx < 255; ++caseIdx) {
        auto isIn = [&](int c) { return ((caseIdx >> c) & 1) != 0; };
        auto inCorner = [&](int e) {
            return isIn(kCubeEdgeCorners[e][0]) ? kCubeEdgeCorners[e][0] : kCubeEdgeCorners[e][1];
        };
        auto outCorner = [&](int e) {
            return isIn(kCubeEdgeCorners[e][0]) ? kCubeEdgeCorners[e][1] : kCubeEdgeCorners[e][0];
        };

        // corner regions along the cube surface (union-find)
        int rep[8];
        for (int i = 0; i < 8; ++i) rep[i] = i;
        auto find = [&](int x) { while (rep[x] != x) x = rep[x]; return x; };
        auto unite = [&](int a, int b) { rep[find(a)] = find(b); };
        for (int e = 0; e < 12; ++e)
            if (isIn(kCubeEdgeCorners[e][0]) == isIn(kCubeEdgeCorners[e][1]))
                unite(kCubeEdgeCorners[e][0], kCubeEdgeCorners[e][1]);

        // (1) directed face segments
        int next[12];
        for (int e = 0; e < 12; ++e) next[e] = -1;
        auto addSegment = [&](int f, int e0, int e1) {
            // orient so that faceNormal x (m1 - m0) points from in to out
            const V m0 = midPos(e0), m1 = midPos(e1);
            const int a = f / 2, sgn = (f & 1) ? 1 : -1;
            const int t[3] = {m1.x[0] - m0.x[0], m1.x[1] - m0.x[1], m1.x[2] - m0.x[2]};
            int side[3] = {0, 0, 0};             // (sgn * axis_a) x t
            const int b = (a + 1) % 3, c = (a + 2) % 3;
            side[c] = sgn * t[b];
            side[b] = -sgn * t[c];
            const V pi = cornerPos(inCorner(e0)), po = cornerPos(outCorner(e0));
            int d = 0;
            for (int k = 0; k < 3; ++k) d += side[k] * (po.x[k] - pi.x[k]);
            if (d == 0) fail(caseIdx, "degenerate face segment");
            if (d < 0) std::swap(e0, e1);
            if (next[e0] != -1) fail(caseIdx, "edge with two outgoing segments");
            next[e0] = e1;
        };
        for (int f = 0; f < 6; ++f) {
            const int* fc = faceCorners[f];
            int crossing[4], nc = 0;
            for (int j = 0; j < 4; ++j)
                if (isIn(fc[j]) != isIn(fc[(j + 1) % 4]))
                    crossing[nc++] = edgeBetween(fc[j], fc[(j + 1) % 4]);
            if (nc == 2) {
                addSegment(f, crossing[0], crossing[1]);
            } else if (nc == 4) {
                // ambiguous face: fg 18/26 joins the in-pair (cut off the
                // out-corners), fg 6 joins the out-pair (cut off the in-corners)
                const bool cutIn = (fg == 6);
                for (int j = 0; j < 4; ++j) {
                    if (isIn(fc[j]) != cutIn) continue;
                    addSegment(f, edgeBetween(fc[(j + 3) % 4], fc[j]),
                        edgeBetween(fc[j], fc[(j + 1) % 4]));
                }
                // the corners not cut off are joined across the face
                const int keep = cutIn ? 0 : 1;
                for (int j = 0; j < 4; ++j)
                    if (int(isIn(fc[j])) == keep) unite(fc[j], fc[(j + 2) % 4]);
            }
        }

        // (2) loops
        struct Loop { std::vector<int> v; int inRegion, outRegion; bool disk = true; };
        std::vector<Loop> loops;
        {
            bool seen[12] = {false};
            for (int e = 0; e < 12; ++e) {
                if (next[e] < 0 || seen[e]) continue;
                Loop loop;
                for (int cur = e; !seen[cur]; cur = next[cur]) {
                    if (next[cur] < 0) fail(caseIdx, "open loop");
                    seen[cur] = true;
                    loop.v.push_back(cur);
                }
                if (loop.v.front() != next[loop.v.back()]) fail(caseIdx, "malformed loop");
                loop.inRegion = find(inCorner(e));
                loop.outRegion = find(outCorner(e));
                loops.push_back(loop);
            }
        }
        // crossing edges must all be covered
        for (int e = 0; e < 12; ++e)
            if ((isIn(kCubeEdgeCorners[e][0]) != isIn(kCubeEdgeCorners[e][1])) != (next[e] >= 0))
                fail(caseIdx, "crossing edge without segment");

        std::vector<std::array<int, 3>> tris;

        // (3) tubes joining same-side regions required by that side's adjacency
        auto dist2 = [&](int e0, int e1) {
            const V a = midPos(e0), b = midPos(e1);
            int d = 0;
            for (int k = 0; k < 3; ++k) d += (a.x[k] - b.x[k]) * (a.x[k] - b.x[k]);
            return d;
        };
        auto addTube = [&](const std::vector<int>& A, const std::vector<int>& B) {
            // Strip between loop A (walked forward) and loop B (walked
            // backward); both keep their disk orientation, which is the
            // boundary orientation the annulus induces.
            const int m = int(A.size()), n = int(B.size());
            int k = 0;
            for (int j = 1; j < n; ++j) if (dist2(A[0], B[j]) < dist2(A[0], B[k])) k = j;
            int i = 0, j = 0; // steps taken along A and B
            while (i < m || j < n) {
                const int a0 = A[i % m], a1 = A[(i + 1) % m];
                const int b0 = B[((k - j) % n + n) % n], b1 = B[((k - j - 1) % n + n) % n];
                const bool advanceA = j >= n || (i < m && dist2(a1, b0) <= dist2(a0, b1));
                if (advanceA) { tris.push_back({a0, a1, b0}); ++i; }
                else          { tris.push_back({b1, b0, a0}); ++j; }
            }
        };
        for (int side = 0; side < 2; ++side) {
            const bool in = side == 0;
            const int n = in ? fg : bg;
            for (int a = 0; a < 8; ++a)
                for (int b = a + 1; b < 8; ++b) {
                    if (isIn(a) != in || isIn(b) != in || !adjacent(a, b, n)) continue;
                    if (find(a) == find(b)) continue;
                    // two still-disk loops bounding a's and b's regions and a
                    // common region of the other side
                    int la = -1, lb = -1;
                    for (size_t p = 0; p < loops.size() && lb < 0; ++p)
                        for (size_t q = 0; q < loops.size() && lb < 0; ++q) {
                            if (p == q || !loops[p].disk || !loops[q].disk) continue;
                            const Loop &P = loops[p], &Q = loops[q];
                            const int pr = in ? P.inRegion : P.outRegion;
                            const int qr = in ? Q.inRegion : Q.outRegion;
                            const int po = in ? P.outRegion : P.inRegion;
                            const int qo = in ? Q.outRegion : Q.inRegion;
                            if (find(pr) == find(a) && find(qr) == find(b)
                                && find(po) == find(qo)) {
                                la = int(p); lb = int(q);
                            }
                        }
                    if (lb < 0) fail(caseIdx, "no loop pair to join by a tube");
                    loops[la].disk = loops[lb].disk = false;
                    addTube(loops[la].v, loops[lb].v);
                    unite(a, b);
                }
        }

        // (4) disks: fan from a vertex none of whose chords lies in a face;
        // among those, the apex whose fan (with midpoint vertices) best keeps
        // every triangle normal along the loop's in-to-out direction
        for (const Loop& loop : loops) {
            if (!loop.disk) continue;
            const int n = int(loop.v.size());
            // in-to-out direction, scaled to integers: |I| sum(out) - |O| sum(in)
            int sumIn[3] = {0, 0, 0}, sumOut[3] = {0, 0, 0}, nIn = 0, nOut = 0;
            {
                int seenC = 0;
                for (int e : loop.v) {
                    for (int c : {inCorner(e), outCorner(e)}) {
                        if (seenC & (1 << c)) continue;
                        seenC |= 1 << c;
                        const V p = cornerPos(c);
                        int* s = isIn(c) ? sumIn : sumOut;
                        (isIn(c) ? nIn : nOut) += 1;
                        for (int k = 0; k < 3; ++k) s[k] += p.x[k];
                    }
                }
            }
            int dir[3];
            for (int k = 0; k < 3; ++k) dir[k] = nIn * sumOut[k] - nOut * sumIn[k];
            auto alignment = [&](int e0, int e1, int e2) {
                const V a = midPos(e0), b = midPos(e1), c = midPos(e2);
                const int u[3] = {b.x[0] - a.x[0], b.x[1] - a.x[1], b.x[2] - a.x[2]};
                const int w[3] = {c.x[0] - a.x[0], c.x[1] - a.x[1], c.x[2] - a.x[2]};
                const int nrm[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0]};
                return nrm[0] * dir[0] + nrm[1] * dir[1] + nrm[2] * dir[2];
            };
            int apex = -1, bestScore = 0;
            for (int r = 0; r < n; ++r) {
                bool ok = true;
                for (int s = 2; s < n - 1; ++s)
                    if (edgeFaces[loop.v[r]] & edgeFaces[loop.v[(r + s) % n]]) ok = false;
                if (!ok) continue;
                int score = std::numeric_limits<int>::max();
                for (int s = 1; s + 1 < n; ++s)
                    score = std::min(score, alignment(loop.v[r], loop.v[(r + s) % n],
                        loop.v[(r + s + 1) % n]));
                if (apex < 0 || score > bestScore) { apex = r; bestScore = score; }
            }
            if (apex < 0) fail(caseIdx, "no valid fan apex");
            for (int s = 1; s + 1 < n; ++s)
                tris.push_back({loop.v[apex], loop.v[(apex + s) % n], loop.v[(apex + s + 1) % n]});
        }

        if (tris.size() > 16) fail(caseIdx, "more than 16 triangles");
        CaseTriangulation& ct = table[caseIdx];
        ct.count = uint8_t(tris.size());
        for (size_t t = 0; t < tris.size(); ++t)
            for (int k = 0; k < 3; ++k) ct.tris[t][k] = uint8_t(tris[t][k]);
    }
    return table;
}


/// @brief The case table for a connectivity pair, generated on first use
/// (thread-safe function-local statics).
inline const CaseTable&
caseTable(MultiObjectConnectivity conn)
{
    switch (conn) {
        case MultiObjectConnectivity::FG26_BG6: {
            static const CaseTable t = generateCaseTable(26, 6);
            return t;
        }
        case MultiObjectConnectivity::FG6_BG26: {
            static const CaseTable t = generateCaseTable(6, 26);
            return t;
        }
        case MultiObjectConnectivity::FG18_BG6: {
            static const CaseTable t = generateCaseTable(18, 6);
            return t;
        }
        case MultiObjectConnectivity::FG6_BG18: {
            static const CaseTable t = generateCaseTable(6, 18);
            return t;
        }
    }
    OPENVDB_THROW(ValueError, "unknown MultiObjectConnectivity");
}

////////////////////////////////////////
// Contouring


/// Bit f is set when edge @a e lies on cube face f (f = 2*axis + side).
inline uint8_t
cubeEdgeFaces(int e)
{
    const int a = e / 4, k = e % 4;
    const int b = a == 0 ? 1 : 0, c = a == 2 ? 1 : 2;
    return uint8_t((1 << (2 * b + (k & 1))) | (1 << (2 * c + (k >> 1))));
}

/// @brief One surface patch of a case: a disk bounded by one loop or a tube
/// (annulus) bounded by two, with the table's triangles.
/// @details Loops list edge ids in boundary order, i.e. the order in which
/// the patch's counter-clockwise triangles traverse them.
struct CasePatch
{
    std::vector<std::array<uint8_t, 3>> tris;
    std::vector<uint8_t> loop[2];
    int tubeAxis[2] = {-1, -1};  ///< tube: the two antipodal corners it surrounds
    bool tubeIn = false;         ///< tube around two in-corners (else two out-corners)
    bool isTube() const { return !loop[1].empty(); }
};

using CasePatches = std::array<std::vector<CasePatch>, 256>;

/// @brief Splits every case of @a table into its patches (connected
/// components of triangles sharing an edge) and their boundary loops.
/// @throw ValueError if a patch is neither a disk nor a tube around a body
/// diagonal (the table generator never produces one).
inline CasePatches
buildCasePatches(const CaseTable& table)
{
    CasePatches out;
    for (int c = 1; c < 255; ++c) {
        const CaseTriangulation& ct = table[c];
        const int n = ct.count;
        int rep[16];
        for (int i = 0; i < n; ++i) rep[i] = i;
        auto find = [&](int x) { while (rep[x] != x) x = rep[x]; return x; };
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                int shared = 0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) shared += ct.tris[i][a] == ct.tris[j][b];
                if (shared >= 2) rep[find(i)] = find(j);
            }
        for (int root = 0; root < n; ++root) {
            if (find(root) != root) continue;
            CasePatch patch;
            for (int i = 0; i < n; ++i)
                if (find(i) == root) patch.tris.push_back(ct.tris[i]);
            auto hasDirected = [&](int a, int b) {
                for (const auto& t : patch.tris)
                    for (int k = 0; k < 3; ++k)
                        if (t[k] == a && t[(k + 1) % 3] == b) return true;
                return false;
            };
            int next[12];
            for (int e = 0; e < 12; ++e) next[e] = -1;
            for (const auto& t : patch.tris)
                for (int k = 0; k < 3; ++k) {
                    const int a = t[k], b = t[(k + 1) % 3];
                    if (hasDirected(b, a)) continue;
                    if (next[a] != -1) {
                        OPENVDB_THROW(ValueError, "multi-object mesh: non-manifold case patch");
                    }
                    next[a] = b;
                }
            bool seen[12] = {false};
            int nLoops = 0;
            for (int e = 0; e < 12; ++e) {
                if (next[e] < 0 || seen[e]) continue;
                if (nLoops == 2) {
                    OPENVDB_THROW(ValueError, "multi-object mesh: case patch with > 2 loops");
                }
                for (int cur = e; !seen[cur]; cur = next[cur]) {
                    seen[cur] = true;
                    patch.loop[nLoops].push_back(uint8_t(cur));
                }
                ++nLoops;
            }
            if (patch.isTube()) {
                int ins = 0, outs = 0;
                for (const auto& t : patch.tris)
                    for (int k = 0; k < 3; ++k) {
                        const int c0 = kCubeEdgeCorners[t[k]][0], c1 = kCubeEdgeCorners[t[k]][1];
                        const bool in0 = (c >> c0) & 1;
                        ins |= 1 << (in0 ? c0 : c1);
                        outs |= 1 << (in0 ? c1 : c0);
                    }
                auto antipodal = [](int mask, int* pair) {
                    int m = 0;
                    for (int i = 0; i < 8; ++i) {
                        if (mask & (1 << i)) { if (m < 2) pair[m] = i; ++m; }
                    }
                    return m == 2 && (pair[0] ^ pair[1]) == 7;
                };
                if (antipodal(ins, patch.tubeAxis)) patch.tubeIn = true;
                else if (!antipodal(outs, patch.tubeAxis)) {
                    OPENVDB_THROW(ValueError,
                        "multi-object mesh: tube patch without a diagonal axis");
                }
            }
            out[c].push_back(std::move(patch));
        }
    }
    return out;
}

/// The patches of caseTable(@a conn), built on first use.
inline const CasePatches&
casePatches(MultiObjectConnectivity conn)
{
    switch (conn) {
        case MultiObjectConnectivity::FG26_BG6: {
            static const CasePatches p = buildCasePatches(caseTable(conn));
            return p;
        }
        case MultiObjectConnectivity::FG6_BG26: {
            static const CasePatches p = buildCasePatches(caseTable(conn));
            return p;
        }
        case MultiObjectConnectivity::FG18_BG6: {
            static const CasePatches p = buildCasePatches(caseTable(conn));
            return p;
        }
        case MultiObjectConnectivity::FG6_BG18: {
            static const CasePatches p = buildCasePatches(caseTable(conn));
            return p;
        }
    }
    OPENVDB_THROW(ValueError, "unknown MultiObjectConnectivity");
}

/// Vertices stay at least this fraction of a cell edge away from its corners
/// (so no triangle collapses when phi vanishes at a voxel center).
constexpr double kMeshThetaMin = 1e-3;

/// A patch whose table triangulation has a triangle of quality below this
/// is re-triangulated (see the file documentation).
constexpr double kMeshMinQuality = 1e-2;

/// @brief Signed triangle quality 2*sqrt(3)*(n . ref) / (sum of squared edge
/// lengths), with n = (b - a) x (c - a) and @a ref a unit reference normal:
/// 1 for an equilateral triangle facing @a ref, 0 for a degenerate one,
/// negative for a triangle folded against @a ref.
inline double
triangleQuality(const Vec3d& a, const Vec3d& b, const Vec3d& c, const Vec3d& ref)
{
    const Vec3d n = (b - a).cross(c - a);
    const double s = (b - a).lengthSqr() + (c - b).lengthSqr() + (a - c).lengthSqr();
    if (!(s > 0.0)) return -1.0;
    return 2.0 * std::sqrt(3.0) * n.dot(ref) / s;
}

/// @brief Triangulation of the loop of edge ids @a e[0..n-1] (n <= 12) that
/// maximizes the minimum of quality(i, j, k) over its triangles (i < j < k
/// loop positions, wound in loop order); chords between two vertices on a
/// common cube face are not used (they could coincide with a neighboring
/// cell's edge). O(n^3) dynamic programming.
/// @return the minimum quality, or -infinity if no triangulation exists
/// (never for table loops, whose fan is one).
template<typename QualityFn>
inline double
bestDiskTriangulation(const uint8_t* e, int n, const QualityFn& quality,
    std::vector<std::array<int, 3>>& tris)
{
    constexpr double kNone = -std::numeric_limits<double>::infinity();
    double best[12][12];
    int arg[12][12];
    auto chordOk = [&](int i, int j) {
        return j == i + 1 || (i == 0 && j == n - 1)
            || (cubeEdgeFaces(e[i]) & cubeEdgeFaces(e[j])) == 0;
    };
    for (int i = 0; i + 1 < n; ++i) {
        best[i][i + 1] = std::numeric_limits<double>::infinity();
        arg[i][i + 1] = -1;
    }
    for (int len = 2; len < n; ++len) {
        for (int i = 0; i + len < n; ++i) {
            const int j = i + len;
            best[i][j] = kNone;
            arg[i][j] = -1;
            for (int k = i + 1; k < j; ++k) {
                if (!chordOk(i, k) || !chordOk(k, j)) continue;
                if (best[i][k] == kNone || best[k][j] == kNone) continue;
                const double v = std::min({best[i][k], best[k][j], quality(i, k, j)});
                if (arg[i][j] < 0 || v > best[i][j]) { best[i][j] = v; arg[i][j] = k; }
            }
        }
    }
    tris.clear();
    if (n < 3 || arg[0][n - 1] < 0) return kNone;
    std::vector<std::pair<int, int>> stack{{0, n - 1}};
    while (!stack.empty()) {
        const auto ij = stack.back();
        stack.pop_back();
        const int k = arg[ij.first][ij.second];
        if (k < 0) continue;
        tris.push_back({ij.first, k, ij.second});
        stack.push_back({ij.first, k});
        stack.push_back({k, ij.second});
    }
    return best[0][n - 1];
}

/// @brief Triangle strip between loops @a A (m edges) and @a B (n edges) of
/// a tube that maximizes the minimum quality, over every start chord.
/// @details Vertices are numbered 0..m-1 for A and m..m+n-1 for B. Walking
/// forward along A and backward along B, each triangle is either
/// (A[i], A[i+1], B[j]) or (B[j-1], B[j], A[i]), which keeps the winding of
/// both loops. Chords between vertices on a common cube face are not used.
/// A path state (s, t) is the chord (A[i0+s], B[j0-t]), so (0, t) and
/// (m, t) are the same chord, as are (s, 0) and (s, n): a path with m
/// consecutive A steps from s = 0 or n consecutive B steps from t = 0 would
/// use a chord twice (a pinched strip). The search state therefore records
/// whether the current run of A steps started at s = 0 and whether the
/// current run of B steps started at t = 0, and such full runs are
/// forbidden. As a backstop, a strip that still repeats a chord is
/// discarded.
/// @return the minimum quality, or -infinity if no strip exists (the caller
/// then keeps the table strip).
template<typename QualityFn>
inline double
bestTubeTriangulation(const uint8_t* A, int m, const uint8_t* B, int n, const QualityFn& quality,
    std::vector<std::array<int, 3>>& tris)
{
    constexpr double kNone = -std::numeric_limits<double>::infinity();
    auto chordOk = [&](int a, int b) { return (cubeEdgeFaces(A[a]) & cubeEdgeFaces(B[b])) == 0; };
    double bestQ = kNone;
    // f[s][t][k]: best minimum quality reaching state (s, t) with flags
    // k = 2*ra + rb; ra: the path is on a row it entered at s = 0 (A steps
    // from there would reach s = m on that row), rb: the path is on a column
    // it entered at t = 0.
    double f[13][13][4];
    int8_t from[13][13][4]; // previous flags*2 + step (0: A step, 1: B step), -1 none
    std::vector<std::array<int, 3>> cand;
    for (int i0 = 0; i0 < m; ++i0) {
        for (int j0 = 0; j0 < n; ++j0) {
            if (!chordOk(i0, j0)) continue;
            auto ai = [&](int s) { return (i0 + s) % m; };
            auto bj = [&](int t) { return ((j0 - t) % n + n) % n; };
            for (int s = 0; s <= m; ++s) {
                for (int t = 0; t <= n; ++t) {
                    for (int k = 0; k < 4; ++k) { f[s][t][k] = kNone; from[s][t][k] = -1; }
                    if (s == 0 && t == 0) {
                        f[0][0][3] = std::numeric_limits<double>::infinity();
                        continue;
                    }
                    if (!chordOk(ai(s), bj(t))) continue;
                    for (int k = 0; k < 4; ++k) {
                        // A step from (s-1, t, k): ra kept, rb set iff t == 0
                        if (s > 0 && f[s - 1][t][k] != kNone) {
                            const int ra = k >> 1;
                            if (!(s == m && ra)) { // m A steps from column 0
                                const int nk = 2 * ra + (t == 0 ? 1 : 0);
                                const double v = std::min(f[s - 1][t][k],
                                    quality(ai(s - 1), ai(s), m + bj(t)));
                                if (from[s][t][nk] < 0 || v > f[s][t][nk]) {
                                    f[s][t][nk] = v;
                                    from[s][t][nk] = int8_t(2 * k);
                                }
                            }
                        }
                        // B step from (s, t-1, k): rb kept, ra set iff s == 0
                        if (t > 0 && f[s][t - 1][k] != kNone) {
                            const int rb = k & 1;
                            if (!(t == n && rb)) { // n B steps from row 0
                                const int nk = 2 * (s == 0 ? 1 : 0) + rb;
                                const double v = std::min(f[s][t - 1][k],
                                    quality(m + bj(t), m + bj(t - 1), ai(s)));
                                if (from[s][t][nk] < 0 || v > f[s][t][nk]) {
                                    f[s][t][nk] = v;
                                    from[s][t][nk] = int8_t(2 * k + 1);
                                }
                            }
                        }
                    }
                }
            }
            int bk = -1;
            for (int k = 0; k < 4; ++k) {
                if (from[m][n][k] >= 0 && (bk < 0 || f[m][n][k] > f[m][n][bk])) bk = k;
            }
            if (bk < 0 || !(f[m][n][bk] > bestQ)) continue;
            // trace back, checking that no chord repeats (backstop)
            cand.clear();
            std::set<std::pair<int, int>> chords{{ai(m), bj(n)}};
            bool repeated = false;
            for (int s = m, t = n, k = bk; s > 0 || t > 0;) {
                const int code = from[s][t][k];
                if (code % 2 == 0) { cand.push_back({ai(s - 1), ai(s), m + bj(t)}); --s; }
                else { cand.push_back({m + bj(t), m + bj(t - 1), ai(s)}); --t; }
                k = code / 2;
                if (s > 0 || t > 0) repeated |= !chords.insert({ai(s), bj(t)}).second;
            }
            if (repeated) continue;
            bestQ = f[m][n][bk];
            tris = cand;
        }
    }
    return bestQ;
}

/// Counters of one contouring run (for tests and reports).
struct MeshStats
{
    size_t patches = 0;         ///< (cell, label) surface patches
    /// patches whose table triangulation was replaced
    size_t retriangulated = 0;
    /// patches left with a triangle of quality <= 0
    size_t folded = 0;
    /// vertex placements moved by junction correction (a vertex on an edge
    /// shared by two blocks counts twice)
    size_t corrected = 0;

    MeshStats& operator+=(const MeshStats& o)
    {
        patches += o.patches; retriangulated += o.retriangulated;
        folded += o.folded; corrected += o.corrected;
        return *this;
    }
};

/// Output of one block of cells, in block-local vertex numbering.
struct MeshBlock
{
    struct Vertex
    {
        int32_t label;
        Coord lo;       ///< lower end of the cell edge
        uint8_t axis;   ///< direction of the cell edge
        bool shared;    ///< the edge also belongs to cells of other blocks
        Vec3s point;    ///< world space
        Vec3s normal;
    };
    struct Triangle { int32_t label; uint32_t v[3]; };
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    MeshStats stats;
};

/// @brief Per-task contouring state: accessors and settings.
template<typename FloatTreeT, typename LabelTreeT>
struct MeshContext
{
    static_assert(std::is_same<typename FloatTreeT::ValueType, float>::value,
        "the distance tree must have float values");
    using LabelT = typename LabelTreeT::ValueType;

    MeshContext(const FloatTreeT& phiTree, const LabelTreeT& labelTree,
        const math::Transform& xf, const MultiObjectMeshSettings& s,
        const std::vector<LabelT>* req)
        : phi(phiTree), lab(labelTree), xform(xf), settings(s), requested(req)
        , patches(casePatches(s.connectivity)), dx(xf.voxelSize()[0]) {}

    /// phi at @a x, at least settings.minimumPhi.
    double phiAt(const Coord& x)
    {
        return std::max(double(phi.getValue(x)), settings.minimumPhi);
    }

    /// psi_l(x): -phi inside label l, +phi elsewhere (labels via getValue).
    double psi(const Coord& x, LabelT l)
    {
        const double v = phiAt(x);
        return lab.getValue(x) == l ? -v : v;
    }

    /// True if the 18 voxels of the four cells around the edge
    /// (lo, lo + e_axis), i.e. the union of their corners (2 along the
    /// axis by 3x3 across it), hold three or more distinct labels. This is
    /// slightly wider than "some cell around the edge has three labels", so
    /// a few edges next to a junction are corrected too; the correction is
    /// a function of the edge's endpoint and label only, so it stays
    /// consistent between cells.
    bool junctionEdge(const Coord& lo, int axis)
    {
        const int b = axis == 0 ? 1 : 0, c = axis == 2 ? 1 : 2;
        LabelT seen[2] = {0, 0};
        int n = 0;
        for (int da = 0; da <= 1; ++da)
            for (int db = -1; db <= 1; ++db)
                for (int dc = -1; dc <= 1; ++dc) {
                    Coord x = lo;
                    x[axis] += da; x[b] += db; x[c] += dc;
                    const LabelT v = lab.getValue(x);
                    if ((n > 0 && v == seen[0]) || (n > 1 && v == seen[1])) continue;
                    if (n == 2) return true;
                    seen[n++] = v;
                }
        return false;
    }

    /// @brief Junction-corrected psi_l at the endpoint @a q outside label
    /// @a l of a junction edge.
    /// @details phi_q is the distance to the boundary of q's own label, which
    /// next to a third label may be that label's boundary rather than l's.
    /// As in MultiLevelSet::extractSignedDistance two bounds hold:
    /// psi_l(q) >= phi_q, and psi_l(q) <= |q - p| - phi_p for
    /// every voxel p of label l (its ball of radius phi_p lies in l); the
    /// latter is taken over the 5x5x5 neighborhood of q. With
    /// settings.closestPointJunctions (the default) the estimate is
    /// closestPointPsi (the closest point on a local planar reconstruction
    /// of l's boundary, within those bounds) where a plane decides it
    /// (ClosestPointInfo::planeDecided); otherwise it is one
    /// local Godunov step on psi_l (per axis the smaller psi_l of the two
    /// face neighbors, -phi in l and +phi elsewhere), clamped to
    /// [phi_q, max(phi_q, upper bound)]. It depends only on q and l, so every
    /// cell sharing the edge places the same vertex.
    double correctedOutside(const Coord& q, LabelT l, double phiQ)
    {
        if (settings.closestPointJunctions) {
            ClosestPointInfo info;
            const double v = closestPointPsi(phi, lab, q, int32_t(l), dx, settings.minimumPhi,
                &info);
            if (info.planeDecided) return v;
        }
        double upper = std::numeric_limits<double>::max();
        for (int k = -2; k <= 2; ++k)
            for (int j = -2; j <= 2; ++j)
                for (int i = -2; i <= 2; ++i) {
                    const Coord p = q.offsetBy(i, j, k);
                    if (lab.getValue(p) != l) continue;
                    upper = std::min(upper,
                        dx * std::sqrt(double(i * i + j * j + k * k)) - phiAt(p));
                }
        double abar[3], coef[3] = {1.0, 1.0, 1.0};
        int m = 0;
        for (int axis = 0; axis < 3; ++axis) {
            Coord lo = q, hi = q;
            lo[axis] -= 1;
            hi[axis] += 1;
            abar[m++] = std::min(psi(lo, l), psi(hi, l));
        }
        const double est = solveGodunov(abar, coef, m, dx);
        return std::min(std::max(est, phiQ), std::max(upper, phiQ));
    }

    tree::ValueAccessor<const FloatTreeT> phi;
    tree::ValueAccessor<const LabelTreeT> lab;
    const math::Transform& xform;
    const MultiObjectMeshSettings& settings;
    const std::vector<LabelT>* requested;
    const CasePatches& patches;
    double dx;
};

/// @brief Contours the cells whose minimum corner lies in the block
/// [o, o + dim - 1]^3.
template<typename FloatTreeT, typename LabelTreeT>
void
contourBlock(MeshContext<FloatTreeT, LabelTreeT>& ctx, const Coord& o, int dim, MeshBlock& out)
{
    using LabelT = typename LabelTreeT::ValueType;
    const int P = dim + 1;
    std::vector<LabelT> L(size_t(P) * P * P);
    auto at = [P](int x, int y, int z) { return (size_t(x) * P + y) * P + z; };
    bool uniform = true;
    for (int x = 0; x < P; ++x)
        for (int y = 0; y < P; ++y)
            for (int z = 0; z < P; ++z) {
                L[at(x, y, z)] = ctx.lab.getValue(o.offsetBy(x, y, z));
                uniform = uniform && L[at(x, y, z)] == L[0];
            }
    if (uniform) return;

    std::unordered_map<uint64_t, uint32_t> vertexIds;
    std::vector<Vec3d> indexPos; // index-space position of every block vertex
    const bool normals = ctx.settings.computeNormals;
    const bool correct = ctx.settings.junctionCorrection;

    // vertex of label l on the edge (lo, lo + e_axis), lo block-local
    auto vertex = [&](int lx, int ly, int lz, int axis, LabelT l, bool cellJunction,
                      Vec3d& pos) -> uint32_t {
        const uint64_t key = (uint64_t(uint32_t(l)) << 32)
            | uint64_t(((size_t(lx) * P + ly) * P + lz) * 3 + axis);
        auto it = vertexIds.find(key);
        const Coord lo = o.offsetBy(lx, ly, lz);
        if (it != vertexIds.end()) {
            pos = indexPos[it->second];
            return it->second;
        }
        Coord hi = lo;
        hi[axis] += 1;
        const bool loIn = ctx.lab.getValue(lo) == l;
        const Coord p = loIn ? lo : hi, q = loIn ? hi : lo;
        const double phiP = ctx.phiAt(p);
        double psiQ = ctx.phiAt(q);
        if (correct && (cellJunction || ctx.junctionEdge(lo, axis))) {
            const double c = ctx.correctedOutside(q, l, psiQ);
            if (c != psiQ) ++out.stats.corrected;
            psiQ = c;
        }
        double theta = phiP + psiQ > 0.0 ? phiP / (phiP + psiQ) : 0.5;
        theta = std::min(std::max(theta, kMeshThetaMin), 1.0 - kMeshThetaMin);
        pos = p.asVec3d() + (q.asVec3d() - p.asVec3d()) * theta;
        MeshBlock::Vertex v;
        v.label = int32_t(l);
        v.lo = lo;
        v.axis = uint8_t(axis);
        const int b = axis == 0 ? 1 : 0, c = axis == 2 ? 1 : 2;
        const int lb = (b == 0 ? lx : (b == 1 ? ly : lz)), lc = (c == 1 ? ly : lz);
        v.shared = lb == 0 || lb == dim || lc == 0 || lc == dim;
        v.point = Vec3s(ctx.xform.indexToWorld(pos));
        v.normal = Vec3s(0.0f);
        if (normals) {
            auto grad = [&](const Coord& x) {
                Vec3d g;
                for (int a = 0; a < 3; ++a) {
                    Coord m = x, n = x;
                    m[a] -= 1; n[a] += 1;
                    g[a] = 0.5 * (ctx.psi(n, l) - ctx.psi(m, l));
                }
                return g;
            };
            Vec3d g = grad(p) * (1.0 - theta) + grad(q) * theta;
            g = ctx.xform.baseMap()->applyIJT(g);
            if (!(g.length() > 0.0)) g = ctx.xform.baseMap()->applyIJT(q.asVec3d() - p.asVec3d());
            v.normal = Vec3s(g.unit());
        }
        const uint32_t id = uint32_t(out.vertices.size());
        out.vertices.push_back(v);
        indexPos.push_back(pos);
        vertexIds.emplace(key, id);
        return id;
    };

    std::vector<std::array<int, 3>> dpTris;
    for (int x = 0; x < dim; ++x)
        for (int y = 0; y < dim; ++y)
            for (int z = 0; z < dim; ++z) {
                LabelT cl[8];
                bool mixed = false;
                for (int c = 0; c < 8; ++c) {
                    cl[c] = L[at(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1))];
                    mixed = mixed || cl[c] != cl[0];
                }
                if (!mixed) continue;
                // distinct labels of the cell, sorted ascending (insertion
                // sort: at most 8 values, and std::sort on a stack array draws
                // a false -Warray-bounds from GCC 14)
                LabelT uniq[8];
                int nu = 0;
                for (int c = 0; c < 8; ++c) {
                    bool found = false;
                    for (int u = 0; u < nu && !found; ++u) found = uniq[u] == cl[c];
                    if (found) continue;
                    int pos = nu++;
                    while (pos > 0 && uniq[pos - 1] > cl[c]) { uniq[pos] = uniq[pos - 1]; --pos; }
                    uniq[pos] = cl[c];
                }
                const bool cellJunction = nu >= 3;
                const Vec3d cellMin = o.offsetBy(x, y, z).asVec3d();
                for (int u = 0; u < nu; ++u) {
                    const LabelT l = uniq[u];
                    if (l == LabelT(0)) continue;
                    if (ctx.requested && !std::binary_search(
                            ctx.requested->begin(), ctx.requested->end(), l)) continue;
                    int caseIdx = 0;
                    for (int c = 0; c < 8; ++c) caseIdx |= int(cl[c] == l) << c;
                    for (const CasePatch& patch : ctx.patches[caseIdx]) {
                        ++out.stats.patches;
                        // vertices of the patch, by edge id
                        uint32_t vid[12];
                        Vec3d pos[12];
                        for (int k = 0; k < 2; ++k)
                            for (uint8_t e : patch.loop[k]) {
                                const int c0 = kCubeEdgeCorners[e][0];
                                vid[e] = vertex(x + (c0 & 1), y + ((c0 >> 1) & 1),
                                    z + ((c0 >> 2) & 1), e / 4, l, cellJunction, pos[e]);
                            }
                        // reference normal of the patch, for re-triangulation
                        Vec3d diskRef(0.0), axisLo, axisDir;
                        if (patch.isTube()) {
                            auto corner = [&](int c) {
                                return cellMin + Vec3d(c & 1, (c >> 1) & 1, (c >> 2) & 1);
                            };
                            axisLo = corner(patch.tubeAxis[0]);
                            axisDir = (corner(patch.tubeAxis[1]) - axisLo).unit();
                        } else {
                            const auto& lp = patch.loop[0];
                            const Vec3d& p0 = pos[lp[0]];
                            for (size_t i = 1; i + 1 < lp.size(); ++i)
                                diskRef += (pos[lp[i]] - p0).cross(pos[lp[i + 1]] - p0);
                            if (diskRef.length() > 0.0) diskRef.normalize();
                        }
                        auto quality = [&](int e0, int e1, int e2) {
                            const Vec3d &a = pos[e0], &b = pos[e1], &c = pos[e2];
                            if (!patch.isTube()) return triangleQuality(a, b, c, diskRef);
                            const Vec3d cen = (a + b + c) / 3.0 - axisLo;
                            Vec3d radial = cen - axisDir * axisDir.dot(cen);
                            if (radial.length() > 0.0) radial.normalize();
                            return triangleQuality(a, b, c, patch.tubeIn ? radial : -radial);
                        };
                        double minQ = std::numeric_limits<double>::infinity();
                        for (const auto& t : patch.tris)
                            minQ = std::min(minQ, quality(t[0], t[1], t[2]));
                        bool replaced = false;
                        if (minQ < kMeshMinQuality) {
                            double q;
                            if (patch.isTube()) {
                                const auto &A = patch.loop[0], &B = patch.loop[1];
                                const int m = int(A.size());
                                auto edgeOf = [&](int i) { return i < m ? A[i] : B[i - m]; };
                                q = bestTubeTriangulation(A.data(), m, B.data(), int(B.size()),
                                    [&](int i, int j, int k) {
                                        return quality(edgeOf(i), edgeOf(j), edgeOf(k)); }, dpTris);
                                for (auto& t : dpTris) for (int& i : t) i = edgeOf(i);
                            } else {
                                const auto& lp = patch.loop[0];
                                q = bestDiskTriangulation(lp.data(), int(lp.size()),
                                    [&](int i, int j, int k) {
                                        return quality(lp[i], lp[j], lp[k]); }, dpTris);
                                for (auto& t : dpTris) for (int& i : t) i = lp[i];
                            }
                            if (q > minQ) {
                                replaced = true;
                                minQ = q;
                                ++out.stats.retriangulated;
                            }
                        }
                        if (!(minQ > 0.0)) ++out.stats.folded;
                        if (replaced) {
                            for (const auto& t : dpTris)
                                out.triangles.push_back(
                                    {int32_t(l), {vid[t[0]], vid[t[1]], vid[t[2]]}});
                        } else {
                            for (const auto& t : patch.tris)
                                out.triangles.push_back(
                                    {int32_t(l), {vid[t[0]], vid[t[1]], vid[t[2]]}});
                        }
                    }
                }
            }
}

/// Key of a vertex shared between blocks.
struct MeshEdgeKey
{
    Coord lo;
    int32_t label;
    uint8_t axis;
    bool operator==(const MeshEdgeKey& o) const
    {
        return lo == o.lo && label == o.label && axis == o.axis;
    }
};

struct MeshEdgeKeyHash
{
    size_t operator()(const MeshEdgeKey& k) const
    {
        size_t h = std::hash<Coord>()(k.lo);
        h ^= (size_t(uint32_t(k.label)) * 0x9E3779B97F4A7C15ull) + size_t(k.axis)
            + (h << 6) + (h >> 2);
        return h;
    }
};

/// @brief multiObjectToMesh with optional statistics.
template<typename MultiLevelSetT, typename InterruptT>
std::map<int32_t, LabelMesh>
contourMultiObject(const MultiLevelSetT& mls, const std::vector<int32_t>& labels,
    const MultiObjectMeshSettings& settings, InterruptT* interrupter, MeshStats* stats)
{
    using FloatTreeT = typename MultiLevelSetT::FloatGridT::TreeType;
    using LabelTreeT = typename MultiLevelSetT::LabelGridT::TreeType;
    using LabelT = typename LabelTreeT::ValueType;

    const FloatTreeT& phiTree = mls.phi()->tree();
    const LabelTreeT& labelTree = mls.labels()->tree();
    const math::Transform& xform = mls.phi()->transform();

    std::map<int32_t, LabelMesh> result;
    std::vector<LabelT> requested;
    for (int32_t l : labels) {
        if (l == 0) continue;
        result[l];
        // a label the tree cannot hold has an empty mesh (never truncated)
        if (lsmo_internal::labelFits<LabelT>(l)) requested.push_back(LabelT(l));
    }
    std::sort(requested.begin(), requested.end());
    requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
    const bool all = labels.empty();
    if (!all && requested.empty()) return result; // nothing the label tree can hold

    // blocks: every leaf of the label tree and its lower neighbors (cells
    // whose minimum corner is one voxel below a leaf have corners in it)
    const int dim = int(LabelTreeT::LeafNodeType::DIM);
    std::vector<Coord> blocks;
    for (auto leaf = labelTree.cbeginLeaf(); leaf; ++leaf) {
        const Coord o = leaf->origin();
        for (int d = 0; d < 8; ++d)
            blocks.push_back(o.offsetBy(
                -dim * (d & 1), -dim * ((d >> 1) & 1), -dim * ((d >> 2) & 1)));
    }
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());

    if (interrupter) interrupter->start("Contouring multi-object level set");
    std::vector<MeshBlock> results(blocks.size());
    std::atomic<bool> interrupted(false);
    const bool parallel = settings.grainSize > 0;
    auto body = [&](const tbb::blocked_range<size_t>& r) {
        MeshContext<FloatTreeT, LabelTreeT> ctx(phiTree, labelTree, xform, settings,
            all ? nullptr : &requested);
        for (size_t i = r.begin(); i != r.end(); ++i) {
            if (util::wasInterrupted(interrupter)) {
                interrupted = true;
                if (parallel) thread::cancelGroupExecution();
                return;
            }
            contourBlock(ctx, blocks[i], dim, results[i]);
        }
    };
    if (parallel) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, blocks.size(), size_t(settings.grainSize)), body);
    } else {
        body(tbb::blocked_range<size_t>(0, blocks.size()));
    }
    if (interrupted) {
        if (interrupter) interrupter->end();
        return {};
    }

    // serial merge in block order (deterministic for any grain size)
    const bool flip = xform.baseMap()->determinant() < 0.0;
    std::unordered_map<MeshEdgeKey, uint32_t, MeshEdgeKeyHash> shared;
    MeshStats total;
    std::vector<uint32_t> remap;
    for (MeshBlock& block : results) {
        total += block.stats;
        remap.resize(block.vertices.size());
        int32_t lastLabel = 0;
        LabelMesh* mesh = nullptr;
        auto meshOf = [&](int32_t l) -> LabelMesh& {
            if (!mesh || l != lastLabel) { mesh = &result[l]; lastLabel = l; }
            return *mesh;
        };
        for (size_t i = 0; i < block.vertices.size(); ++i) {
            const MeshBlock::Vertex& v = block.vertices[i];
            LabelMesh& m = meshOf(v.label);
            if (v.shared) {
                auto ins = shared.emplace(MeshEdgeKey{v.lo, v.label, v.axis},
                    uint32_t(m.points.size()));
                if (!ins.second) { remap[i] = ins.first->second; continue; }
            }
            remap[i] = uint32_t(m.points.size());
            m.points.push_back(v.point);
            if (settings.computeNormals) m.normals.push_back(v.normal);
        }
        for (const MeshBlock::Triangle& t : block.triangles) {
            LabelMesh& m = meshOf(t.label);
            if (flip) m.triangles.emplace_back(remap[t.v[0]], remap[t.v[2]], remap[t.v[1]]);
            else m.triangles.emplace_back(remap[t.v[0]], remap[t.v[1]], remap[t.v[2]]);
        }
        MeshBlock().vertices.swap(block.vertices);
        MeshBlock().triangles.swap(block.triangles);
    }
    if (stats) *stats = total;
    if (interrupter) interrupter->end();
    return result;
}


} // namespace lsmo_internal


/// @brief Contours every requested label of a multi-object level set into a
/// closed, consistently wound 2-manifold triangle mesh whose topology matches
/// the label's digital region under settings.connectivity.
/// @details See the file documentation.
/// @param mls     the phi / label pair
/// @param labels  labels to mesh; empty meshes every non-zero label that has a
///                boundary. Label 0 is never meshed. A requested label always
///                has a key in the result (with an empty mesh if absent).
/// @param settings connectivity, junction correction, normals, grain size
/// @param interrupter checked once per block of cells; if it fires, the
///                result is empty
template<typename MultiLevelSetT, typename InterruptT = util::NullInterrupter>
std::map<int32_t, LabelMesh>
multiObjectToMesh(const MultiLevelSetT& mls, const std::vector<int32_t>& labels = {},
    const MultiObjectMeshSettings& settings = {}, InterruptT* interrupter = nullptr)
{
    return lsmo_internal::contourMultiObject(mls, labels, settings, interrupter, nullptr);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVEL_SET_MULTI_OBJECT_MESH_HAS_BEEN_INCLUDED
