//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan_unittests.cpp
//  \brief Problem generator for unit tests of utils/scan/line_scan.hpp
//
//  Prefix and suffix scans in every direction are compared, in every cell of
//  scan_data on every rank, with host references built from the mesh layout and
//  physical coordinates, independently of line_scan.hpp:
//   - on uniform meshes, an integer field whose sums are exact
//   - a linear field, whose upstream sums are exact integrals at any
//     refinement
//   - nonlinear fields, checked against a brute force implementation of the
//     restriction and prolongation rules
//  The scans of the linear and nonlinear fields run concurrently and are reused
//  on new fields, after a forced rebuild of the exchange plan, and with AMR
//  again in the final function after the mesh has changed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
// The mesh headers must come before eos and hydro, whose headers need complete
// Mesh types, so they are kept in their own block
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"

#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "utils/scan/line_scan.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif  // MPI_PARALLEL_ENABLED

namespace line_scan_test {
using line_scan::Direction;
using line_scan::ScanKind;
using line_scan::Stage;

// Ghost cells hold a huge value, so a scan that reads them fails
constexpr Real kGhostValue = 1.0e30;

const char* DirName(const Direction d) {
  if (d == Direction::I) return "I";
  if (d == Direction::J) return "J";
  return "K";
}

const char* KindName(const ScanKind kind) {
  return (kind == ScanKind::Prefix) ? "prefix" : "suffix";
}

// Sums a check's mismatches over the ranks and reports them. Returns the sum
int Report(const std::string& name, int nerr) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nerr, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif  // MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    std::cout << "LineScan " << name << ": "
              << ((nerr == 0) ? "passed" : "FAILED") << " (" << nerr
              << " mismatches)" << std::endl;
  }
  return nerr;
}

// The axes (0, 1, 2 for x1, x2, x3) along a scan and across its faces. Faces
// are indexed [t2 * n_t1 + t1]
struct Axes {
  int s, t1, t2;
};

constexpr Axes AxesOf(const Direction d) {
  if (d == Direction::I) return {0, 1, 2};
  if (d == Direction::J) return {1, 0, 2};
  return {2, 0, 1};
}

//----------------------------------------------------------------------------------------
// Exact scans on uniform meshes

// The global index of each MeshBlock's first real cell along x1, x2, x3
std::array<std::int64_t, 3> FirstCell(const Mesh* pm, const int gid) {
  const LogicalLocation& lloc = pm->lloc_eachmb[gid];
  const RegionIndcs& indcs = pm->mb_indcs;
  return {std::int64_t{lloc.lx1} * indcs.nx1, std::int64_t{lloc.lx2} * indcs.nx2,
          std::int64_t{lloc.lx3} * indcs.nx3};
}

// Integers on the global cell indices, so every sum is exact
constexpr Real kCoef[3] = {1.0, 7.0, 13.0};
Real CellValue(const std::array<std::int64_t, 3>& g) {
  return 1.0 + kCoef[0] * g[0] + kCoef[1] * g[1] + kCoef[2] * g[2];
}

// Runs one scan of q and compares every cell of scan_data exactly with the
// global exclusive scan of CellValue. Returns the mismatches
template <Direction Dir, ScanKind Kind>
int CheckExactScan(MeshBlockPack* pmbp, const DvceArray4D<Real>& q) {
  auto value_func = KOKKOS_LAMBDA(const int m, const int k, const int j,
                                  const int i, const RegionSize&)
                        ->Real {
    return q(m, k, j, i);
  };
  line_scan::LineScan<Dir, Kind, decltype(value_func)> scan(pmbp, value_func);
  // A sentinel catches cells the scan never writes
  Kokkos::deep_copy(scan.scan_data, -1.0);
  Kokkos::fence();
  while (scan.Driver() != Stage::Completed) {
  }
  auto h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scan.scan_data);

  const Mesh* pm = pmbp->pmesh;
  const RegionIndcs& indcs = pm->mb_indcs;
  const int s = AxesOf(Dir).s;
  const int nx[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
  const int nroot[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  const std::int64_t n_global = std::int64_t{nroot[s]} * nx[s];

  int nerr = 0;
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const std::array<std::int64_t, 3> first = FirstCell(pm, pmbp->gids + m);
    for (int k = 0; k < indcs.nx3; ++k) {
      for (int j = 0; j < indcs.nx2; ++j) {
        for (int i = 0; i < indcs.nx1; ++i) {
          std::array<std::int64_t, 3> g = {first[0] + i, first[1] + j,
                                           first[2] + k};
          // The sum of the arithmetic series over the global cells upstream
          const std::int64_t a = (Kind == ScanKind::Prefix) ? 0 : g[s] + 1;
          const std::int64_t b = (Kind == ScanKind::Prefix) ? g[s] : n_global;
          g[s] = 0;
          const Real expected = (b - a) * CellValue(g) +
                                kCoef[s] * ((a + b - 1) * (b - a) / 2);
          if (h(m, k, j, i) != expected) {
            if (nerr < 10) {
              std::cout << "  mismatch at m=" << m << " k=" << k << " j=" << j
                        << " i=" << i << ": got " << h(m, k, j, i)
                        << ", expected " << expected << std::endl;
            }
            ++nerr;
          }
        }
      }
    }
  }

  return Report(std::string("exact ") + DirName(Dir) + " " + KindName(Kind),
                nerr);
}

// Exact scans of the integer field, read through a KOKKOS_LAMBDA. Returns the
// mismatches on all ranks
int CheckExact(Mesh* pm) {
  MeshBlockPack* pmbp = pm->pmb_pack;
  const RegionIndcs& indcs = pm->mb_indcs;
  const int nc1 = indcs.nx1 + 2 * indcs.ng;
  const int nc2 = (indcs.nx2 > 1) ? indcs.nx2 + 2 * indcs.ng : 1;
  const int nc3 = (indcs.nx3 > 1) ? indcs.nx3 + 2 * indcs.ng : 1;

  // Indexed like u0 with ghost cells, which hold a huge value so a scan that
  // reads them fails
  DvceArray4D<Real> q("line_scan_test_q", pmbp->nmb_thispack, nc3, nc2, nc1);
  auto h = Kokkos::create_mirror_view(q);
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const auto first = FirstCell(pm, pmbp->gids + m);
    for (int k = 0; k < nc3; ++k) {
      for (int j = 0; j < nc2; ++j) {
        for (int i = 0; i < nc1; ++i) {
          const bool real = (i >= indcs.is && i <= indcs.ie && j >= indcs.js &&
                             j <= indcs.je && k >= indcs.ks && k <= indcs.ke);
          h(m, k, j, i) = real ? CellValue({first[0] + i - indcs.is,
                                            first[1] + j - indcs.js,
                                            first[2] + k - indcs.ks})
                               : kGhostValue;
        }
      }
    }
  }
  Kokkos::deep_copy(q, h);
  // The scans run on their own instances, so q must be ready before they start
  Kokkos::fence();

  return CheckExactScan<Direction::I, ScanKind::Prefix>(pmbp, q) +
         CheckExactScan<Direction::I, ScanKind::Suffix>(pmbp, q) +
         CheckExactScan<Direction::J, ScanKind::Prefix>(pmbp, q) +
         CheckExactScan<Direction::J, ScanKind::Suffix>(pmbp, q) +
         CheckExactScan<Direction::K, ScanKind::Prefix>(pmbp, q) +
         CheckExactScan<Direction::K, ScanKind::Suffix>(pmbp, q);
}

//----------------------------------------------------------------------------------------
// Scans across MeshBlocks and ranks

// Tolerance relative to the column (upstream sum plus line total)
constexpr Real kRelTol = 1024 * std::numeric_limits<Real>::epsilon();
// Fills scan_data before each run, so entries a scan doesn't write are caught
constexpr Real kSentinel = std::numeric_limits<Real>::lowest();

// The value_func of every scan: the field in u0(IDN), which persists across
// mesh refinement, times the cell width along the scan
template <Direction Dir>
struct FieldTimesDx {
  DvceArray5D<Real> u0;

  KOKKOS_INLINE_FUNCTION
  Real operator()(const int m, const int k, const int j, const int i,
                  const RegionSize& size) const {
    if constexpr (Dir == Direction::I) {
      return u0(m, IDN, k, j, i) * size.dx1;
    } else if constexpr (Dir == Direction::J) {
      return u0(m, IDN, k, j, i) * size.dx2;
    } else {
      return u0(m, IDN, k, j, i) * size.dx3;
    }
  }
};

template <Direction Dir, ScanKind Kind>
using Scan = line_scan::LineScan<Dir, Kind, FieldTimesDx<Dir>>;

// Prefix and suffix scans in every direction. They are constructed, and ForEach
// visits them, in the same order on every rank, as LineScan requires
struct ScanSet {
  explicit ScanSet(MeshBlockPack* pmbp)
      : i_prefix(pmbp, {pmbp->phydro->u0}),
        i_suffix(pmbp, {pmbp->phydro->u0}),
        j_prefix(pmbp, {pmbp->phydro->u0}),
        j_suffix(pmbp, {pmbp->phydro->u0}),
        k_prefix(pmbp, {pmbp->phydro->u0}),
        k_suffix(pmbp, {pmbp->phydro->u0}) {}

  template <typename F>
  void ForEach(F&& f) {
    f(i_prefix);
    f(i_suffix);
    f(j_prefix);
    f(j_suffix);
    f(k_prefix);
    f(k_suffix);
  }

  Scan<Direction::I, ScanKind::Prefix> i_prefix;
  Scan<Direction::I, ScanKind::Suffix> i_suffix;
  Scan<Direction::J, ScanKind::Prefix> j_prefix;
  Scan<Direction::J, ScanKind::Suffix> j_suffix;
  Scan<Direction::K, ScanKind::Prefix> k_prefix;
  Scan<Direction::K, ScanKind::Suffix> k_suffix;
};

//----------------------------------------------------------------------------------------
// Host geometry and fields

// A MeshBlock's cells and physical extent along each axis. An unused dimension
// has one cell spanning the Mesh
struct BlockGeometry {
  int level;
  int n[3];
  Real lo[3], hi[3], dx[3];

  Real Center(const int axis, const int c) const {
    return CellCenterX(c, n[axis], lo[axis], hi[axis]);
  }
};

BlockGeometry Geometry(const Mesh* pm, const int gid) {
  const LogicalLocation& lloc = pm->lloc_eachmb[gid];
  const RegionSize& ms = pm->mesh_size;
  const int lx[3] = {lloc.lx1, lloc.lx2, lloc.lx3};
  const int nroot[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  const int ncells[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  const Real xmin[3] = {ms.x1min, ms.x2min, ms.x3min};
  const Real xmax[3] = {ms.x1max, ms.x2max, ms.x3max};

  BlockGeometry geo;
  geo.level = lloc.level;
  for (int a = 0; a < 3; ++a) {
    const int nblocks =
        (ncells[a] > 1) ? (nroot[a] << (lloc.level - pm->root_level)) : 1;
    geo.n[a] = ncells[a];
    geo.lo[a] =
        (lx[a] == 0) ? xmin[a] : LeftEdgeX(lx[a], nblocks, xmin[a], xmax[a]);
    geo.hi[a] = (lx[a] == nblocks - 1)
                    ? xmax[a]
                    : LeftEdgeX(lx[a] + 1, nblocks, xmin[a], xmax[a]);
    geo.dx[a] = (geo.hi[a] - geo.lo[a]) / ncells[a];
  }
  return geo;
}

// A field of physical position
using Field = std::function<Real(const Real x[3])>;

// f of the position normalized to [0, 1] across the Mesh, so the inputs can
// use any domain
Field Normalized(const Mesh* pm, const Field& f) {
  const RegionSize ms = pm->mesh_size;
  return [ms, f](const Real x[3]) {
    const Real xn[3] = {(x[0] - ms.x1min) / (ms.x1max - ms.x1min),
                        (x[1] - ms.x2min) / (ms.x2max - ms.x2min),
                        (x[2] - ms.x3min) / (ms.x3max - ms.x3min)};
    return f(xn);
  };
}

// Positive and linear, so prolongation and area averaging are exact for it
Field Linear(const Mesh* pm) {
  return Normalized(pm, [](const Real x[3]) {
    return 2.0 + 0.5 * x[0] - 0.7 * x[1] + 0.3 * x[2];
  });
}

// Positive, with a kink at y = y0 and near zero cells next to large ones, so
// the line totals exercise the limiter, one sided slopes and positivity clamp
Field Nonlinear(const Mesh* pm, const Real scale, const Real eps, const Real c,
                const Real y0, const Real amp, const std::array<Real, 3> q) {
  return Normalized(pm, [=](const Real x[3]) {
    Real waves = amp;
    for (int d = 0; d < 3; ++d) {
      waves *= std::pow(std::sin(Kokkos::numbers::pi * q[d] * x[d]), 2);
    }
    return scale * (eps + c * std::fabs(x[1] - y0) + waves);
  });
}

Field NonlinearA(const Mesh* pm) {
  return Nonlinear(pm, 1.0, 0.003, 0.1, 0.37, 1.0, {8.3, 13.1, 11.7});
}

Field NonlinearB(const Mesh* pm) {
  return Nonlinear(pm, 1.0e3, 0.01, 0.3, 0.61, 1.5, {7.1, 12.7, 10.3});
}

//----------------------------------------------------------------------------------------
// Host references

// The value of every cell, and the expected upstream sums of a field
class Reference {
 public:
  Reference(const Mesh* pm, Field field) : pm_(pm), field_(std::move(field)) {
    for (int g = 0; g < pm->nmb_total; ++g) {
      geo_.push_back(Geometry(pm, g));
    }
  }
  virtual ~Reference() = default;

  // The field times dx along the scan at cell c (from the first real cell)
  Real Cell(const int gid, const int c[3], const int scan_axis) const {
    const BlockGeometry& geo = geo_[gid];
    const Real x[3] = {geo.Center(0, c[0]), geo.Center(1, c[1]),
                       geo.Center(2, c[2])};
    return field_(x) * geo.dx[scan_axis];
  }

  // The expected upstream face of MeshBlock gid
  virtual std::vector<Real> Upstream(int gid, const Axes& ax, bool prefix) = 0;

 protected:
  const Mesh* pm_;
  Field field_;
  std::vector<BlockGeometry> geo_;
};

// For a linear field the upstream sum is the field's integral from the domain
// edge to the MeshBlock, at the face cell center (the midpoint rule is exact)
class LinearReference : public Reference {
 public:
  using Reference::Reference;

  std::vector<Real> Upstream(const int gid, const Axes& ax,
                             const bool prefix) override {
    const BlockGeometry& geo = geo_[gid];
    const RegionSize& ms = pm_->mesh_size;
    const Real xmin[3] = {ms.x1min, ms.x2min, ms.x3min};
    const Real xmax[3] = {ms.x1max, ms.x2max, ms.x3max};
    const Real s0 = prefix ? xmin[ax.s] : geo.hi[ax.s];
    const Real s1 = prefix ? geo.lo[ax.s] : xmax[ax.s];

    const int n1 = geo.n[ax.t1], n2 = geo.n[ax.t2];
    std::vector<Real> face(n1 * n2);
    for (int t2 = 0; t2 < n2; ++t2) {
      for (int t1 = 0; t1 < n1; ++t1) {
        Real x[3];
        x[ax.s] = 0.5 * (s0 + s1);
        x[ax.t1] = geo.Center(ax.t1, t1);
        x[ax.t2] = geo.Center(ax.t2, t2);
        face[t2 * n1 + t1] = (s1 - s0) * field_(x);
      }
    }
    return face;
  }
};

// Brute force upstream sums for any field, by physical position. Every block C
// overlapping the target B across the scan and wholly upstream of it adds its
// line totals mapped to B's face:
//  - C as fine or finer: each C cell adds its total times its share of the
//    area of the B cell containing its center
//  - C coarser: a linear reconstruction at each B cell center, with minmod
//    slopes, one sided at C's face edges, scaled so it can't go negative
class MappedReference : public Reference {
 public:
  using Reference::Reference;

  std::vector<Real> Upstream(const int gid, const Axes& ax,
                             const bool prefix) override {
    const std::vector<std::vector<Real>>& totals = Totals(ax);
    const BlockGeometry& b = geo_[gid];
    std::vector<Real> face(b.n[ax.t1] * b.n[ax.t2], 0.0);
    for (int g = 0; g < pm_->nmb_total; ++g) {
      const BlockGeometry& c = geo_[g];
      if (Overlap(c, b, ax.t1) && Overlap(c, b, ax.t2) &&
          IsBefore(c, b, ax.s, prefix)) {
        if (c.level >= b.level) {
          AddRestricted(c, totals[g], b, ax, face);
        } else {
          AddProlongated(c, totals[g], b, ax, face);
        }
      }
    }
    return face;
  }

 private:
  // The tolerances absorb round off in the edge positions
  static bool Overlap(const BlockGeometry& a, const BlockGeometry& b,
                      const int axis) {
    const Real tol = 0.25 * std::min(a.dx[axis], b.dx[axis]);
    return a.lo[axis] < b.hi[axis] - tol && b.lo[axis] < a.hi[axis] - tol;
  }

  // Whether a lies wholly upstream of b
  static bool IsBefore(const BlockGeometry& a, const BlockGeometry& b,
                       const int axis, const bool prefix) {
    const Real tol = 0.25 * std::min(a.dx[axis], b.dx[axis]);
    return prefix ? a.hi[axis] <= b.lo[axis] + tol
                  : a.lo[axis] >= b.hi[axis] - tol;
  }

  // Every MeshBlock's line totals along ax.s, computed once per axis
  const std::vector<std::vector<Real>>& Totals(const Axes& ax) {
    std::vector<std::vector<Real>>& totals = totals_[ax.s];
    if (!totals.empty()) return totals;
    for (int g = 0; g < pm_->nmb_total; ++g) {
      const BlockGeometry& geo = geo_[g];
      std::vector<Real> face(geo.n[ax.t1] * geo.n[ax.t2]);
      for (int t2 = 0; t2 < geo.n[ax.t2]; ++t2) {
        for (int t1 = 0; t1 < geo.n[ax.t1]; ++t1) {
          int c[3];
          c[ax.t1] = t1;
          c[ax.t2] = t2;
          Real sum = 0.0;
          for (c[ax.s] = 0; c[ax.s] < geo.n[ax.s]; ++c[ax.s]) {
            sum += Cell(g, c, ax.s);
          }
          face[t2 * geo.n[ax.t1] + t1] = sum;
        }
      }
      totals.push_back(face);
    }
    return totals;
  }

  void AddRestricted(const BlockGeometry& src, const std::vector<Real>& total,
                     const BlockGeometry& dst, const Axes& ax,
                     std::vector<Real>& face) {
    const Real area_fraction = (src.dx[ax.t1] / dst.dx[ax.t1]) *
                               (src.dx[ax.t2] / dst.dx[ax.t2]);
    for (int c2 = 0; c2 < src.n[ax.t2]; ++c2) {
      for (int c1 = 0; c1 < src.n[ax.t1]; ++c1) {
        // The target cell containing the source cell's center
        const int i1 = static_cast<int>(std::floor(
            (src.Center(ax.t1, c1) - dst.lo[ax.t1]) / dst.dx[ax.t1]));
        const int i2 = static_cast<int>(std::floor(
            (src.Center(ax.t2, c2) - dst.lo[ax.t2]) / dst.dx[ax.t2]));
        face[i2 * dst.n[ax.t1] + i1] +=
            area_fraction * total[c2 * src.n[ax.t1] + c1];
      }
    }
  }

  void AddProlongated(const BlockGeometry& src, const std::vector<Real>& total,
                      const BlockGeometry& dst, const Axes& ax,
                      std::vector<Real>& face) {
    const int t[2] = {ax.t1, ax.t2};
    auto u = [&](const int c[2]) { return total[c[1] * src.n[ax.t1] + c[0]]; };

    for (int i2 = 0; i2 < dst.n[ax.t2]; ++i2) {
      for (int i1 = 0; i1 < dst.n[ax.t1]; ++i1) {
        const Real x[2] = {dst.Center(ax.t1, i1), dst.Center(ax.t2, i2)};
        // The source cell containing x, and x's position in it, from -1/2 to
        // 1/2
        int c[2];
        Real xi[2];
        for (int d = 0; d < 2; ++d) {
          c[d] =
              static_cast<int>(std::floor((x[d] - src.lo[t[d]]) / src.dx[t[d]]));
          xi[d] = (x[d] - src.Center(t[d], c[d])) / src.dx[t[d]];
        }
        const Real u0 = u(c);

        Real slope[2];
        for (int d = 0; d < 2; ++d) {
          int left[2] = {c[0], c[1]}, right[2] = {c[0], c[1]};
          --left[d];
          ++right[d];
          const bool has_left = (c[d] > 0),
                     has_right = (c[d] < src.n[t[d]] - 1);
          const Real dl = has_left ? u0 - u(left) : 0.0;
          const Real dr = has_right ? u(right) - u0 : 0.0;
          if (has_left && has_right) {
            slope[d] =
                (dl * dr <= 0.0)
                    ? 0.0
                    : std::copysign(std::min(std::fabs(dl), std::fabs(dr)), dl);
          } else {
            slope[d] = has_left ? dl : dr;
          }
        }
        // The reconstruction's minimum is u0 - (|slope0| + |slope1|) / 2
        const Real slope_sum = std::fabs(slope[0]) + std::fabs(slope[1]);
        if (slope_sum > 2.0 * u0) {
          slope[0] *= 2.0 * u0 / slope_sum;
          slope[1] *= 2.0 * u0 / slope_sum;
        }
        face[i2 * dst.n[ax.t1] + i1] +=
            u0 + slope[0] * xi[0] + slope[1] * xi[1];
      }
    }
  }

  std::vector<std::vector<Real>> totals_[3];
};

//----------------------------------------------------------------------------------------
// Running and checking the scans

// Sets u0(IDN) to the field in the real cells and kGhostValue in the ghosts
void FillField(MeshBlockPack* pmbp, const Field& field) {
  auto& u0 = pmbp->phydro->u0;
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u0);
  auto size = pmbp->pmb->mb_size.h_view;
  const RegionIndcs& indcs = pmbp->pmesh->mb_indcs;
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int k = 0; k < static_cast<int>(h.extent(2)); ++k) {
      for (int j = 0; j < static_cast<int>(h.extent(3)); ++j) {
        for (int i = 0; i < static_cast<int>(h.extent(4)); ++i) {
          const bool real = (i >= indcs.is && i <= indcs.ie && j >= indcs.js &&
                             j <= indcs.je && k >= indcs.ks && k <= indcs.ke);
          const Real x[3] = {
              CellCenterX(i - indcs.is, indcs.nx1, size(m).x1min,
                          size(m).x1max),
              CellCenterX(j - indcs.js, indcs.nx2, size(m).x2min,
                          size(m).x2max),
              CellCenterX(k - indcs.ks, indcs.nx3, size(m).x3min,
                          size(m).x3max)};
          h(m, IDN, k, j, i) = real ? field(x) : kGhostValue;
        }
      }
    }
  }
  Kokkos::deep_copy(u0, h);
  // The scans run on their own execution space instances
  Kokkos::fence();
}

// Compares every entry of scan_data on every rank with ref and reports the
// result. Returns the mismatches on all ranks
template <Direction Dir, ScanKind Kind>
int CheckGlobalScan(const Scan<Dir, Kind>& scan, const Mesh* pm,
                    const std::string& check, Reference& ref) {
  const std::string name =
      check + " " + DirName(Dir) + " " + KindName(Kind);
  constexpr Axes ax = AxesOf(Dir);
  constexpr bool prefix = (Kind == ScanKind::Prefix);
  auto h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scan.scan_data);
  const int n[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  const int ns = n[ax.s], n1 = n[ax.t1], n2 = n[ax.t2];

  int nerr = 0;
  std::vector<Real> expected(ns);
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    const int gid = pm->pmb_pack->gids + m;
    const std::vector<Real> in = ref.Upstream(gid, ax, prefix);
    for (int t2 = 0; t2 < n2; ++t2) {
      for (int t1 = 0; t1 < n1; ++t1) {
        int c[3];
        c[ax.t1] = t1;
        c[ax.t2] = t2;
        // The global scan starts from the upstream sum
        Real sum = in[t2 * n1 + t1];
        for (int p = 0; p < ns; ++p) {
          c[ax.s] = prefix ? p : ns - 1 - p;
          expected[c[ax.s]] = sum;
          sum += ref.Cell(gid, c, ax.s);
        }
        const Real tol = kRelTol * std::fabs(sum);

        for (c[ax.s] = 0; c[ax.s] < ns; ++c[ax.s]) {
          const Real got = h(m, c[2], c[1], c[0]);
          if (!(std::fabs(got - expected[c[ax.s]]) <= tol)) {
            if (nerr < 5) {
              std::cout << "  rank " << global_variable::my_rank << " " << name
                        << ": mismatch at gid=" << gid << " cell=" << c[ax.s]
                        << " t1=" << t1 << " t2=" << t2 << ": got " << got
                        << ", expected " << expected[c[ax.s]] << std::endl;
            }
            ++nerr;
          }
        }
      }
    }
  }
  return Report(name, nerr);
}

// Sets u0 to the field, drives all the scans concurrently and checks them
// against ref. Returns the mismatches on all ranks
int Check(ScanSet& scans, Mesh* pm, const std::string& check, Reference& ref,
          const Field& field) {
  FillField(pm->pmb_pack, field);
  scans.ForEach([&](auto& scan) {
    scan.Reset();
    Kokkos::deep_copy(scan.scan_data, kSentinel);
  });
  Kokkos::fence();

  // Poll round-robin until all complete, as LineScan requires
  bool done = false;
  while (!done) {
    done = true;
    scans.ForEach(
        [&](auto& scan) { done &= (scan.Driver() == Stage::Completed); });
  }

  int nerr = 0;
  scans.ForEach(
      [&](auto& scan) { nerr += CheckGlobalScan(scan, pm, check, ref); });
  return nerr;
}

//----------------------------------------------------------------------------------------
// The checks

// Kept alive until the final function in AMR runs, to check reuse after the
// mesh changes
std::unique_ptr<ScanSet> column_scans;
int mesh_seq_at_pgen = 0;

// The linear field, whose run builds the plan for a new or changed mesh, then
// nonlinear field A, which reuses it
int CheckLinearAndNonlinear(Mesh* pm) {
  const Field linear = Linear(pm), nonlinear = NonlinearA(pm);
  LinearReference linear_ref(pm, linear);
  MappedReference nonlinear_ref(pm, nonlinear);
  return Check(*column_scans, pm, "linear", linear_ref, linear) +
         Check(*column_scans, pm, "nonlinear", nonlinear_ref, nonlinear);
}

void Finish(const int nerr, const std::string& when) {
  if (global_variable::my_rank == 0) {
    std::cout << "LineScan unit test "
              << ((nerr == 0) ? "passed " : "FAILED ") << when << std::endl;
  }
  if (nerr != 0) {
    column_scans.reset();
    std::exit(EXIT_FAILURE);
  }
}

// AMR criterion: refine inside a sphere (in normalized coordinates) that moves
// along x1 every cycle and derefine outside it, so MeshBlocks move between
// ranks
void RefineMovingRegion(MeshBlockPack* pmbp) {
  Mesh* pm = pmbp->pmesh;
  auto& refine_flag = pm->pmr->refine_flag;
  auto size = pmbp->pmb->mb_size.h_view;
  const RegionSize& ms = pm->mesh_size;
  const Real center[3] = {0.2 + 0.12 * pm->ncycle, 0.5, 0.5};
  const Real radius = 0.25;
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const Real x[3] = {(0.5 * (size(m).x1min + size(m).x1max) - ms.x1min) /
                           (ms.x1max - ms.x1min),
                       (0.5 * (size(m).x2min + size(m).x2max) - ms.x2min) /
                           (ms.x2max - ms.x2min),
                       (0.5 * (size(m).x3min + size(m).x3max) - ms.x3min) /
                           (ms.x3max - ms.x3min)};
    Real r2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      r2 += (x[d] - center[d]) * (x[d] - center[d]);
    }
    const int gid = pmbp->gids + m;
    const int level = pm->lloc_eachmb[gid].level;
    if (r2 < radius * radius && level < pm->max_level) {
      refine_flag.h_view(gid) = 1;
    } else if (r2 >= radius * radius && level > pm->root_level) {
      refine_flag.h_view(gid) = -1;
    }
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

void CheckInitialMesh(Mesh* pm) {
  MeshBlockPack* pmbp = pm->pmb_pack;
  // A fluid at rest with uniform pressure, valid whatever the density, for
  // AMR runs that evolve it
  auto& u0 = pmbp->phydro->u0;
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Kokkos::deep_copy(u0, 0.0);
  Kokkos::deep_copy(Kokkos::subview(u0, Kokkos::ALL, static_cast<int>(IEN),
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
                    1.0 / gm1);

  int nerr = pm->multilevel ? 0 : CheckExact(pm);
  column_scans = std::make_unique<ScanSet>(pmbp);
  nerr += CheckLinearAndNonlinear(pm);

  // Reuse on another field after forcing the exchange plans to be rebuilt
  pm->MarkMeshUpdated();
  const Field nonlinear = NonlinearB(pm);
  MappedReference ref(pm, nonlinear);
  nerr += Check(*column_scans, pm, "nonlinear rebuilt", ref, nonlinear);
  Finish(nerr, "on the initial mesh");
  mesh_seq_at_pgen = pm->GetAMRLoadBalanceUpdateSeq();
}

// Final function of AMR runs: reruns the scans on the refined mesh
void CheckAfterAMR(ParameterInput* pin, Mesh* pm) {
  int nerr = 0;
  if (pm->GetAMRLoadBalanceUpdateSeq() == mesh_seq_at_pgen) {
    if (global_variable::my_rank == 0) {
      std::cout << "LineScan unit test: the mesh was never refined"
                << std::endl;
    }
    ++nerr;
  }
  nerr += CheckLinearAndNonlinear(pm);
  column_scans.reset();
  Finish(nerr, "after AMR");
}
}  // namespace line_scan_test

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::LineScan()
//! \brief Unit tests of the line scans. Exits with EXIT_FAILURE on any mismatch.
//! With AMR the final function checks them again after the mesh has changed.

void ProblemGenerator::LineScan(ParameterInput* pin, const bool restart) {
  namespace test = line_scan_test;
  if (restart) return;
  test::CheckInitialMesh(pmy_mesh_);
  // LineScans must be destroyed before MPI and Kokkos are finalized
  if (pmy_mesh_->adaptive) {
    user_ref_func = test::RefineMovingRegion;
    pgen_final_func = test::CheckAfterAMR;
  } else {
    test::column_scans.reset();
  }
}
