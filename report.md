# Point Cloud Local Histogram Equalization — Algorithm Reference

## Problem Statement

Given **N** 3D points, each with integer coordinates `(x, y, z)` and an intensity
value in `{0..255}`, apply *local* histogram equalization to each point using its
**k nearest neighbors** as the local neighborhood. Three methods are implemented,
all producing a per-point remapped intensity as output.

```
Input format:
  N  k  T
  x0 y0 z0 intensity0
  x1 y1 z1 intensity1
  ...

Constraints: N ≤ 1e5,  0 < k ≤ 128,  T ≤ 50,  coordinates are integers
```

---

## Shared Foundation — Histogram Equalization

All three modes reduce to the same equalization formula once a neighborhood of
k points has been identified. This formula is the backbone of every output value.

### The Formula

For a query point `q` with intensity `I_q` and neighbor set `N(q)` of size `k`:

```
1. Build histogram H[v] = count of neighbors with intensity v,  v ∈ {0..255}

2. CDF(v)    = Σ H[u]  for u = 0..v

3. CDF_min   = first non-zero value of CDF  (= CDF at the lowest occupied bin)

4.                  CDF(I_q) - CDF_min
   I_new  =  round( ─────────────────── × 255 )
                       k  -  CDF_min

   Special case: if k == CDF_min (all neighbors have the same intensity),
   return I_q unchanged to avoid division by zero.
```

This is the standard histogram equalization CDF mapping applied locally.
The denominator `k - CDF_min` is the effective dynamic range of the neighborhood.

### GPU Implementation

Each thread independently builds its own 256-bin histogram in thread-local
registers, then performs an inline prefix-sum CDF scan. No shared memory or
atomics are needed for this step — it is fully private per thread.

```
equalize_from_neighbors(pts, neighbor_indices[], k, self_intensity)
  → int new_intensity
```

Both KNN and approximate KNN reuse `equalize_knn_kernel` which calls this
function. K-means uses a separate `equalize_kmeans_kernel` because the
neighborhood is defined differently (cluster membership, not a neighbor list).

---

## Shared Foundation — Max-Heap for KNN

Both exact and approximate KNN maintain a **bounded max-heap of size k** per
thread. The heap root is always the *worst* current neighbor (farthest or
lexicographically largest on ties), so eviction is O(1) + O(log k) sift-down.

### Heap Ordering — Tiebreaking

Equal distances would produce non-deterministic neighbor sets, so a strict
total order is enforced:

```
A is worse than B  iff:
  dist²(A) > dist²(B)                           (primary:   farther = worse)
  OR  dist² equal AND A.x > B.x                 (secondary: larger x = worse)
  OR  x equal      AND A.y > B.y                (tertiary:  larger y = worse)
  OR  y equal      AND A.z > B.z                (quaternary: larger z = worse)
```

This gives lexicographically-smaller points priority in the neighborhood, making
the result fully deterministic even for datasets with many equidistant points.

### Distance Metric

```
dist²(A, B) = (A.x−B.x)² + (A.y−B.y)² + (A.z−B.z)²
```

Computed with `int64_t` arithmetic — no floating point involved, so there is no
rounding error and the tiebreaker is exact.

---

## Mode 1 — Exact KNN

### Goal

For every point `q_i`, find its exact k nearest neighbors among all other N−1
points, then apply histogram equalization using those k neighbors.

### Complexity

| Phase       | Time        | Space   |
|-------------|-------------|---------|
| KNN search  | O(N² log k) | O(N·k)  |
| Equalization| O(N·k)      | O(N)    |

### GPU Strategy — Tiled Shared Memory

A naïve approach would have every thread scan all N points independently from
global memory, which is wasteful: N threads reading the same N-point database
means N² global memory transactions with no reuse.

The tiled approach exploits the fact that all threads in a block are searching
the same database:

```
For each tile of BLOCK_SZ points from the database:
  1. All threads cooperatively load the tile into shared memory (one LDG each)
  2. Each thread scans all BLOCK_SZ points in the tile from shared memory
     → shared memory access is ~100× faster than global memory
  3. Threads synchronize before loading the next tile
```

Each of the N database points is loaded from global memory exactly once, into
shared memory, and then broadcast to all BLOCK_SZ threads in the block.
This reduces global memory traffic from O(N²) to O(N) loads at the cost of
O(N/BLOCK_SZ) kernel synchronization points per query.

```
Grid layout:  ceil(N / BLOCK_SZ) blocks × BLOCK_SZ threads
              one thread  ←→  one query point
Shared mem:   BLOCK_SZ × 16 bytes = 4 KB per block (fits in L1)
Per-thread:   max-heap of k HeapEntry structs live in registers/L1 cache
```

### Step-by-Step

```
knn_kernel(pts[], N, k) → knn_out[N × k]:

  qi  = this thread's query index
  q   = pts[qi]

  for base = 0, BLOCK_SZ, 2×BLOCK_SZ, ..., N-1:
    tile[threadIdx] = pts[base + threadIdx]          // cooperative load
    __syncthreads()

    for t = 0 .. tile_end-1:
      j = base + t
      if j == qi: skip                               // exclude self
      e = { dist²(q, tile[t]), tile[t].x/y/z, j }

      if heap not full (sz < k):
        heap_push(e)                                 // O(log k)
      else if e is better than heap root:
        heap[0] = e
        heap_sift_down()                             // O(log k) — evict worst

    __syncthreads()

  write heap[0..k-1] → knn_out[qi × k .. qi×k + k-1]
```

### Equalization Pass

A second kernel reads the KNN index table and applies the equalization formula:

```
equalize_knn_kernel(pts[], knn[][k], N, k) → out_intensity[N]:

  i = this thread's point index
  out_intensity[i] = equalize_from_neighbors(pts, knn[i], k, pts[i].intensity)
```

This is a fully independent, embarrassingly parallel operation — one thread per
point, no inter-thread communication.

---

## Mode 2 — Approximate KNN (Morton Z-order)

### Goal

Find an approximate k nearest neighbor set for each point, trading some recall
accuracy for significantly lower complexity. The approximation quality is
controlled by `APPROX_WIN_FACTOR` (default 16): larger values → closer to exact.

### Key Idea — Space-Filling Curves

A **Morton code** (Z-order curve) interleaves the binary representations of the
three coordinates into a single integer. Points that are spatially close in 3D
tend to be close in Morton order. Sorting by Morton code therefore groups
spatial neighbors together in a 1D array, making a linear window search a
reasonable approximation to a true radius search.

```
Morton code for (x, y, z):
  Expand each 10-bit coordinate into every 3rd bit position, then OR together:

  morton(x,y,z) = expand(x) | (expand(y) << 1) | (expand(z) << 2)

  expand(v):  0000001111111111
           →  001001001001001001001001001001   (bit i → bit 3i)
```

This gives a 30-bit code from 10-bit coordinates. Points with similar codes are
spatially nearby with high probability.

### Pipeline (5 GPU kernels + 1 OpenMP host step)

```
Step 0 — Bounding Box (OpenMP, host)
  OpenMP parallel reduction finds (xmin,xmax), (ymin,ymax), (zmin,zmax)
  across all N points. Used to normalize coordinates to [0,1023].

Step 1 — Normalize and Encode (compute_morton_kernel)
  For each point i:
    nx = (pts[i].x - xmin) * 1023 / xrange    (integer, no fp error)
    ny = (pts[i].y - ymin) * 1023 / yrange
    nz = (pts[i].z - zmin) * 1023 / zrange
    morton_pairs[i] = { morton3D(nx,ny,nz), i }

Step 2 — GPU Bitonic Sort (bitonic_sort_step, repeated)
  Sort the (morton_code, original_index) array of length n_padded (next
  power of 2 ≥ N) in ascending order of morton_code.

  Padded slots are initialized to UINT32_MAX so they sort to the end
  and are never accessed during neighbor search.

  Bitonic sort runs O(log²N) stages; each stage launches one kernel.
  Total: ~(log N)(log N + 1)/2 kernel launches. For N=131072: 153 launches.

Step 3 — Build Inverse Rank Map (build_rank_kernel)
  sorted_pairs[pos].y = original_index of the point at sorted position pos

  rank[original_index] = pos    (inverse lookup)
  Thread pos writes rank[sorted_pairs[pos].y] = pos.

Step 4 — Window Search (approx_knn_kernel)
  window W = min(N, max(APPROX_WIN_FACTOR × k, k+1))

  For each query i:
    p  = rank[i]                             // position in sorted order
    lo = clamp(p - W/2, 0, N-1)
    hi = clamp(lo + W - 1, 0, N-1)          // re-clamp if edge of array
    lo = max(0, hi - W + 1)                 // adjust lo if hi was clamped

    Run the same heap-based KNN on the W-point window,
    reading from sorted_pairs to map positions back to original indices.

Step 5 — Equalization
  Identical to Mode 1: equalize_knn_kernel reads the approximate KNN table.
```

### Window Geometry

```
                   sorted Morton-order array
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │    │    │    │ lo │ ..←─── W ───→.. │ hi │    │    │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
                              ↑
                         rank[i] = p
```

Points outside the window are never considered. The window is centered at
the query's Morton rank and clamped at the array boundaries.

### Why Bitonic Sort

Bitonic sort is well-suited to GPU because:
- Every stage is a fixed set of independent compare-swap operations → perfectly
  parallelizable with no data-dependent branching
- All memory accesses are strided but predictable → good cache utilization
- No atomic operations or divergence

For N ≤ 100,000, the O(N log²N) cost is dominated by the subsequent O(NW)
window search when W is large.

### Accuracy vs. Speed Tradeoff

| `APPROX_WIN_FACTOR` | Window W (k=16) | Typical MAE vs exact |
|:-------------------:|:---------------:|:--------------------:|
| 4                   | 64              | ~5–10                |
| 8                   | 128             | ~2–5                 |
| 16 (default)        | 256             | ~1–2                 |
| 32                  | 512             | <1                   |

MAE values are approximate and depend on point cloud structure.

---

## Mode 3 — K-means Clustering

### Goal

Partition the N points into k clusters using Lloyd's algorithm (k-means),
then apply histogram equalization *per cluster*: all points in a cluster
share one histogram and all receive equalized intensities drawn from it.

### Algorithm Overview

```
Initialization:  centroids[c] = pts[c]  for c = 0..k-1  (first k points)

Repeat up to T times:
  1. Assignment   — each point claims the nearest centroid
  2. Convergence  — if no assignment changed, stop
  3. Accumulation — sum coordinates of points in each cluster
  4. Finalization — new centroid = mean of cluster members
                    (frozen if cluster is empty)

Post-convergence:
  5. Build cluster histograms
  6. Equalize each point using its cluster's histogram
```

### Kernel 1 — Assignment (`kmeans_assign_kernel`)

Centroids are read frequently by every thread, so they are loaded into shared
memory once per block at kernel start.

```
__shared__ float4 c[MAX_K]          // k ≤ 128 → ≤ 2 KB, fits in L1

for i in parallel (one thread per point):
  find ci = argmin_c  dist²(pts[i], c[c])
  if assignments[i] ≠ ci:
    assignments[i] = ci
    *d_changed = 1           // idempotent write — no atomic needed
                             // racing writes of 1 from multiple threads are safe
```

`d_changed` is reset to 0 before each iteration and read back to host via
`cudaMemcpy`, which also acts as a GPU synchronization point.

### Kernels 2–3 — Centroid Update (Two-Level Reduction)

Accumulating N points into k cluster sums naïvely with global atomics causes
severe contention when k is small. The two-level approach reduces contention:

```
Level 1: kmeans_accumulate_kernel — per-block shared-memory atomics

  extern __shared__ float smem[]      // layout: [cx(k) | cy(k) | cz(k) | cnt(k)]
                                      // total: (3×float + 1×int) × k
                                      // = 16k bytes → 2 KB for k=128

  Initialize smem to zero
  Each thread atomically adds its point's (x,y,z) to smem[assignment]
  __syncthreads()
  Each thread flushes smem → global accumulator arrays with global atomics

Level 2: kmeans_finalize_kernel
  centroid[c] = { cx[c]/cnt[c], cy[c]/cnt[c], cz[c]/cnt[c] }
  (skipped for empty clusters — centroid frozen)
```

The benefit: with 256 threads per block, at most 256 global atomic operations
are issued per block instead of one per thread. For N=100,000 and BLOCK_SZ=256,
this is ~391 blocks × k global atomics instead of N × k global atomics — roughly
a 256× reduction in global atomic pressure.

### Kernels 4–5 — Equalization

After clustering converges, equalization proceeds in two steps:

```
Step 4 — Build cluster histograms (kmeans_hist_kernel)
  cluster_hist[c × 256 + intensity]++ for each point i in cluster c
  (global atomicAdd — contention is acceptable; k clusters × 256 bins is small)

Step 5 — Equalize (equalize_kmeans_kernel)
  For each point i:
    c = assignments[i]
    Apply the standard CDF equalization formula using cluster_hist[c][*]
    and cluster_size[c] as the neighborhood size
```

The cluster_size array is derived on the host by summing each cluster's
histogram bins (256 × k × 4 bytes transfer, negligible cost).

### Shared Memory Layout for Accumulation

```
smem:  [ cx[0] cx[1] ... cx[k-1] | cy[0] ... cy[k-1] | cz[0] ... cz[k-1] | cnt[0] ... cnt[k-1] ]
        ←──────── float[k] ──────→  ←── float[k] ───→  ←── float[k] ───→  ←── int[k] ──────→
        total = (3 × sizeof(float) + sizeof(int)) × k = 16k bytes
```

---

## Data Structures

### `Point` (16 bytes, `__align__(16)`)

```cpp
struct __align__(16) Point { int x, y, z, intensity; };
```

The 16-byte alignment enables a single 128-bit global load instruction (`LDG.128`)
per point on NVIDIA hardware, reducing the number of memory transactions by 4×
compared to loading each field separately.

### `HeapEntry` (28 bytes)

```cpp
struct HeapEntry {
    int64_t dist2;    // 8 bytes — integer squared distance, no fp error
    int     x, y, z; // 12 bytes — cached for tiebreaking
    int     idx;      // 4 bytes — original point index
};
```

`x`, `y`, `z` are cached in the heap entry so tiebreaking comparisons do not
require re-reading from global memory. `dist2` is `int64_t` because with
coordinates up to ~32,000 and three dimensions, the squared distance can
reach `3 × 32000² ≈ 3 × 10⁹`, which overflows `int32_t`.

---

## Profiling — Nvidia Nsight Integration

### NVTX Ranges

Every major phase is wrapped in a named, colored NVTX range visible in the
Nsight Systems timeline:

| Range name           | Color  | What it covers                              |
|----------------------|--------|---------------------------------------------|
| `KNN_total`          | Blue   | Entire KNN pipeline including H2D/D2H       |
| `KNN_compute`        | Blue   | `knn_kernel` execution only                 |
| `KNN_equalize`       | Blue   | `equalize_knn_kernel` execution             |
| `ApproxKNN_total`    | Green  | Entire approx pipeline                      |
| `ApproxKNN_morton`   | Green  | Morton code computation                     |
| `ApproxKNN_sort`     | Green  | Bitonic sort (all stages)                   |
| `ApproxKNN_rank`     | Green  | Inverse rank map construction               |
| `ApproxKNN_search`   | Green  | Window KNN search                           |
| `KMeans_total`       | Orange | Entire k-means pipeline                     |
| `KMeans_iterations`  | Orange | All assign/accumulate/finalize iterations   |
| `KMeans_equalize`    | Orange | Histogram build + equalization              |
| `IO_*`               | Red    | All file I/O and H2D/D2H transfers          |

### GpuTimer

Each kernel is individually timed with `cudaEvent_t` (device-side timer,
not subject to OS scheduling jitter unlike `clock()`). Results print to stdout.

### Profiling Commands

```bash
# Nsight Systems — whole-program timeline
nsys profile --trace=cuda,nvtx,osrt -o report ./a2 input.txt knn
nsys-ui report.nsys-rep         # open GUI

# Nsight Compute — deep single-kernel analysis
# (roofline, memory bandwidth, warp occupancy, instruction throughput)
ncu --set full -o ncu_knn ./a2 input.txt knn

# Profile only the knn_kernel to reduce collection overhead
ncu --kernel-name knn_kernel --set full -o ncu_knn ./a2 input.txt knn

# Nsight Compute roofline specifically
ncu --set roofline -o roofline ./a2 input.txt knn
```

### What to Look for in Nsight

| Symptom in profile                   | Likely cause                          | Fix                            |
|--------------------------------------|---------------------------------------|--------------------------------|
| Long red IO bars dominate timeline   | File read/write bottleneck            | Parallelize I/O (not needed for N≤1e5) |
| H2D/D2H overlaps nothing             | No async streams used                 | Expected — simplicity tradeoff |
| `knn_kernel` low occupancy           | Per-thread heap too large (MAX_K=128) | Reduce k or use register spill |
| `kmeans_accumulate` high L2 traffic  | Global atomics contend                | Already 2-level; reduce BLOCK_SZ |
| Bitonic sort many tiny kernels       | O(log²N) launches                     | Use CUB radix sort for larger N |

---

## Compile and Run Reference

```bash
# Compile (replace sm_86 with your GPU arch)
nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp" main.cu -o a2 -lnvToolsExt

# Run
./a2 input.txt knn           # exact KNN          → knn.txt
./a2 input.txt approx_knn    # approximate KNN    → approx_knn.txt
./a2 input.txt kmeans        # k-means cluster    → kmeans.txt

# Correctness testing
nvcc -O2 -arch=sm_86 -Xcompiler "-fopenmp" test.cu -o test -lnvToolsExt
./test                       # run all built-in test cases
./test input.txt             # test all three modes on a given file
./test --gen 1000 8 10 42    # generate random input (N K T seed) → stdout
```

---

## Complexity Summary

| Mode        | Kernel(s)                    | Time Complexity      | Space on GPU      |
|-------------|------------------------------|----------------------|-------------------|
| Exact KNN   | `knn_kernel`                 | O(N² / P × log k)   | O(N·k)            |
|             | `equalize_knn_kernel`        | O(N / P)             | O(N)              |
| Approx KNN  | `compute_morton_kernel`      | O(N / P)             | O(N)              |
|             | `bitonic_sort_step` × O(log²N) | O(N log²N / P)    | O(N)              |
|             | `build_rank_kernel`          | O(N / P)             | O(N)              |
|             | `approx_knn_kernel`          | O(N·W / P × log k)  | O(N·k)            |
|             | `equalize_knn_kernel`        | O(N / P)             | O(N)              |
| K-means     | `kmeans_assign_kernel` × T   | O(N·k / P) × T      | O(N + k)          |
|             | `kmeans_accumulate_kernel` × T| O(N / P) × T       | O(k) shared       |
|             | `kmeans_finalize_kernel` × T  | O(k / P) × T       | O(k)              |
|             | `kmeans_hist_kernel`         | O(N / P)             | O(k × 256)        |
|             | `equalize_kmeans_kernel`     | O(N / P)             | O(N)              |

P = number of GPU threads active in parallel (thousands to tens of thousands).
W = `APPROX_WIN_FACTOR × k` (window size for approx search).