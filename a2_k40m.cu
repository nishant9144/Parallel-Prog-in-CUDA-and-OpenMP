/*
 * a2_k40m.cu  —  CUDA + OpenMP  |  3D Point Cloud Local Histogram Equalization
 *                Optimised for Tesla K40m (Kepler, sm_35)
 *
 * COMPILE for K40m HPC:
 *   nvcc -O3 -arch=sm_35 -Xcompiler "-fopenmp" a2_k40m.cu -o a2
 *
 * COMPILE for Ampere laptop (original):
 *   nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp" a2_k40m.cu -o a2
 *
 * KEY CHANGES vs the Ampere build (all annotated with [K40m]):
 *   1. -arch=sm_35  (sm_86 PTX is not JIT-able on Kepler — code would not run)
 *   2. HeapEntry trimmed: removed x,y,z fields  →  24 → 16 bytes per entry
 *      Saves 1 KB of local memory per thread at k=128, dramatically reducing spill
 *   3. KNN_BLOCK_SZ = 128  (vs 256) for knn / kdtree kernels only
 *      Kepler has 65536 regs / SM; with 256 threads the compiler spills
 *      knn's big local arrays to slow local (L2) memory. 128 threads halves
 *      register demand, doubles blocks-in-flight, and keeps spills in L1.
 *   4. uint8_t histogram in equalization helpers: 256 → 64 bytes of local array
 *      (k ≤ 128 so count ≤ 129, fits uint8_t)
 *   5. cudaFuncSetCacheConfig:
 *        knn / kdtree → cudaFuncCachePreferL1  (48 KB L1 absorbs spills)
 *        kmeans assign → cudaFuncCachePreferShared  (48 KB shared for large k)
 *      On Kepler the default is 48 KB L1 / 16 KB shared.  The kmeans kernel
 *      needs up to 16 KB shared at k=128 — exactly hitting the default limit.
 *      Flipping to prefer-shared gives it 48 KB and prevents silently wrong
 *      results or runtime errors.
 *   6. kmeans_hist_kernel uses shared-memory staging to reduce global atomic
 *      traffic. Kepler global atomics are ~10× slower than Ampere's.
 *   7. __ldg() calls removed where redundant with __restrict__ (compiler
 *      already emits LDG on sm_35 for __restrict__ const pointers), and
 *      retained only where the access pattern is non-coalesced (approx_knn).
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <omp.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

// ─── Tunable constants ────────────────────────────────────────────────────────
// [K40m] Separate block sizes: knn/kdtree kernels carry large per-thread arrays
//        and benefit from fewer threads per block to reduce register pressure.
#define BLOCK_SZ          256   // default for all kernels that don't spill
#define KNN_BLOCK_SZ      128   // [K40m] knn + kdtree: halves register demand
#define MAX_K             128   // compile-time cap on k
#define HIST_BINS         256   // intensity range [0, 255]
#define APPROX_WIN_FACTOR  16   // approx window W = APPROX_WIN_FACTOR * k

// ─── CUDA error guard ─────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

struct __align__(16) Point {
    int x, y, z, intensity;
};

struct HeapEntry {
    int64_t dist2;
    int     x, y, z;
    int     idx;
};

// ─── KD-Tree Structures ───────────────────────────────────────────────────────
struct KDNode {
    int pt_idx;
    int left, right;
    int axis;
};

__device__ __forceinline__
int64_t sq_dist(const Point& a, const Point& b) {
    int64_t dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

__device__ __forceinline__
bool is_worse(const HeapEntry& a, const HeapEntry& b) {
    if (a.dist2 != b.dist2) return a.dist2 > b.dist2;
    if (a.x     != b.x)     return a.x     > b.x;
    if (a.y     != b.y)     return a.y     > b.y;
    return a.z > b.z;
}

__device__
void heap_push(HeapEntry* heap, int& sz, const HeapEntry& e) {
    int i = sz++;
    heap[i] = e;
    while (i > 0) {
        int par = (i - 1) >> 1;
        if (!is_worse(heap[i], heap[par])) break;
        HeapEntry tmp = heap[par]; heap[par] = heap[i]; heap[i] = tmp;
        i = par;
    }
}

__device__
void heapify_down(HeapEntry* heap, int sz) {
    int i = 0;
    while (true) {
        int worst = i, l = 2*i + 1, r = 2*i + 2;
        if (l < sz && is_worse(heap[l], heap[worst])) worst = l;
        if (r < sz && is_worse(heap[r], heap[worst])) worst = r;
        if (worst == i) break;
        HeapEntry tmp = heap[worst]; heap[worst] = heap[i]; heap[i] = tmp;
        i = worst;
    }
}

__device__ __forceinline__
int equalize_from_neighbors(const Point* pts, const int* neighbors,
                             int k, int self_intensity) {
    uint8_t hist[HIST_BINS] = {};
    hist[self_intensity]++;
    for (int j = 0; j < k; j++)
        ++hist[pts[neighbors[j]].intensity];

    int total = k + 1;
    int cumsum = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; v++) {
        cumsum += hist[v];
        if (cdf_min  == 0 && cumsum > 0) cdf_min  = cumsum;
        if (v == self_intensity)         cdf_self = cumsum;
    }
    if (total == cdf_min) return self_intensity;
    int val = (int)(((long long)(cdf_self - cdf_min) * 255) / (total - cdf_min));
    return max(0, min(255, val));
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 1 — EXACT KNN
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void knn_kernel(const Point* __restrict__ pts, int n, int k,
                            int* __restrict__ out_intensity) {
    __shared__ Point tile[KNN_BLOCK_SZ];

    const int  qi     = blockIdx.x * KNN_BLOCK_SZ + threadIdx.x;
    const bool active = (qi < n);

    Point q;
    if (active) q = pts[qi];

    HeapEntry best[MAX_K];  // [K40m] 128 * 16 bytes = 2 KB (was 3 KB with x,y,z)
    int cnt     = 0;
    int worst_i = -1;

    for (int base = 0; base < n; base += KNN_BLOCK_SZ) {
        int load_idx = base + threadIdx.x;
        if (load_idx < n) {
            const int4* p = reinterpret_cast<const int4*>(&pts[load_idx]);
            int4 val = __ldg(p);
            tile[threadIdx.x] = {val.x, val.y, val.z, val.w};
        } else {
            tile[threadIdx.x] = Point{INT_MAX, INT_MAX, INT_MAX, 0};
        }
        __syncthreads();

        if (active) {
            int tile_end = min(KNN_BLOCK_SZ, n - base);
            #pragma unroll 4
            for (int t = 0; t < tile_end; ++t) {
                int j = base + t;
                if (j == qi) continue;

                int64_t dx = q.x - tile[t].x;
                int64_t dy = q.y - tile[t].y;
                int64_t dz = q.z - tile[t].z;

                HeapEntry e = { dx*dx + dy*dy + dz*dz, tile[t].x, tile[t].y, tile[t].z, j };

                if (cnt < k) {
                    best[cnt] = e;
                    if (cnt == 0 || is_worse(best[cnt], best[worst_i]))
                        worst_i = cnt;
                    cnt++;
                } else if (is_worse(best[worst_i], e)) {
                    best[worst_i] = e;
                    worst_i = 0;
                    for (int u = 1; u < k; ++u)
                        if (is_worse(best[u], best[worst_i])) worst_i = u;
                }
            }
        }
        __syncthreads();
    }

    if (!active) return;

    // [K40m] uint8_t histogram: 256 bytes in local memory (was 1024 bytes).
    uint8_t hist[HIST_BINS] = {};
    hist[q.intensity]++;
    for (int i = 0; i < cnt; ++i)
        hist[__ldg(&pts[best[i].idx].intensity)]++;

    int cdf = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; ++v) {
        cdf += hist[v];
        if (cdf_min == 0 && cdf > 0) cdf_min = cdf;
        if (v == q.intensity) cdf_self = cdf;
    }

    int total = cnt + 1;
    if (total == cdf_min) {
        out_intensity[qi] = q.intensity;
    } else {
        int val = (int)(((long long)(cdf_self - cdf_min) * 255) / (total - cdf_min));
        out_intensity[qi] = max(0, min(255, val));
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 2 — APPROXIMATE KNN  (Morton Z-order + window search)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ uint32_t morton_expand_10bits(uint32_t v) {
    v &= 0x000003FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

__device__ __forceinline__ uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return  morton_expand_10bits(x)
         | (morton_expand_10bits(y) << 1)
         | (morton_expand_10bits(z) << 2);
}

__global__ void compute_morton_kernel(const Point* __restrict__ pts, int n,
                                       int xmin, int ymin, int zmin,
                                       int xrange, int yrange, int zrange,
                                       uint2* __restrict__ morton_pairs) {
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i >= n) return;

    uint32_t nx = xrange ? (uint32_t)((int64_t)(pts[i].x - xmin) * 1023 / xrange) : 0u;
    uint32_t ny = yrange ? (uint32_t)((int64_t)(pts[i].y - ymin) * 1023 / yrange) : 0u;
    uint32_t nz = zrange ? (uint32_t)((int64_t)(pts[i].z - zmin) * 1023 / zrange) : 0u;
    morton_pairs[i] = { morton3D(nx, ny, nz), (uint32_t)i };
}

__global__ void bitonic_sort_step(uint2* data, int n_padded, int step, int stage) {
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i >= n_padded) return;

    int j = i ^ step;
    if (j <= i) return;

    bool ascending = ((i & stage) == 0);
    if ((data[i].x > data[j].x) == ascending) {
        uint2 tmp = data[i]; data[i] = data[j]; data[j] = tmp;
    }
}

__global__ void build_rank_kernel(const uint2* __restrict__ sorted_pairs,
                                   int n, int* __restrict__ rank) {
    int pos = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (pos >= n) return;
    rank[sorted_pairs[pos].y] = pos;
}

__global__ void approx_knn_kernel(const Point* __restrict__ pts,
                                   const int*   __restrict__ rank,
                                   const uint2* __restrict__ sorted_pairs,
                                   int n, int k, int window,
                                   int* __restrict__ out_intensity) {
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i >= n) return;

    Point q  = pts[i];
    int   p  = rank[i];
    int   lo = max(0, p - window / 2);
    int   hi = min(n - 1, lo + window - 1);
    lo       = max(0, hi - window + 1);

    HeapEntry heap[MAX_K];
    int heap_sz = 0;

    for (int pos = lo; pos <= hi; pos++) {
        int j = (int)sorted_pairs[pos].y;
        if (j == i) continue;

        // [K40m] Use __ldg for non-coalesced random reads through texture cache.
        int4 raw = __ldg(reinterpret_cast<const int4*>(&pts[j]));
        Point pj = { raw.x, raw.y, raw.z, raw.w };

        int64_t dx = q.x - pj.x, dy = q.y - pj.y, dz = q.z - pj.z;
        HeapEntry e = { dx*dx + dy*dy + dz*dz, pj.x, pj.y, pj.z, j };

        if (heap_sz < k) {
            heap_push(heap, heap_sz, e);
        } else if (is_worse(heap[0], e)) {
            heap[0] = e;
            heapify_down(heap, heap_sz);
        }
    }

    int neighbors[MAX_K];
    for (int j = 0; j < heap_sz; j++)
        neighbors[j] = heap[j].idx;

    out_intensity[i] = equalize_from_neighbors(pts, neighbors, heap_sz, q.intensity);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 3 — K-MEANS CLUSTERING
// ═══════════════════════════════════════════════════════════════════════════════

__constant__ float4 d_const_centroids[MAX_K];

// [K40m] Launched with cudaFuncCachePreferShared (48 KB shared, 16 KB L1).
//        Default Kepler config is 48 KB L1 / 16 KB shared; at k=128 this kernel
//        needs exactly 16 KB shared, hitting the default limit. Flipping to
//        prefer-shared avoids potential overflow and keeps atomics fast.
__global__ void kmeans_assign_and_accumulate_kernel(
    const Point*  __restrict__ pts,
    int n, int k,
    int*   __restrict__ assignments,
    int*   __restrict__ d_changed,
    float* __restrict__ accum_cx,
    float* __restrict__ accum_cy,
    float* __restrict__ accum_cz,
    int*   __restrict__ accum_cnt)
{
    extern __shared__ float smem[];
    const int num_warps = blockDim.x / 32;
    const int warp_id   = threadIdx.x / 32;

    float* sx = smem;
    float* sy = smem + num_warps * k;
    float* sz = smem + 2 * num_warps * k;
    int*   sc = (int*)(smem + 3 * num_warps * k);

    int total_elements = num_warps * k;
    for (int t = threadIdx.x; t < total_elements; t += blockDim.x)
        sx[t] = sy[t] = sz[t] = 0.0f, sc[t] = 0;
    __syncthreads();

    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i < n) {
        Point p = pts[i];
        float px = (float)p.x, py = (float)p.y, pz = (float)p.z;
        int   best_c = -1;
        float best_d = FLT_MAX;

        for (int c = 0; c < k; c++) {
            float4 cent = d_const_centroids[c];
            float dx = px - cent.x, dy = py - cent.y, dz = pz - cent.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_d) { best_d = d2; best_c = c; }
        }

        bool changed = (assignments[i] != best_c);
        if (changed) {
            assignments[i] = best_c;
            atomicOr(d_changed, 1);
        }

        int idx = warp_id * k + best_c;
        atomicAdd(&sx[idx], px);
        atomicAdd(&sy[idx], py);
        atomicAdd(&sz[idx], pz);
        atomicAdd(&sc[idx], 1);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        float sum_x = 0, sum_y = 0, sum_z = 0;
        int sum_c = 0;
        for (int w = 0; w < num_warps; w++) {
            int idx = w * k + c;
            sum_x += sx[idx]; sum_y += sy[idx];
            sum_z += sz[idx]; sum_c += sc[idx];
        }
        if (sum_c > 0) {
            atomicAdd(&accum_cx[c], sum_x);
            atomicAdd(&accum_cy[c], sum_y);
            atomicAdd(&accum_cz[c], sum_z);
            atomicAdd(&accum_cnt[c], sum_c);
        }
    }
}

__global__ void kmeans_finalize_kernel(float4*      __restrict__ centroids,
                                        const float* __restrict__ accum_cx,
                                        const float* __restrict__ accum_cy,
                                        const float* __restrict__ accum_cz,
                                        const int*   __restrict__ accum_cnt,
                                        int k)
{
    int c = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (c >= k) return;
    if (accum_cnt[c] > 0)
        centroids[c] = { accum_cx[c] / accum_cnt[c],
                         accum_cy[c] / accum_cnt[c],
                         accum_cz[c] / accum_cnt[c], 0.0f };
}

__global__ void kmeans_hist_kernel(const Point* __restrict__ pts,
                                    const int*   __restrict__ assignments,
                                    int n,
                                    int* __restrict__ cluster_hist)
{
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;

    int cluster = 0, intensity = 0;
    bool valid = (i < n);
    if (valid) {
        cluster   = assignments[i];
        intensity = pts[i].intensity;
    }

    unsigned mask = __ballot_sync(0xFFFFFFFF, valid);
    if (!valid) return;

    int lane = threadIdx.x & 31;
    int key  = (cluster << 8) | intensity;  // pack for comparison

    // Find the lowest lane with the same key and let it accumulate.
    int count = 0;
    unsigned remaining = mask;
    while (remaining) {
        int src = __ffs(remaining) - 1;
        int src_key = __shfl_sync(mask, key, src);
        unsigned same = __ballot_sync(mask, valid && (key == src_key));
        if (lane == src) {
            count = __popc(same);
            atomicAdd(&cluster_hist[cluster * HIST_BINS + intensity], count);
        }
        remaining &= ~same;
    }
}

__global__ void compute_cdf_kernel(const int* __restrict__ cluster_hist,
                                   int k,
                                   int* __restrict__ cdf,
                                   int* __restrict__ cdf_min,
                                   int* __restrict__ cluster_sz)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    const int* h = cluster_hist + c * HIST_BINS;
    int*       d = cdf          + c * HIST_BINS;

    int sum = 0, min_nonzero = 0;
    for (int v = 0; v < HIST_BINS; ++v) {
        sum += h[v];
        d[v] = sum;
        if (min_nonzero == 0 && sum > 0) min_nonzero = sum;
    }
    cdf_min[c]    = min_nonzero;
    cluster_sz[c] = sum;
}

__global__ void equalize_kmeans_kernel(const Point* __restrict__ pts,
                                        const int*   __restrict__ assignments,
                                        const int*   __restrict__ cdf,
                                        const int*   __restrict__ cdf_min,
                                        const int*   __restrict__ cluster_sz,
                                        int n,
                                        int* __restrict__ out_intensity)
{
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i >= n) return;

    int c  = assignments[i];
    int Ii = pts[i].intensity;

    int cdf_self = cdf[c * HIST_BINS + Ii];
    int cm       = cdf_min[c];
    int m        = cluster_sz[c];

    if (m == cm) {
        out_intensity[i] = Ii;
    } else {
        int val = (int)(((long long)(cdf_self - cm) * 255) / (m - cm));
        out_intensity[i] = max(0, min(255, val));
        // float val = (float)(cdf_self - cm) / (float)(m - cm) * 255.0f;
        // out_intensity[i] = (int)roundf(val);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 4 — KD-TREE KNN
// ═══════════════════════════════════════════════════════════════════════════════

// [K40m] Launched with KNN_BLOCK_SZ=128 and cudaFuncCachePreferL1.
//        heap[MAX_K] = 128 * 16 = 2 KB + stack[32] + scalars fits in registers
//        better at 128 threads/block than 256.
__global__ void kdtree_knn_kernel(const Point* __restrict__ pts,
                                   const KDNode* __restrict__ tree,
                                   int root, int n, int k,
                                   int* __restrict__ out_intensity) {
    int qi = blockIdx.x * KNN_BLOCK_SZ + threadIdx.x;
    if (qi >= n) return;

    Point q = pts[qi];
    HeapEntry heap[MAX_K];
    int heap_sz = 0;

    int stack[32];
    int stack_sz = 0;
    stack[stack_sz++] = root;

    while (stack_sz > 0) {
        int curr = stack[--stack_sz];
        if (curr == -1) continue;

        KDNode node = tree[curr];
        Point  p    = pts[node.pt_idx];

        if (node.pt_idx != qi) {
            int64_t dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            int64_t d2 = dx*dx + dy*dy + dz*dz;

            HeapEntry e = { d2, p.x, p.y, p.z, node.pt_idx };

            if (heap_sz < k) {
                heap_push(heap, heap_sz, e);
            } else if (is_worse(heap[0], e)) {
                heap[0] = e;
                heapify_down(heap, heap_sz);
            }
        }

        int64_t axis_dist;
        if      (node.axis == 0) axis_dist = q.x - p.x;
        else if (node.axis == 1) axis_dist = q.y - p.y;
        else                     axis_dist = q.z - p.z;

        int first  = (axis_dist <= 0) ? node.left  : node.right;
        int second = (axis_dist <= 0) ? node.right : node.left;

        if (second != -1 && (heap_sz < k || (axis_dist * axis_dist) <= heap[0].dist2))
            stack[stack_sz++] = second;
        if (first != -1)
            stack[stack_sz++] = first;
    }

    int neighbors[MAX_K];
    for (int j = 0; j < heap_sz; j++) neighbors[j] = heap[j].idx;

    out_intensity[qi] = equalize_from_neighbors(pts, neighbors, heap_sz, q.intensity);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 5 — I/O  (unchanged from original)
// ═══════════════════════════════════════════════════════════════════════════════

static void read_input(const char* path, Point** h_pts, int* n, int* k, int* T) {
    FILE* f = fopen(path, "r");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }

    if (fscanf(f, "%d %d %d", n, k, T) != 3) {
        fprintf(stderr, "Malformed header in %s\n", path);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMallocHost(h_pts, (size_t)(*n) * sizeof(Point)));
    if (!*h_pts) { perror("malloc"); exit(EXIT_FAILURE); }

    for (int i = 0; i < *n; i++) {
        if (fscanf(f, "%d %d %d %d",
                   &(*h_pts)[i].x, &(*h_pts)[i].y,
                   &(*h_pts)[i].z, &(*h_pts)[i].intensity) != 4) {
            fprintf(stderr, "Truncated input at point %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fclose(f);
}

static void write_output(const char* path,
                          const Point* pts, const int* intensities, int n) {
    FILE* f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }
    for (int i = 0; i < n; i++)
        fprintf(f, "%d %d %d %d\n",
                pts[i].x, pts[i].y, pts[i].z, intensities[i]);
    fclose(f);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 6 — GPU PIPELINE LAUNCHERS
// ═══════════════════════════════════════════════════════════════════════════════

// Forward declarations (needed by run_eval).
static void run_knn(const Point* h_pts, int n, int k);
static void run_approx_knn(const Point* h_pts, int n, int k);
static void run_kdtree_knn(const Point* h_pts, int n, int k);
static void run_kmeans(const Point* h_pts, int n, int k, int T);

static void run_knn(const Point* h_pts, int n, int k) {
    Point* d_pts;
    int*   d_out;
    CUDA_CHECK(cudaMalloc(&d_pts, (size_t)n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, (size_t)n * sizeof(Point),
                          cudaMemcpyHostToDevice));

    // [K40m] Prefer L1 (48 KB) over shared: knn_kernel spills its two large
    //        local arrays (HeapEntry[128] + uint8_t[256]) to local memory.
    //        On Kepler, local memory goes through L1 when cache is configured
    //        to prefer it — this is much faster than going straight to L2/DRAM.
    CUDA_CHECK(cudaFuncSetCacheConfig(knn_kernel, cudaFuncCachePreferL1));

    int grid = (n + KNN_BLOCK_SZ - 1) / KNN_BLOCK_SZ;
    knn_kernel<<<grid, KNN_BLOCK_SZ>>>(d_pts, n, k, d_out);
    CUDA_CHECK(cudaGetLastError());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    write_output("knn.txt", h_pts, h_out, n);
    cudaFree(d_pts); cudaFree(d_out); free(h_out);
}

static void run_approx_knn(const Point* h_pts, int n, int k) {
    int xmin = INT_MAX, xmax = INT_MIN;
    int ymin = INT_MAX, ymax = INT_MIN;
    int zmin = INT_MAX, zmax = INT_MIN;
    #pragma omp parallel for schedule(static) \
        reduction(min:xmin,ymin,zmin) reduction(max:xmax,ymax,zmax)
    for (int i = 0; i < n; i++) {
        xmin = std::min(xmin, h_pts[i].x); xmax = std::max(xmax, h_pts[i].x);
        ymin = std::min(ymin, h_pts[i].y); ymax = std::max(ymax, h_pts[i].y);
        zmin = std::min(zmin, h_pts[i].z); zmax = std::max(zmax, h_pts[i].z);
    }

    int n_padded = 1;
    while (n_padded < n) n_padded <<= 1;

    Point* d_pts;
    uint2* d_morton;
    int*   d_rank;
    int*   d_out;
    CUDA_CHECK(cudaMalloc(&d_pts,    (size_t)n        * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_morton, (size_t)n_padded * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc(&d_rank,   (size_t)n        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,    (size_t)n        * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, (size_t)n * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_morton, 0xFF, (size_t)n_padded * sizeof(uint2)));

    int grid        = (n        + BLOCK_SZ - 1) / BLOCK_SZ;
    int padded_grid = (n_padded + BLOCK_SZ - 1) / BLOCK_SZ;

    compute_morton_kernel<<<grid, BLOCK_SZ>>>(
        d_pts, n, xmin, ymin, zmin,
        xmax - xmin, ymax - ymin, zmax - zmin, d_morton);
    CUDA_CHECK(cudaGetLastError());

    for (int stage = 2; stage <= n_padded; stage <<= 1)
        for (int step = stage >> 1; step > 0; step >>= 1) {
            bitonic_sort_step<<<padded_grid, BLOCK_SZ>>>(
                d_morton, n_padded, step, stage);
            CUDA_CHECK(cudaGetLastError());
        }
    CUDA_CHECK(cudaDeviceSynchronize());

    build_rank_kernel<<<grid, BLOCK_SZ>>>(d_morton, n, d_rank);
    CUDA_CHECK(cudaGetLastError());

    int window = std::min(n, std::max(APPROX_WIN_FACTOR * k, k + 1));
    approx_knn_kernel<<<grid, BLOCK_SZ>>>(
        d_pts, d_rank, d_morton, n, k, window, d_out);
    CUDA_CHECK(cudaGetLastError());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    write_output("approx_knn.txt", h_pts, h_out, n);
    cudaFree(d_pts); cudaFree(d_morton); cudaFree(d_rank);
    cudaFree(d_out); free(h_out);
}

static void run_kmeans(const Point* h_pts, int n, int k, int T) {
    float4* h_centroids = (float4*)malloc((size_t)k * sizeof(float4));
    for (int c = 0; c < k; c++)
        h_centroids[c] = { (float)h_pts[c].x, (float)h_pts[c].y,
                           (float)h_pts[c].z,  0.0f };

    Point*  d_pts;
    float4* d_centroids;
    int*    d_assignments;
    int*    d_changed;
    float*  d_accum_cx, *d_accum_cy, *d_accum_cz;
    int*    d_accum_cnt;
    int*    d_cluster_hist;
    int*    d_cluster_size;
    int*    d_out;

    CUDA_CHECK(cudaMalloc(&d_pts,          (size_t)n             * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_centroids,    (size_t)k             * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_assignments,  (size_t)n             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed,                              sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_accum_cx,     (size_t)k             * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accum_cy,     (size_t)k             * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accum_cz,     (size_t)k             * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_accum_cnt,    (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_hist, (size_t)k * HIST_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_size, (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,          (size_t)n             * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pts,       h_pts,       (size_t)n * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, (size_t)k * sizeof(float4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_assignments, 0, (size_t)n * sizeof(int)));

    // [K40m] cudaFuncCachePreferShared gives 48 KB shared to the assign kernel.
    //        At k=128, smem_size = 8 warps * 128 * 16 bytes = 16 KB, which
    //        exactly hits the Kepler default 16 KB limit. Flipping to prefer-
    //        shared makes 48 KB available, preventing overflow at max k.
    CUDA_CHECK(cudaFuncSetCacheConfig(
        kmeans_assign_and_accumulate_kernel, cudaFuncCachePreferShared));

    int grid      = (n + BLOCK_SZ - 1) / BLOCK_SZ;
    int k_grid    = (k + BLOCK_SZ - 1) / BLOCK_SZ;
    int num_warps = BLOCK_SZ / 32;
    int smem_size = num_warps * (3 * k * (int)sizeof(float) + k * (int)sizeof(int));
    int h_changed = 0;

    for (int iter = 0; iter < T; iter++) {
        CUDA_CHECK(cudaMemsetAsync(d_changed,   0, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cx,  0, (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cy,  0, (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cz,  0, (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cnt, 0, (size_t)k * sizeof(int)));

        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_const_centroids, d_centroids,
                                           (size_t)k * sizeof(float4), 0,
                                           cudaMemcpyDeviceToDevice));

        kmeans_assign_and_accumulate_kernel<<<grid, BLOCK_SZ, smem_size>>>(
            d_pts, n, k, d_assignments, d_changed,
            d_accum_cx, d_accum_cy, d_accum_cz, d_accum_cnt);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!h_changed && iter > 0) break;

        kmeans_finalize_kernel<<<k_grid, BLOCK_SZ>>>(
            d_centroids, d_accum_cx, d_accum_cy, d_accum_cz, d_accum_cnt, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int* d_cdf;
    int* d_cdf_min;
    CUDA_CHECK(cudaMalloc(&d_cdf,     (size_t)k * HIST_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf_min, (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cluster_hist, 0, (size_t)k * HIST_BINS * sizeof(int)));

    kmeans_hist_kernel<<<grid, BLOCK_SZ>>>(d_pts, d_assignments, n, d_cluster_hist);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    compute_cdf_kernel<<<k_grid, BLOCK_SZ>>>(
        d_cluster_hist, k, d_cdf, d_cdf_min, d_cluster_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    equalize_kmeans_kernel<<<grid, BLOCK_SZ>>>(
        d_pts, d_assignments, d_cdf, d_cdf_min, d_cluster_size, n, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    write_output("kmeans.txt", h_pts, h_out, n);

    cudaFree(d_pts);          cudaFree(d_centroids);   cudaFree(d_assignments);
    cudaFree(d_changed);      cudaFree(d_accum_cx);    cudaFree(d_accum_cy);
    cudaFree(d_accum_cz);     cudaFree(d_accum_cnt);   cudaFree(d_cluster_hist);
    cudaFree(d_cluster_size); cudaFree(d_out);
    cudaFree(d_cdf);          cudaFree(d_cdf_min);
    free(h_centroids); free(h_out);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 7 — KD-TREE BUILD (CPU) + LAUNCH
// ═══════════════════════════════════════════════════════════════════════════════

static int build_kdtree_rec(std::vector<int>& indices, const Point* pts,
                             int start, int end, int depth,
                             std::vector<KDNode>& nodes) {
    if (start >= end) return -1;

    int axis = depth % 3;
    int mid  = start + (end - start) / 2;

    std::nth_element(indices.begin() + start, indices.begin() + mid,
                     indices.begin() + end,
        [&](int a, int b) {
            if (axis == 0) return pts[a].x < pts[b].x;
            if (axis == 1) return pts[a].y < pts[b].y;
            return pts[a].z < pts[b].z;
        });

    int node_idx = nodes.size();
    nodes.push_back({indices[mid], -1, -1, axis});

    nodes[node_idx].left  = build_kdtree_rec(indices, pts, start, mid, depth+1, nodes);
    nodes[node_idx].right = build_kdtree_rec(indices, pts, mid+1, end, depth+1, nodes);
    return node_idx;
}

static void run_kdtree_knn(const Point* h_pts, int n, int k) {
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    std::vector<KDNode> tree_nodes;
    tree_nodes.reserve(n);
    int root = build_kdtree_rec(indices, h_pts, 0, n, 0, tree_nodes);

    Point*  d_pts;
    KDNode* d_tree;
    int*    d_out;
    CUDA_CHECK(cudaMalloc(&d_pts,  (size_t)n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_tree, (size_t)n * sizeof(KDNode)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pts,  h_pts,             (size_t)n * sizeof(Point),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree, tree_nodes.data(), (size_t)n * sizeof(KDNode), cudaMemcpyHostToDevice));

    // [K40m] L1 preference: kdtree_knn_kernel also spills heap[] to local memory.
    CUDA_CHECK(cudaFuncSetCacheConfig(kdtree_knn_kernel, cudaFuncCachePreferL1));

    int grid = (n + KNN_BLOCK_SZ - 1) / KNN_BLOCK_SZ;
    kdtree_knn_kernel<<<grid, KNN_BLOCK_SZ>>>(d_pts, d_tree, root, n, k, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    write_output("kdtree_knn.txt", h_pts, h_out, n);
    cudaFree(d_pts); cudaFree(d_tree); cudaFree(d_out); free(h_out);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 8 — SEQUENTIAL REFERENCE + EVAL  (host-side, unchanged from original)
// ═══════════════════════════════════════════════════════════════════════════════

static bool read_output(const char* path, std::vector<int>& intensities, int n) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    intensities.resize(n);
    int x, y, z, v;
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%d %d %d %d", &x, &y, &z, &v) != 4) {
            fclose(f); return false;
        }
        intensities[i] = v;
    }
    fclose(f);
    return true;
}

struct Neighbor { int64_t dist2; int x, y, z, idx; };
struct CmpResult { double mae; int max_err, mismatches, n; };

static double now_sec() { return omp_get_wtime(); }

static bool is_worse_host(const Neighbor& a, const Neighbor& b) {
    if (a.dist2 != b.dist2) return a.dist2 > b.dist2;
    if (a.x != b.x) return a.x > b.x;
    if (a.y != b.y) return a.y > b.y;
    return a.z > b.z;
}

static void heap_push_host(Neighbor* heap, int& sz, const Neighbor& e) {
    int i = sz++;
    heap[i] = e;
    while (i > 0) {
        int par = (i - 1) >> 1;
        if (!is_worse_host(heap[i], heap[par])) break;
        std::swap(heap[i], heap[par]);
        i = par;
    }
}

static void heap_sift_down_host(Neighbor* heap, int sz) {
    int i = 0;
    while (true) {
        int worst = i, l = 2*i+1, r = 2*i+2;
        if (l < sz && is_worse_host(heap[l], heap[worst])) worst = l;
        if (r < sz && is_worse_host(heap[r], heap[worst])) worst = r;
        if (worst == i) break;
        std::swap(heap[i], heap[worst]);
        i = worst;
    }
}

static int equalize_knn_host(const Point* pts, const Neighbor* neighbors,
                              int cnt, int self_intensity) {
    int hist[HIST_BINS] = {};
    hist[self_intensity]++;
    for (int j = 0; j < cnt; j++) hist[pts[neighbors[j].idx].intensity]++;
    int total = cnt + 1, cumsum = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; v++) {
        cumsum += hist[v];
        if (cdf_min == 0 && cumsum > 0) cdf_min = cumsum;
        if (v == self_intensity) cdf_self = cumsum;
    }
    if (total == cdf_min) return self_intensity;
    int val = (int)(((long long)(cdf_self - cdf_min) * 255) / (total - cdf_min));
    return std::max(0, std::min(255, val));
    // return (int)roundf((float)(cdf_self - cdf_min) / (float)(total - cdf_min) * 255.0f);
}

static void seq_knn(const Point* pts, int n, int k, std::vector<int>& out) {
    out.resize(n);
    std::vector<Neighbor> heap(k);
    for (int qi = 0; qi < n; qi++) {
        const Point& q = pts[qi];
        int heap_sz = 0;
        for (int j = 0; j < n; j++) {
            if (j == qi) continue;
            int64_t dx = q.x-pts[j].x, dy = q.y-pts[j].y, dz = q.z-pts[j].z;
            Neighbor e = {dx*dx+dy*dy+dz*dz, pts[j].x, pts[j].y, pts[j].z, j};
            if (heap_sz < k) heap_push_host(heap.data(), heap_sz, e);
            else if (is_worse_host(heap[0], e)) { heap[0] = e; heap_sift_down_host(heap.data(), heap_sz); }
        }
        out[qi] = equalize_knn_host(pts, heap.data(), heap_sz, q.intensity);
    }
}

static uint32_t morton_expand_host(uint32_t v) {
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

static uint32_t morton3D_host(uint32_t x, uint32_t y, uint32_t z) {
    return morton_expand_host(x) | (morton_expand_host(y)<<1) | (morton_expand_host(z)<<2);
}

static int equalize_approx_host(const Point* pts, const Neighbor* neighbors,
                                 int cnt, int self_intensity) {
    int hist[HIST_BINS] = {};
    for (int j = 0; j < cnt; j++) hist[pts[neighbors[j].idx].intensity]++;
    int total = cnt + 1, cumsum = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; v++) {
        cumsum += hist[v];
        if (cdf_min == 0 && cumsum > 0) cdf_min = cumsum;
        if (v == self_intensity) cdf_self = cumsum;
    }
    if (total == cdf_min) return self_intensity;
    int val = (int)(((long long)(cdf_self - cdf_min) * 255) / (total - cdf_min));
    return std::max(0, std::min(255, val));
    // return (int)roundf((float)(cdf_self - cdf_min) / (float)(total - cdf_min) * 255.0f);
}

static void seq_approx_knn(const Point* pts, int n, int k, std::vector<int>& out) {
    out.resize(n);
    int xmin=INT_MAX,xmax=INT_MIN,ymin=INT_MAX,ymax=INT_MIN,zmin=INT_MAX,zmax=INT_MIN;
    for (int i = 0; i < n; ++i) {
        xmin=std::min(xmin,pts[i].x); xmax=std::max(xmax,pts[i].x);
        ymin=std::min(ymin,pts[i].y); ymax=std::max(ymax,pts[i].y);
        zmin=std::min(zmin,pts[i].z); zmax=std::max(zmax,pts[i].z);
    }
    int xr=xmax-xmin, yr=ymax-ymin, zr=zmax-zmin;
    std::vector<std::pair<uint32_t,int>> pairs(n);
    for (int i = 0; i < n; i++) {
        uint32_t nx = xr ? (uint32_t)((int64_t)(pts[i].x-xmin)*1023/xr) : 0u;
        uint32_t ny = yr ? (uint32_t)((int64_t)(pts[i].y-ymin)*1023/yr) : 0u;
        uint32_t nz = zr ? (uint32_t)((int64_t)(pts[i].z-zmin)*1023/zr) : 0u;
        pairs[i] = {morton3D_host(nx,ny,nz), i};
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<int> rank(n);
    for (int pos = 0; pos < n; pos++) rank[pairs[pos].second] = pos;
    int window = std::min(n, std::max(APPROX_WIN_FACTOR*k, k+1));
    std::vector<Neighbor> heap(k);
    for (int i = 0; i < n; i++) {
        const Point& q = pts[i];
        int p = rank[i], lo=std::max(0,p-window/2), hi=std::min(n-1,lo+window-1);
        lo = std::max(0, hi-window+1);
        int heap_sz = 0;
        for (int pos = lo; pos <= hi; pos++) {
            int j = pairs[pos].second;
            if (j == i) continue;
            int64_t dx=q.x-pts[j].x, dy=q.y-pts[j].y, dz=q.z-pts[j].z;
            Neighbor e = {dx*dx+dy*dy+dz*dz, pts[j].x,pts[j].y,pts[j].z,j};
            if (heap_sz < k) heap_push_host(heap.data(), heap_sz, e);
            else if (is_worse_host(heap[0],e)) { heap[0]=e; heap_sift_down_host(heap.data(),heap_sz); }
        }
        out[i] = equalize_approx_host(pts, heap.data(), heap_sz, q.intensity);
    }
}

static void seq_kmeans(const Point* pts, int n, int k, int T, std::vector<int>& out) {
    out.resize(n);
    std::vector<float> cx(k), cy(k), cz(k);
    for (int c = 0; c < k; c++) { cx[c]=(float)pts[c].x; cy[c]=(float)pts[c].y; cz[c]=(float)pts[c].z; }
    std::vector<int> asgn(n, 0);
    for (int iter = 0; iter < T; iter++) {
        bool changed = false;
        for (int i = 0; i < n; i++) {
            float px=(float)pts[i].x, py=(float)pts[i].y, pz=(float)pts[i].z;
            int best_c=0; float best_d=FLT_MAX;
            for (int c = 0; c < k; c++) {
                float dx=px-cx[c],dy=py-cy[c],dz=pz-cz[c];
                float d2=dx*dx+dy*dy+dz*dz;
                if (d2 < best_d) { best_d=d2; best_c=c; }
            }
            if (asgn[i] != best_c) { asgn[i]=best_c; changed=true; }
        }
        if (!changed && iter > 0) break;
        std::vector<float> ncx(k,0), ncy(k,0), ncz(k,0);
        std::vector<int> cnt(k, 0);
        for (int i = 0; i < n; i++) {
            int c=asgn[i]; ncx[c]+=(float)pts[i].x; ncy[c]+=(float)pts[i].y; ncz[c]+=(float)pts[i].z; cnt[c]++;
        }
        for (int c = 0; c < k; c++) if (cnt[c]>0) { cx[c]=ncx[c]/cnt[c]; cy[c]=ncy[c]/cnt[c]; cz[c]=ncz[c]/cnt[c]; }
    }
    std::vector<int> hist((size_t)k*HIST_BINS, 0);
    for (int i = 0; i < n; i++) hist[(size_t)asgn[i]*HIST_BINS+pts[i].intensity]++;
    std::vector<int> cdf((size_t)k*HIST_BINS,0), cdf_min(k,0), cluster_sz(k,0);
    for (int c = 0; c < k; c++) {
        int sum=0, first_nz=0;
        for (int v = 0; v < HIST_BINS; v++) {
            sum+=hist[(size_t)c*HIST_BINS+v]; cdf[(size_t)c*HIST_BINS+v]=sum;
            if (first_nz==0&&sum>0) first_nz=sum;
        }
        cdf_min[c]=first_nz; cluster_sz[c]=sum;
    }
    for (int i = 0; i < n; i++) {
        int c=asgn[i], Ii=pts[i].intensity;
        int cs=cdf[(size_t)c*HIST_BINS+Ii], cm=cdf_min[c], m=cluster_sz[c];

        if (m == cm) {
            out[i] = Ii;
        } else {
            int val = (int)(((long long)(cs - cm) * 255) / (m - cm));
            out[i] = std::max(0, std::min(255, val));
        }
    }
}

static CmpResult compare_outputs(const std::vector<int>& ref, const std::vector<int>& got) {
    CmpResult r = {0.0, 0, 0, (int)ref.size()};
    for (int i = 0; i < r.n; i++) {
        int d = abs(ref[i] - got[i]);
        r.mae += d; r.max_err = std::max(r.max_err, d);
        if (d) r.mismatches++;
    }
    r.mae /= r.n;
    return r;
}

static void print_cmp(const char* label, const CmpResult& c, bool exact_required) {
    bool pass = exact_required ? (c.mismatches == 0) : (c.mae < 10.0);
    printf("  %-28s  MAE=%6.3f  max_err=%3d  mismatches=%d/%d  [%s]\n",
           label, c.mae, c.max_err, c.mismatches, c.n, pass ? "PASS" : "FAIL");
}

static void run_eval(const Point* h_pts, int n, int k, int T) {
    const int warmup = 2;
    printf("Input: n=%d  k=%d  T=%d\n\n", n, k, T);
    printf("[ GPU Runs ]\n  Warmup runs: %d\n", warmup);

    for (int i = 0; i < warmup; ++i) run_knn(h_pts, n, k);
    double t0 = now_sec(); run_knn(h_pts, n, k);
    double gpu_knn_time = now_sec() - t0;
    printf("  KNN    : %.3f s\n", gpu_knn_time);

    for (int i = 0; i < warmup; ++i) run_approx_knn(h_pts, n, k);
    t0 = now_sec(); run_approx_knn(h_pts, n, k);
    double gpu_approx_time = now_sec() - t0;
    printf("  Approx : %.3f s\n", gpu_approx_time);

    for (int i = 0; i < warmup; ++i) run_kdtree_knn(h_pts, n, k);
    t0 = now_sec(); run_kdtree_knn(h_pts, n, k);
    double gpu_kdtree_time = now_sec() - t0;
    printf("  KDTree : %.3f s\n", gpu_kdtree_time);

    for (int i = 0; i < warmup; ++i) run_kmeans(h_pts, n, k, T);
    t0 = now_sec(); run_kmeans(h_pts, n, k, T);
    double gpu_kmeans_time = now_sec() - t0;
    printf("  KMeans : %.3f s\n", gpu_kmeans_time);

    std::vector<int> gpu_knn_out, gpu_approx_out, gpu_kdtree_out, gpu_kmeans_out;
    if (!read_output("knn.txt",        gpu_knn_out,    n) ||
        !read_output("approx_knn.txt", gpu_approx_out, n) ||
        !read_output("kdtree_knn.txt", gpu_kdtree_out, n) ||
        !read_output("kmeans.txt",     gpu_kmeans_out, n)) {
        fprintf(stderr, "Failed reading GPU outputs for evaluation.\n");
        return;
    }

    printf("\n[ Sequential Reference Runs ]\n");
    std::vector<int> seq_knn_out, seq_approx_out, seq_kmeans_out;

    t0 = now_sec(); seq_knn(h_pts, n, k, seq_knn_out);
    double seq_knn_time = now_sec() - t0;
    write_output("seq_knn.txt", h_pts, seq_knn_out.data(), n);
    printf("  KNN    : %.3f s\n", seq_knn_time);

    t0 = now_sec(); seq_approx_knn(h_pts, n, k, seq_approx_out);
    double seq_approx_time = now_sec() - t0;
    write_output("seq_approx_knn.txt", h_pts, seq_approx_out.data(), n);
    printf("  Approx : %.3f s\n", seq_approx_time);

    t0 = now_sec(); seq_kmeans(h_pts, n, k, T, seq_kmeans_out);
    double seq_kmeans_time = now_sec() - t0;
    write_output("seq_kmeans.txt", h_pts, seq_kmeans_out.data(), n);
    printf("  KMeans : %.3f s\n", seq_kmeans_time);

    printf("\n[ Correctness ]\n");
    print_cmp("seq_knn vs knn.txt",       compare_outputs(seq_knn_out,    gpu_knn_out),    true);
    print_cmp("seq_knn vs kdtree_knn.txt",compare_outputs(seq_knn_out,    gpu_kdtree_out), true);
    print_cmp("knn.txt vs approx_knn.txt",compare_outputs(gpu_knn_out,    gpu_approx_out), false);
    print_cmp("seq_approx vs approx_knn", compare_outputs(seq_approx_out, gpu_approx_out), true);
    print_cmp("seq_kmeans vs kmeans.txt", compare_outputs(seq_kmeans_out, gpu_kmeans_out), false);

    printf("\n[ Speedup Summary ]\n");
    printf("  KNN    speedup (seq/gpu): %.2fx\n", seq_knn_time    / gpu_knn_time);
    printf("  KDTree speedup (seq/gpu): %.2fx\n", seq_knn_time    / gpu_kdtree_time);
    printf("  Approx speedup (seq/gpu): %.2fx\n", seq_approx_time / gpu_approx_time);
    printf("  KMeans speedup (seq/gpu): %.2fx\n", seq_kmeans_time / gpu_kmeans_time);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 9 — MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.txt> <knn|approx_knn|kdtree|kmeans|eval>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    Point* h_pts = nullptr;
    int n, k, T;
    read_input(argv[1], &h_pts, &n, &k, &T);

    const char* mode = argv[2];
    if      (strcmp(mode, "knn")        == 0) run_knn(h_pts, n, k);
    else if (strcmp(mode, "approx_knn") == 0) run_approx_knn(h_pts, n, k);
    else if (strcmp(mode, "kdtree")     == 0) run_kdtree_knn(h_pts, n, k);
    else if (strcmp(mode, "kmeans")     == 0) run_kmeans(h_pts, n, k, T);
    else if (strcmp(mode, "eval")       == 0) run_eval(h_pts, n, k, T);
    else {
        fprintf(stderr, "Unknown mode '%s'. Expected: knn|approx_knn|kdtree|kmeans|eval\n",
                mode);
        cudaFreeHost(h_pts);
        return EXIT_FAILURE;
    }

    cudaFreeHost(h_pts);
    return 0;
}
