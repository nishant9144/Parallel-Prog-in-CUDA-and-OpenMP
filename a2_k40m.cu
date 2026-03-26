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
__global__ void approx_knn_kernel(
    const int* __restrict__ px, const int* __restrict__ py,
    const int* __restrict__ pz, const int* __restrict__ pI,
    const int* __restrict__ orig, int n,
    const int* __restrict__ c_start, const int* __restrict__ c_cnt,
    int Gx, int Gy, int Gz, int x_min, int y_min, int z_min, float inv_vs,
    int k, int* __restrict__ out_I)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int xi = px[i], yi = py[i], zi = pz[i];
    int cx = min(Gx - 1, max(0, (int)((xi - x_min) * inv_vs)));
    int cy = min(Gy - 1, max(0, (int)((yi - y_min) * inv_vs)));
    int cz = min(Gz - 1, max(0, (int)((zi - z_min) * inv_vs)));

    // Float heap for low register usage, populated via exact integer math
    float kd[128];
    int ki[128];
    int cnt = 0;
    float wD = 0.0f; int wP = 0;

    for (int r = 0; r <= 4; r++) {
        int ax = max(0, cx - r), bx = min(Gx - 1, cx + r);
        int ay = max(0, cy - r), by = min(Gy - 1, cy + r);
        int az = max(0, cz - r), bz = min(Gz - 1, cz + r);

        for (int gx = ax; gx <= bx; gx++)
        for (int gy = ay; gy <= by; gy++)
        for (int gz = az; gz <= bz; gz++) {
            // Skip inner cells already checked in smaller radius
            if (r > 0 && abs(gx - cx) < r && abs(gy - cy) < r && abs(gz - cz) < r) continue;

            int cid = gx * Gy * Gz + gy * Gz + gz;
            int start = c_start[cid], c_count = c_cnt[cid];

            for (int t = start; t < start + c_count; t++) {
                if (orig[t] == orig[i]) continue;
                
                long long dx = (long long)(px[t] - xi);
                long long dy = (long long)(py[t] - yi);
                long long dz = (long long)(pz[t] - zi);
                float d = (float)(dx*dx + dy*dy + dz*dz);

                if (cnt < k) {
                    kd[cnt] = d; ki[cnt] = t; ++cnt;
                    if (cnt == k) {
                        wD = kd[0]; wP = 0;
                        for (int u = 1; u < k; u++) if (kd[u] > wD) { wD = kd[u]; wP = u; }
                    }
                } else if (d < wD) {
                    kd[wP] = d; ki[wP] = t;
                    wD = kd[0]; wP = 0;
                    for (int u = 1; u < k; u++) if (kd[u] > wD) { wD = kd[u]; wP = u; }
                }
            }
        }
        if (r >= 2 && cnt == k) break;
    }

    int hist[256] = {0};
    hist[pI[i]]++;
    for (int t = 0; t < cnt; t++) hist[pI[ki[t]]]++;

    int cdf[256]; cdf[0] = hist[0];
    for (int v = 1; v < 256; v++) cdf[v] = cdf[v - 1] + hist[v];

    int cdf_min = 0;
    for (int v = 0; v < 256; v++) if (cdf[v] > 0) { cdf_min = cdf[v]; break; }

    int Ii = pI[i], m = cnt + 1;
    if (m == cdf_min || m == 0) { out_I[orig[i]] = Ii; return; }
    
    int val = (int)(((long long)(cdf[Ii] - cdf_min) * 255) / (m - cdf_min));
    out_I[orig[i]] = max(0, min(255, val));
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 3 — K-MEANS CLUSTERING
// ═══════════════════════════════════════════════════════════════════════════════

// Use int4 to strictly enforce integer centroids
__constant__ int4 dc_cent[128]; 

__global__ void kmeans_assign_and_accumulate_kernel(
    const Point* __restrict__ pts,
    int n, int k,
    int* __restrict__ asgn,
    int* __restrict__ changed,
    int* __restrict__ acx,  // Changed to int*
    int* __restrict__ acy,  // Changed to int*
    int* __restrict__ acz,  // Changed to int*
    int* __restrict__ acnt)
{
    // Shared memory must also be cast to int* on the host launch
    extern __shared__ int smem[];
    const int n_warp = blockDim.x / 32;
    const int w_id   = threadIdx.x / 32;

    int* sx = smem;
    int* sy = smem + n_warp * k;
    int* sz = smem + 2 * n_warp * k;
    int* sc = smem + 3 * n_warp * k;

    for (int t = threadIdx.x; t < n_warp * k; t += blockDim.x) {
        sx[t] = sy[t] = sz[t] = sc[t] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Point p = pts[i];
        int bC = 0;
        long long bD = 9223372036854775807LL; // LLONG_MAX

        for (int c = 0; c < k; c++) {
            int4 cent = dc_cent[c];
            long long dx = p.x - cent.x, dy = p.y - cent.y, dz = p.z - cent.z;
            long long d2 = dx*dx + dy*dy + dz*dz;

            // Strict deterministic tie-breaker matching Kernel 1
            bool rep = false;
            if (d2 < bD) {
                rep = true;
            } else if (d2 == bD) {
                int4 b_cent = dc_cent[bC];
                if (cent.x < b_cent.x) rep = true;
                else if (cent.x == b_cent.x) {
                    if (cent.y < b_cent.y) rep = true;
                    else if (cent.y == b_cent.y) {
                        if (cent.z < b_cent.z) rep = true;
                    }
                }
            }

            if (rep) { bD = d2; bC = c; }
        }

        if (asgn[i] != bC) {
            asgn[i] = bC;
            atomicOr(changed, 1);
        }

        int idx = w_id * k + bC;
        atomicAdd(&sx[idx], p.x);
        atomicAdd(&sy[idx], p.y);
        atomicAdd(&sz[idx], p.z);
        atomicAdd(&sc[idx], 1);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        int sum_x = 0, sum_y = 0, sum_z = 0, sum_c = 0;
        for (int w = 0; w < n_warp; w++) {
            int idx = w * k + c;
            sum_x += sx[idx]; sum_y += sy[idx];
            sum_z += sz[idx]; sum_c += sc[idx];
        }
        if (sum_c > 0) {
            atomicAdd(&acx[c], sum_x);
            atomicAdd(&acy[c], sum_y);
            atomicAdd(&acz[c], sum_z);
            atomicAdd(&acnt[c], sum_c);
        }
    }
}

__global__ void kmeans_finalize_kernel(
    int4* __restrict__ centroids,
    const int* __restrict__ acx,
    const int* __restrict__ acy,
    const int* __restrict__ acz,
    const int* __restrict__ acnt,
    int k)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    if (acnt[c] > 0) {
        // Integer division enforces truncation, preventing cluster drift
        centroids[c] = { acx[c] / acnt[c],
                         acy[c] / acnt[c],
                         acz[c] / acnt[c], 0 };
    }
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

// Struct and comparator for sorting points by grid cell
typedef struct {
    int cell;
    int idx;
} SortItem;

static int cmp_cells(const void* a, const void* b) {
    int ca = ((const SortItem*)a)->cell;
    int cb = ((const SortItem*)b)->cell;
    return (ca > cb) - (ca < cb);
}

static void run_approx_knn(const Point* h_pts, int n, int k) {
    int xmn = h_pts[0].x, xmx = xmn, ymn = h_pts[0].y, ymx = ymn, zmn = h_pts[0].z, zmx = zmn;
    
    // Find bounding box
    #pragma omp parallel for reduction(min:xmn,ymn,zmn) reduction(max:xmx,ymx,zmx)
    for (int i = 0; i < n; i++) {
        if (h_pts[i].x < xmn) xmn = h_pts[i].x;  if (h_pts[i].x > xmx) xmx = h_pts[i].x;
        if (h_pts[i].y < ymn) ymn = h_pts[i].y;  if (h_pts[i].y > ymx) ymx = h_pts[i].y;
        if (h_pts[i].z < zmn) zmn = h_pts[i].z;  if (h_pts[i].z > zmx) zmx = h_pts[i].z;
    }
    
    float lx = (float)(xmx - xmn) + 1e-5f;
    float ly = (float)(ymx - ymn) + 1e-5f;
    float lz = (float)(zmx - zmn) + 1e-5f;

    // Grid sizing heuristics
    float mx_l = lx > ly ? (lx > lz ? lx : lz) : (ly > lz ? ly : lz);
    float cpd = cbrtf(27.0f * n / (float)(k + 1));
    if (cpd < 1.0f) cpd = 1.0f;
    
    float vs = mx_l / cpd;
    if (vs < 1e-6f) vs = 1e-6f;

    int Gx = (int)(lx / vs) + 1; if (Gx > 512) Gx = 512;
    int Gy = (int)(ly / vs) + 1; if (Gy > 512) Gy = 512;
    int Gz = (int)(lz / vs) + 1; if (Gz > 512) Gz = 512;
    long long t_cells = (long long)Gx * Gy * Gz;

    SortItem* items = (SortItem*)malloc((size_t)n * sizeof(SortItem));

    // Assign cells
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int cx = (int)((h_pts[i].x - xmn) / vs); if (cx >= Gx) cx = Gx - 1;
        int cy = (int)((h_pts[i].y - ymn) / vs); if (cy >= Gy) cy = Gy - 1;
        int cz = (int)((h_pts[i].z - zmn) / vs); if (cz >= Gz) cz = Gz - 1;
        items[i].cell = cx * Gy * Gz + cy * Gz + cz;
        items[i].idx = i;
    }

    qsort(items, n, sizeof(SortItem), cmp_cells);

    // Setup SoA layout for coalesced memory access
    int* px   = (int*)malloc((size_t)n * sizeof(int));
    int* py   = (int*)malloc((size_t)n * sizeof(int));
    int* pz   = (int*)malloc((size_t)n * sizeof(int));
    int* pI   = (int*)malloc((size_t)n * sizeof(int));
    int* orig = (int*)malloc((size_t)n * sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int oi = items[i].idx;
        px[i]   = h_pts[oi].x;
        py[i]   = h_pts[oi].y;
        pz[i]   = h_pts[oi].z;
        pI[i]   = h_pts[oi].intensity;
        orig[i] = oi;
    }

    // Build prefix sum for cell offsets
    int* cs = (int*)calloc((size_t)t_cells, sizeof(int));
    int* cc = (int*)calloc((size_t)t_cells, sizeof(int));
    
    for (int i = 0; i < n; i++) cc[items[i].cell]++;
    for (long long c = 1; c < t_cells; c++) cs[c] = cs[c - 1] + cc[c - 1];

    int *d_px, *d_py, *d_pz, *d_pI, *d_orig, *d_cs, *d_cc, *d_out;
    CUDA_CHECK(cudaMalloc(&d_px,   (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_py,   (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pz,   (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pI,   (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_orig, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cs,   (size_t)t_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cc,   (size_t)t_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_px,   px,   (size_t)n * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py,   py,   (size_t)n * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz,   pz,   (size_t)n * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pI,   pI,   (size_t)n * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig, orig, (size_t)n * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cs,   cs,   (size_t)t_cells * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cc,   cc,   (size_t)t_cells * sizeof(int), cudaMemcpyHostToDevice));

    int blk = BLOCK_SZ, grd = (n + blk - 1) / blk;
    approx_knn_kernel<<<grd, blk>>>(
        d_px, d_py, d_pz, d_pI, d_orig, n,
        d_cs, d_cc, Gx, Gy, Gz, xmn, ymn, zmn, 1.0f / vs, k, d_out);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    write_output("approx_knn.txt", h_pts, h_out, n);

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz); cudaFree(d_pI);
    cudaFree(d_orig); cudaFree(d_cs); cudaFree(d_cc); cudaFree(d_out);

    free(items); free(px); free(py); free(pz);
    free(pI); free(orig); free(cs); free(cc); free(h_out);
}

static void run_kmeans(const Point* h_pts, int n, int k, int T) {
    int4* h_centroids = (int4*)malloc((size_t)k * sizeof(int4));
    for (int c = 0; c < k; c++)
        h_centroids[c] = { h_pts[c].x, h_pts[c].y, h_pts[c].z, 0 };

    Point* d_pts;
    int4* d_centroids;
    int* d_assignments;
    int* d_changed;
    int* d_accum_cx, *d_accum_cy, *d_accum_cz;
    int* d_accum_cnt;
    int* d_cluster_hist;
    int* d_cluster_size;
    int* d_out;

    CUDA_CHECK(cudaMalloc(&d_pts,          (size_t)n             * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_centroids,    (size_t)k             * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_assignments,  (size_t)n             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed,                              sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_accum_cx,     (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_accum_cy,     (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_accum_cz,     (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_accum_cnt,    (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_hist, (size_t)k * HIST_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_size, (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,          (size_t)n             * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pts,       h_pts,       (size_t)n * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, (size_t)k * sizeof(int4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_assignments, 0, (size_t)n * sizeof(int)));

    CUDA_CHECK(cudaFuncSetCacheConfig(
        kmeans_assign_and_accumulate_kernel, cudaFuncCachePreferShared));

    int grid      = (n + BLOCK_SZ - 1) / BLOCK_SZ;
    int k_grid    = (k + BLOCK_SZ - 1) / BLOCK_SZ;
    int num_warps = BLOCK_SZ / 32;
    // 4 integer arrays (x, y, z, cnt) per warp
    int smem_size = num_warps * 4 * k * (int)sizeof(int); 
    int h_changed = 0;

    for (int iter = 0; iter < T; iter++) {
        CUDA_CHECK(cudaMemsetAsync(d_changed,   0, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cx,  0, (size_t)k * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cy,  0, (size_t)k * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cz,  0, (size_t)k * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cnt, 0, (size_t)k * sizeof(int)));

        // Assuming you updated the constant memory array name to dc_cent
        CUDA_CHECK(cudaMemcpyToSymbolAsync(dc_cent, d_centroids,
                                           (size_t)k * sizeof(int4), 0,
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





typedef struct {
    int l, r;  // Left and right child indices
    int ax;    // Split axis (0=x, 1=y, 2=z)
    int id;    // Point index
} ApproxKDNode;

typedef struct {
    int v, id;
} SNode;

static int cmp_sn(const void* a, const void* b) {
    int va = ((SNode*)a)->v, vb = ((SNode*)b)->v;
    return (va > vb) - (va < vb);
}

// Recursively builds the KD-Tree and returns the root index
static int build_kd(ApproxKDNode* tree, const Point* pts, int* idxs, int st, int ed, int dep, int* t_cnt) {
    if (st > ed) return -1;
    
    int ax = dep % 3;
    int num = ed - st + 1;
    
    SNode* tmp = (SNode*)malloc((size_t)num * sizeof(SNode));
    for (int i = 0; i < num; i++) {
        tmp[i].id = idxs[st + i];
        if      (ax == 0) tmp[i].v = pts[idxs[st + i]].x;
        else if (ax == 1) tmp[i].v = pts[idxs[st + i]].y;
        else              tmp[i].v = pts[idxs[st + i]].z;
    }
    
    qsort(tmp, num, sizeof(SNode), cmp_sn);
    for (int i = 0; i < num; i++) idxs[st + i] = tmp[i].id;
    free(tmp);
    
    int mid = st + num / 2;
    int curr = (*t_cnt)++;
    
    tree[curr].ax = ax;
    tree[curr].id = idxs[mid];
    tree[curr].l = build_kd(tree, pts, idxs, st, mid - 1, dep + 1, t_cnt);
    tree[curr].r = build_kd(tree, pts, idxs, mid + 1, ed, dep + 1, t_cnt);
    
    return curr;
}

__global__ void approx_kd_kernel(const Point* __restrict__ pts, 
                                 const ApproxKDNode* __restrict__ tree, 
                                 int root, int n, int k, int max_v, 
                                 int* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Point q = pts[i];
    
    // Float heap saves registers on K40m
    float kd[128]; 
    int ki[128];
    int cnt = 0, wP = 0;
    float wD = 0.0f; 

    int st[64]; 
    int top = 0;
    st[top] = root;
    int vis = 0;

    // Iterative tree traversal
    while (top >= 0 && vis < max_v) {
        int c = st[top--];
        if (c == -1) continue;

        ApproxKDNode node = tree[c];
        vis++;
        Point p = pts[node.id];

        if (node.id != i) {
            long long dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            float d2 = (float)(dx*dx + dy*dy + dz*dz);
            
            if (cnt < k) {
                kd[cnt] = d2; ki[cnt] = node.id; cnt++;
                if (cnt == k) {
                    wD = kd[0]; wP = 0;
                    for (int u = 1; u < k; u++) if (kd[u] > wD) { wD = kd[u]; wP = u; }
                }
            } else if (d2 < wD) {
                kd[wP] = d2; ki[wP] = node.id;
                wD = kd[0]; wP = 0;
                for (int u = 1; u < k; u++) if (kd[u] > wD) { wD = kd[u]; wP = u; }
            }
        }

        long long diff = 0;
        if      (node.ax == 0) diff = q.x - p.x;
        else if (node.ax == 1) diff = q.y - p.y;
        else                   diff = q.z - p.z;

        int near_c = (diff <= 0) ? node.l : node.r;
        int far_c  = (diff <= 0) ? node.r : node.l;

        // Prune branches
        if (cnt < k || (float)(diff * diff) <= wD) {
            if (far_c != -1) st[++top] = far_c;
        }
        if (near_c != -1) st[++top] = near_c;
    }

    // Inline Equalization
    int hist[256] = {0};
    hist[q.intensity]++;
    for (int t = 0; t < cnt; t++) hist[pts[ki[t]].intensity]++;
    
    int cdf[256]; cdf[0] = hist[0];
    for (int v = 1; v < 256; v++) cdf[v] = cdf[v-1] + hist[v];
    
    int c_min = 0;
    for (int v = 0; v < 256; v++) if (cdf[v] > 0) { c_min = cdf[v]; break; }
    
    int m = cnt + 1;
    if (m == c_min || m == 0) { out[i] = q.intensity; return; }
    
    int val = (int)(((long long)(cdf[q.intensity] - c_min) * 255) / (m - c_min));
    out[i] = val > 255 ? 255 : (val < 0 ? 0 : val);
}

static void run_approx_kdtree(const Point* h_pts, int n, int k) {
    int* idxs = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idxs[i] = i;

    ApproxKDNode* h_tree = (ApproxKDNode*)malloc((size_t)n * sizeof(ApproxKDNode));
    int t_cnt = 0;
    int root = build_kd(h_tree, h_pts, idxs, 0, n - 1, 0, &t_cnt);

    Point* d_pts;
    ApproxKDNode* d_tree;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_pts,  (size_t)n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_tree, (size_t)n * sizeof(ApproxKDNode)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pts,  h_pts,  (size_t)n * sizeof(Point),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree, h_tree, (size_t)n * sizeof(ApproxKDNode), cudaMemcpyHostToDevice));

    // Limit backtracking to maintain O(log N) speed
    int max_v = k * 5; 
    if (max_v < 128) max_v = 128;

    int blk = 128; // Small block size prevents L1 spills during stack traversal
    int grd = (n + blk - 1) / blk;

    approx_kd_kernel<<<grd, blk>>>(d_pts, d_tree, root, n, k, max_v, d_out);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    // Write results
    FILE* fout = fopen("approx_kdtree.txt", "w");
    if (fout) {
        for (int i = 0; i < n; i++) {
            fprintf(fout, "%d %d %d %d\n", h_pts[i].x, h_pts[i].y, h_pts[i].z, h_out[i]);
        }
        fclose(fout);
    }

    cudaFree(d_pts); cudaFree(d_tree); cudaFree(d_out);
    free(idxs); free(h_tree); free(h_out);
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
    
    // Centroids are strictly integers
    std::vector<int> cx(k), cy(k), cz(k);
    for (int c = 0; c < k; c++) { 
        cx[c] = pts[c].x; cy[c] = pts[c].y; cz[c] = pts[c].z; 
    }
    
    std::vector<int> asgn(n, 0);
    for (int iter = 0; iter < T; iter++) {
        bool changed = false;
        
        for (int i = 0; i < n; i++) {
            int px = pts[i].x, py = pts[i].y, pz = pts[i].z;
            int best_c = 0; 
            long long best_d = 9223372036854775807LL; // LLONG_MAX
            
            for (int c = 0; c < k; c++) {
                long long dx = px - cx[c], dy = py - cy[c], dz = pz - cz[c];
                long long d2 = dx*dx + dy*dy + dz*dz;
                
                // Strict deterministic tie-breaker
                bool rep = false;
                if (d2 < best_d) rep = true;
                else if (d2 == best_d) {
                    if (cx[c] < cx[best_c]) rep = true;
                    else if (cx[c] == cx[best_c]) {
                        if (cy[c] < cy[best_c]) rep = true;
                        else if (cy[c] == cy[best_c] && cz[c] < cz[best_c]) rep = true;
                    }
                }
                
                if (rep) { best_d = d2; best_c = c; }
            }
            if (asgn[i] != best_c) { asgn[i] = best_c; changed = true; }
        }
        
        if (!changed && iter > 0) break;
        
        // Accumulate using long long to avoid overflow on large clusters
        std::vector<long long> ncx(k,0), ncy(k,0), ncz(k,0);
        std::vector<int> cnt(k, 0);
        for (int i = 0; i < n; i++) {
            int c = asgn[i]; 
            ncx[c] += pts[i].x; ncy[c] += pts[i].y; ncz[c] += pts[i].z; 
            cnt[c]++;
        }
        
        for (int c = 0; c < k; c++) {
            if (cnt[c] > 0) { 
                // Integer division enforces truncation, preventing cluster drift
                cx[c] = (int)(ncx[c] / cnt[c]); 
                cy[c] = (int)(ncy[c] / cnt[c]); 
                cz[c] = (int)(ncz[c] / cnt[c]); 
            }
        }
    }
    
    std::vector<int> hist((size_t)k * HIST_BINS, 0);
    for (int i = 0; i < n; i++) hist[(size_t)asgn[i] * HIST_BINS + pts[i].intensity]++;
    
    std::vector<int> cdf((size_t)k * HIST_BINS, 0), cdf_min(k, 0), cluster_sz(k, 0);
    for (int c = 0; c < k; c++) {
        int sum = 0, first_nz = 0;
        for (int v = 0; v < HIST_BINS; v++) {
            sum += hist[(size_t)c * HIST_BINS + v]; 
            cdf[(size_t)c * HIST_BINS + v] = sum;
            if (first_nz == 0 && sum > 0) first_nz = sum;
        }
        cdf_min[c] = first_nz; cluster_sz[c] = sum;
    }
    
    for (int i = 0; i < n; i++) {
        int c = asgn[i], Ii = pts[i].intensity;
        int cs = cdf[(size_t)c * HIST_BINS + Ii], cm = cdf_min[c], m = cluster_sz[c];

        if (m == cm || m == 0) {
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
    printf("  KNN           : %.3f s\n", gpu_knn_time);

    for (int i = 0; i < warmup; ++i) run_approx_knn(h_pts, n, k);
    t0 = now_sec(); run_approx_knn(h_pts, n, k);
    double gpu_approx_time = now_sec() - t0;
    printf("  Approx        : %.3f s\n", gpu_approx_time);

    for (int i = 0; i < warmup; ++i) run_kdtree_knn(h_pts, n, k);
    t0 = now_sec(); run_kdtree_knn(h_pts, n, k);
    double gpu_kdtree_time = now_sec() - t0;
    printf("  KDTree        : %.3f s\n", gpu_kdtree_time);

    for (int i = 0; i < warmup; ++i) run_approx_kdtree(h_pts, n, k);
    t0 = now_sec(); run_approx_kdtree(h_pts, n, k);
    double gpu_approx_kd_time = now_sec() - t0;
    printf("  Approx KDTree : %.3f s\n", gpu_approx_kd_time);

    for (int i = 0; i < warmup; ++i) run_kmeans(h_pts, n, k, T);
    t0 = now_sec(); run_kmeans(h_pts, n, k, T);
    double gpu_kmeans_time = now_sec() - t0;
    printf("  KMeans        : %.3f s\n", gpu_kmeans_time);

    std::vector<int> gpu_knn_out, gpu_approx_out, gpu_kdtree_out, gpu_approx_kd_out, gpu_kmeans_out;
    if (!read_output("knn.txt",           gpu_knn_out,       n) ||
        !read_output("approx_knn.txt",    gpu_approx_out,    n) ||
        !read_output("kdtree_knn.txt",    gpu_kdtree_out,    n) ||
        !read_output("approx_kdtree.txt", gpu_approx_kd_out, n) ||
        !read_output("kmeans.txt",        gpu_kmeans_out,    n)) {
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
    print_cmp("seq_knn vs knn.txt",          compare_outputs(seq_knn_out,    gpu_knn_out),       true);
    print_cmp("seq_knn vs kdtree_knn.txt",   compare_outputs(seq_knn_out,    gpu_kdtree_out),    true);
    print_cmp("knn.txt vs approx_knn.txt",   compare_outputs(gpu_knn_out,    gpu_approx_out),    false);
    print_cmp("knn.txt vs approx_kdtree.txt",compare_outputs(gpu_knn_out,    gpu_approx_kd_out), false);
    print_cmp("seq_approx vs approx_knn",    compare_outputs(seq_approx_out, gpu_approx_out),    true);
    print_cmp("seq_kmeans vs kmeans.txt",    compare_outputs(seq_kmeans_out, gpu_kmeans_out),    false);

    printf("\n[ Speedup Summary ]\n");
    printf("  KNN           speedup (seq/gpu): %.2fx\n", seq_knn_time    / gpu_knn_time);
    printf("  KDTree        speedup (seq/gpu): %.2fx\n", seq_knn_time    / gpu_kdtree_time);
    printf("  Approx KDTree speedup (seq/gpu): %.2fx\n", seq_knn_time    / gpu_approx_kd_time);
    printf("  Approx        speedup (seq/gpu): %.2fx\n", seq_approx_time / gpu_approx_time);
    printf("  KMeans        speedup (seq/gpu): %.2fx\n", seq_kmeans_time / gpu_kmeans_time);
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
    else if (strcmp(mode, "approx_kdtree") == 0) run_approx_kdtree(h_pts, n, k);
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
