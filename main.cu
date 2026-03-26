/*
 * main.cu  —  CUDA + OpenMP  |  3D Point Cloud Local Histogram Equalization
 *
 * COMPILE:
 *   nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp" main.cu -o a2
 *   Replace sm_86 with your arch: sm_75 / sm_80 / sm_89 / sm_90
 *
 * USAGE:
 *   ./a2 input.txt knn
 *   ./a2 input.txt approx_knn
 *   ./a2 input.txt kmeans
 *
 * INPUT FORMAT:
 *   N  k  T
 *   x0 y0 z0 intensity0
 *   ...
 *   Constraints: N ≤ 1e5,  0 < k ≤ 128,  T ≤ 50,  intensity ∈ {0..255}
 *
 * OUTPUT:
 *   knn.txt | approx_knn.txt | kmeans.txt
 *   One line per point:  x  y  z  new_intensity
*/

#include <cstddef>
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

// ─── Tunable constants ────────────────────────────────────────────────────────
#define BLOCK_SZ          256   // threads per block (all kernels)
#define MAX_K             128   // compile-time cap on k; sizes per-thread heap
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


#if __CUDA_ARCH__ >= 700
    #define BALLOT(mask, pred) __ballot_sync(mask, pred)
#else
    #define BALLOT(mask, pred) __ballot(pred)   // mask is ignored on <7.0
#endif


// 16-byte alignment → single LDG.128 instruction per point load on GPU.
struct __align__(16) Point {
    int x, y, z, intensity;
};

struct HeapEntry {
    int64_t dist2;    
    int     x, y, z; 
    int     idx;
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
    return                         a.z     > b.z;
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






// Build a 256-bin histogram from neighbor intensities in thread-local registers
__device__ __forceinline__
int equalize_from_neighbors(const Point* pts, const int* neighbors,
                             int k, int self_intensity) {
    int hist[HIST_BINS] = {0};

    for (int j = 0; j < k; j++)
        ++hist[pts[neighbors[j]].intensity];

    int total = k+1;
    int cumsum = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; v++) {
        cumsum += hist[v];
        if (cdf_min  == 0 && cumsum > 0)  cdf_min  = cumsum;
        if (v == self_intensity)          cdf_self = cumsum;
    }
    if (total == cdf_min) return self_intensity;
    return (int)roundf((float)(cdf_self - cdf_min) / (float)(total - cdf_min) * 255.0f);
}




/*
    For finding the nearest neighbours of the points.
*/
__global__ void knn_kernel(const Point* __restrict__ pts, int n, int k,
                            int* __restrict__ out_intensity) {
    __shared__ Point tile[BLOCK_SZ];

    const int  qi     = blockIdx.x * BLOCK_SZ + threadIdx.x;
    const bool active = (qi < n);

    Point q;
    if (active) q = pts[qi];

    HeapEntry best[MAX_K];
    int cnt = 0;
    int worst_i = -1;

    for (int base = 0; base < n; base += BLOCK_SZ) {
        // Safe cooperative load - all threads must participate to avoid deadlock
        int load_idx = base + threadIdx.x;
        if (load_idx < n) {
            const int4* p = reinterpret_cast<const int4*>(&pts[load_idx]);
            int4 val = __ldg(p);
            tile[threadIdx.x] = {val.x, val.y, val.z, val.w};
        } else {
            tile[threadIdx.x] = Point{INT_MAX, INT_MAX, INT_MAX, 0};
        }
        __syncthreads();

        // If this thread is out of bounds, it still must help load shared memory
        // and hit the syncthreads, so we skip computations here using `if (active)`
        if (active) {
            int tile_end = min(BLOCK_SZ, n - base);
            #pragma unroll 4
            for (int t = 0; t < tile_end; ++t) {
                int j = base + t;
                if (j == qi) continue;

                HeapEntry e;
                e.idx = j;
                e.x = tile[t].x; e.y = tile[t].y; e.z = tile[t].z;
                
                // Keep pure integer math to avoid precision loss
                int64_t dx = q.x - e.x, dy = q.y - e.y, dz = q.z - e.z;
                e.dist2 = dx*dx + dy*dy + dz*dz;

                if (cnt < k) {
                    best[cnt] = e;
                    if (cnt == 0 || is_worse(e, best[worst_i])) {
                        worst_i = cnt;
                    }
                    cnt++;
                } else if (is_worse(best[worst_i], e)) {
                    best[worst_i] = e;
                    // Find new worst
                    worst_i = 0;
                    for (int u = 1; u < k; ++u) {
                        if (is_worse(best[u], best[worst_i])) {
                            worst_i = u;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!active) return;

    // Inline equalisation (histogramming)
    int hist[HIST_BINS] = {0};
    hist[q.intensity]++; // self
    
    for (int i = 0; i < cnt; ++i) {
        hist[__ldg(&pts[best[i].idx].intensity)]++;
    }

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
        float val = (float)(cdf_self - cdf_min) / (float)(total - cdf_min) * 255.0f;
        out_intensity[qi] = (int)roundf(val);
    }
}








// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 2 — APPROXIMATE KNN  (Morton Z-order + window search)
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Sort points by Morton code (Z-order space-filling curve): spatially close
//  points cluster together in Morton order, so a contiguous window of W entries
//  centred at a query's Morton rank gives a high-quality approximate neighborhood.
//
//  Pipeline:
//    [OMP host]  Bounding-box reduction
//    [GPU]       Normalize coords to [0,1023]; compute 30-bit Morton codes
//    [GPU]       Bitonic sort on (morton_code, orig_idx) pairs
//    [GPU]       Inverse rank map: rank[orig_idx] = sorted_position
//    [GPU]       Heap-based KNN inside window W = min(N, APPROX_WIN_FACTOR·k)
//    [GPU]       Equalization (reuses equalize_knn_kernel)

// Spread a 10-bit integer across every 3rd bit position (magic-bits method).
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

// Normalize each coordinate to [0, 1023] then emit a (morton_code, orig_idx) pair.
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

// One compare-and-swap pass of GPU bitonic sort.
// Array length must be a power of 2; padded slots carry code = UINT32_MAX
// so they sort to the tail and are never touched by the window search.
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

// Build inverse rank map: rank[original_idx] = position in sorted array.
__global__ void build_rank_kernel(const uint2* __restrict__ sorted_pairs,
                                   int n, int* __restrict__ rank) {
    int pos = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (pos >= n) return;
    rank[sorted_pairs[pos].y] = pos;
}

// Heap-based KNN restricted to a W-wide Morton-order window centred on the query.
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
    lo       = max(0, hi - window + 1);    // re-anchor lo when hi was clamped

    HeapEntry heap[MAX_K];
    int heap_sz = 0;

    for (int pos = lo; pos <= hi; pos++) {
        int j = (int)sorted_pairs[pos].y;
        if (j == i) continue;

        HeapEntry e;
        e.dist2 = sq_dist(q, pts[j]);
        e.x = pts[j].x; e.y = pts[j].y; e.z = pts[j].z; e.idx = j;

        if (heap_sz < k) {
            heap_push(heap, heap_sz, e);
        } else if (is_worse(heap[0], e)) {
            heap[0] = e;
            heapify_down(heap, heap_sz);
        }
    }

    int neighbors[MAX_K];
    for (int j = 0; j < heap_sz; j++) {
        neighbors[j] = heap[j].idx;
    }
    
    out_intensity[i] = equalize_from_neighbors(pts, neighbors, heap_sz, q.intensity);
}












__constant__ float4 d_const_centroids[MAX_K];  

__global__ void kmeans_assign_and_accumulate_kernel(const Point*  __restrict__ pts,
                                      int n, int k,
                                      int* __restrict__ assignments,
                                      int* __restrict__ d_changed,
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
    for (int t = threadIdx.x; t < total_elements; t += blockDim.x) {
        sx[t] = sy[t] = sz[t] = 0.0f;
        sc[t] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i < n) {
        Point p = pts[i];
        float px = (float)p.x, py = (float)p.y, pz = (float)p.z;
        int   best_c = -1;
        float best_d = FLT_MAX;

        for (int c = 0; c < k; c++) {
            float4 cent = d_const_centroids[c];   // broadcast from constant cache
            float dx = px - cent.x;
            float dy = py - cent.y;
            float dz = pz - cent.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_d) { best_d = d2; best_c = c; }
        }

        bool changed = (assignments[i] != best_c);
        if (changed) assignments[i] = best_c;

        // unsigned mask = __ballot_sync(0xFFFFFFFF, changed);
        unsigned mask = BALLOT(0xFFFFFFFF, changed);
        if (mask != 0 && (__ffs(mask) - 1 == (threadIdx.x % 32))) atomicOr(d_changed, 1); 

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
            sum_x += sx[idx];
            sum_y += sy[idx];
            sum_z += sz[idx];
            sum_c += sc[idx];
        }
        
        if (sum_c > 0) {
            atomicAdd(&accum_cx[c], sum_x);
            atomicAdd(&accum_cy[c], sum_y);
            atomicAdd(&accum_cz[c], sum_z);
            atomicAdd(&accum_cnt[c], sum_c);
        }
    }
}

// New centroid = mean of its members.  Empty clusters keep their prior centroid.
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

// Build per-cluster intensity histogram.
// cluster_hist[ c * HIST_BINS + v ] = #members of cluster c with intensity v.
__global__ void kmeans_hist_kernel(const Point* __restrict__ pts,
                                    const int*   __restrict__ assignments,
                                    int n,
                                    int* __restrict__ cluster_hist) 
{
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&cluster_hist[assignments[i] * HIST_BINS + pts[i].intensity], 1);
}



__global__ void compute_cdf_kernel(const int* __restrict__ cluster_hist,
                                   int k,
                                   int* __restrict__ cdf,          // [k*256]
                                   int* __restrict__ cdf_min,      // [k]
                                   int* __restrict__ cluster_sz)   // [k]
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    const int* h = cluster_hist + c * HIST_BINS;
    int*       d = cdf          + c * HIST_BINS;

    int sum = 0;
    int min_nonzero = 0;
    for (int v = 0; v < HIST_BINS; ++v) {
        sum += h[v];
        d[v] = sum;
        if (min_nonzero == 0 && sum > 0) min_nonzero = sum;
    }
    cdf_min[c]   = min_nonzero;
    cluster_sz[c] = sum;               // total points in the cluster
}



// Each point reads its cluster's histogram, computes the CDF inline, remaps.
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
    int cm      = cdf_min[c];
    int m       = cluster_sz[c];

    if (m == cm) {
        out_intensity[i] = Ii;
    } else {
        float val = (float)(cdf_self - cm) / (float)(m - cm) * 255.0f;
        out_intensity[i] = (int)roundf(val);
    }
}





static void read_input(const char* path, Point** h_pts, int* n, int* k, int* T) {
    FILE* f = fopen(path, "r");

    if (!f) { perror("fopen"); 
        exit(EXIT_FAILURE); }

    if (fscanf(f, "%d %d %d", n, k, T) != 3) {
        fprintf(stderr, "Malformed header in %s\n", path); 
        exit(EXIT_FAILURE);
    }


    CUDA_CHECK(cudaMallocHost(h_pts, (size_t)(*n) * sizeof(Point)));

    if (!*h_pts) { perror("malloc"); exit(EXIT_FAILURE); }
    for (int i = 0; i < *n; i++) {
        // TODO: Will later remove the checks here
        if (fscanf(f, "%d %d %d %d",
                   &(*h_pts)[i].x,     &(*h_pts)[i].y,
                   &(*h_pts)[i].z,     &(*h_pts)[i].intensity) != 4) 
        {
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








static void run_knn(const Point* h_pts, int n, int k) {
    Point* d_pts;
    int*   d_out;
    CUDA_CHECK(cudaMalloc(&d_pts, (size_t)n     * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n     * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pts, h_pts, (size_t)n * sizeof(Point),
                          cudaMemcpyHostToDevice));

    int grid = (n + BLOCK_SZ - 1) / BLOCK_SZ;

    knn_kernel<<<grid, BLOCK_SZ>>>(d_pts, n, k, d_out);
    CUDA_CHECK(cudaGetLastError());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    write_output("knn.txt", h_pts, h_out, n);

    cudaFree(d_pts); cudaFree(d_out);
    free(h_out);
}







static void run_approx_knn(const Point* h_pts, int n, int k) {
    // Bounding box — OpenMP parallel min/max reduction on host.
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

    // Bitonic sort requires a power-of-2 array length.
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
    // Padding slots: code = UINT32_MAX → sort to tail, never reached by window search.
    CUDA_CHECK(cudaMemset(d_morton, 0xFF, (size_t)n_padded * sizeof(uint2)));

    int grid        = (n        + BLOCK_SZ - 1) / BLOCK_SZ;
    int padded_grid = (n_padded + BLOCK_SZ - 1) / BLOCK_SZ;

    compute_morton_kernel<<<grid, BLOCK_SZ>>>(
        d_pts, n,
        xmin, ymin, zmin,
        xmax - xmin, ymax - ymin, zmax - zmin,
        d_morton);
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
    // CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    write_output("approx_knn.txt", h_pts, h_out, n);

    cudaFree(d_pts); cudaFree(d_morton); cudaFree(d_rank);
    cudaFree(d_out);
    free(h_out);
}







static void run_kmeans(const Point* h_pts, int n, int k, int T) {
    // Centroids initialized from the first k points.
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

    int grid     = (n + BLOCK_SZ - 1) / BLOCK_SZ;
    int k_grid   = (k + BLOCK_SZ - 1) / BLOCK_SZ;
    int num_warps = BLOCK_SZ / 32;
    int smem_size = num_warps * (3 * k * (int)sizeof(float) + k * (int)sizeof(int));
    int h_changed = 0;

    for (int iter = 0; iter < T; iter++) {
        // Use Async memsets on stream 0 to overlap launch overhead
        CUDA_CHECK(cudaMemsetAsync(d_changed, 0, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cx,  0, (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cy,  0, (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cz,  0, (size_t)k * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_accum_cnt, 0, (size_t)k * sizeof(int)));
        
        // Copy updated centroids to constant memory for the assign kernel
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_const_centroids, d_centroids, (size_t)k * sizeof(float4), 0, cudaMemcpyDeviceToDevice));

        kmeans_assign_and_accumulate_kernel<<<grid, BLOCK_SZ, smem_size>>>(
            d_pts, n, k, d_assignments, d_changed, d_accum_cx, d_accum_cy, d_accum_cz, d_accum_cnt);
        CUDA_CHECK(cudaGetLastError());

        // cudaMemcpy acts as a device sync; piggybacks the convergence read.
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!h_changed && iter > 0) break;

        kmeans_finalize_kernel<<<k_grid, BLOCK_SZ>>>(
            d_centroids,
            d_accum_cx, d_accum_cy, d_accum_cz, d_accum_cnt, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Build per-cluster histograms then equalize.
    int* d_cdf;
    int* d_cdf_min;
    CUDA_CHECK(cudaMalloc(&d_cdf, (size_t)k * HIST_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf_min, (size_t)k * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_cluster_hist, 0, (size_t)k * HIST_BINS * sizeof(int)));

    kmeans_hist_kernel<<<grid, BLOCK_SZ>>>(d_pts, d_assignments, n, d_cluster_hist);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    compute_cdf_kernel<<<k_grid, BLOCK_SZ>>>(d_cluster_hist, k, d_cdf, d_cdf_min, d_cluster_size);
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







int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.txt> <knn|approx_knn|kmeans>\n", argv[0]);
        return EXIT_FAILURE;
    }

    Point* h_pts = nullptr;
    int n, k, T;
    read_input(argv[1], &h_pts, &n, &k, &T);

    const char* mode = argv[2];
    if      (strcmp(mode, "knn")        == 0) run_knn(h_pts, n, k);
    else if (strcmp(mode, "approx_knn") == 0) run_approx_knn(h_pts, n, k);
    else if (strcmp(mode, "kmeans")     == 0) run_kmeans(h_pts, n, k, T);
    else {
        fprintf(stderr, "Unknown mode '%s'. Expected: knn | approx_knn | kmeans\n",
                mode);
        cudaFreeHost(h_pts);
        return EXIT_FAILURE;
    }

    cudaFreeHost(h_pts);
    return 0;
}
