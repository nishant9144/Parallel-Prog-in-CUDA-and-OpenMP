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

// ─── Constants ────────────────────────────────────────────────────────
#define BLOCK_SZ          256
#define KNN_BLOCK_SZ      128
#define MAX_K             128
#define HIST_BINS         256

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
    int x, y, z, I;
};

struct HeapNode {
    int64_t dist2;
    int     x, y, z;
    int     idx;
};

struct TreeNode {
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
bool is_worse(const HeapNode& a, const HeapNode& b) {
    if (a.dist2 != b.dist2) return a.dist2 > b.dist2;
    if (a.x     != b.x)     return a.x     > b.x;
    if (a.y     != b.y)     return a.y     > b.y;
    return a.z > b.z;
}

__device__
void heap_push(HeapNode* heap, int& sz, const HeapNode& e) {
    int i = sz++;
    heap[i] = e;
    while (i > 0) {
        int par = (i - 1) >> 1;
        if (!is_worse(heap[i], heap[par])) break;
        HeapNode tmp = heap[par]; heap[par] = heap[i]; heap[i] = tmp;
        i = par;
    }
}

__device__
void heapify_down(HeapNode* heap, int sz) {
    int i = 0;
    while (true) {
        int worst = i, l = 2*i + 1, r = 2*i + 2;
        if (l < sz && is_worse(heap[l], heap[worst])) worst = l;
        if (r < sz && is_worse(heap[r], heap[worst])) worst = r;
        if (worst == i) break;
        HeapNode tmp = heap[worst]; heap[worst] = heap[i]; heap[i] = tmp;
        i = worst;
    }
}

__device__ __forceinline__
int equalize(const Point* pts, const int* neighbors, int k, int self_I) {
    uint8_t hist[HIST_BINS] = {};
    hist[self_I]++;
    for (int j = 0; j < k; j++)
        ++hist[pts[neighbors[j]].I];

    int total = k + 1;
    int cumsum = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; v++) {
        cumsum += hist[v];
        if (cdf_min  == 0 && cumsum > 0) cdf_min  = cumsum;
        if (v == self_I)         cdf_self = cumsum;
    }
    if (total == cdf_min) return self_I;
    int val = (int)(((long long)(cdf_self - cdf_min) * 255) / (total - cdf_min));
    return max(0, min(255, val));
}






__global__ void approx_knn(
    const Point* __restrict__ pts,
    const int* __restrict__ orig, int n,
    const int* __restrict__ c_start, const int* __restrict__ c_cnt,
    Point grid_dim, Point min_pt, float inv_vs,
    int k, int* __restrict__ out_I)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int xi = pts[i].x, yi = pts[i].y, zi = pts[i].z;
    int cx = min(grid_dim.x - 1, max(0, (int)((xi - min_pt.x) * inv_vs)));
    int cy = min(grid_dim.y - 1, max(0, (int)((yi - min_pt.y) * inv_vs)));
    int cz = min(grid_dim.z - 1, max(0, (int)((zi - min_pt.z) * inv_vs)));

    // Integer heap
    long long kd[128];
    int ki[128];
    int cnt = 0;
    long long wD = 0; int wP = 0;

    for (int r = 0; r <= 4; r++) {
        int ax = max(0, cx - r), bx = min(grid_dim.x - 1, cx + r);
        int ay = max(0, cy - r), by = min(grid_dim.y - 1, cy + r);
        int az = max(0, cz - r), bz = min(grid_dim.z - 1, cz + r);

        for (int gx = ax; gx <= bx; gx++)
        for (int gy = ay; gy <= by; gy++)
        for (int gz = az; gz <= bz; gz++) {
            // Skip inner cells already checked in smaller radius
            if (r > 0 && abs(gx - cx) < r && abs(gy - cy) < r && abs(gz - cz) < r) continue;

            int cid = gx * grid_dim.y * grid_dim.z + gy * grid_dim.z + gz;
            int start = c_start[cid], c_count = c_cnt[cid];

            for (int t = start; t < start + c_count; t++) {
                if (orig[t] == orig[i]) continue;
                
                long long dx = (long long)(pts[t].x - xi);
                long long dy = (long long)(pts[t].y - yi);
                long long dz = (long long)(pts[t].z - zi);
                long long d = dx*dx + dy*dy + dz*dz;

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
    hist[pts[i].I]++;
    for (int t = 0; t < cnt; t++) hist[pts[ki[t]].I]++;

    int cdf[256]; cdf[0] = hist[0];
    for (int v = 1; v < 256; v++) cdf[v] = cdf[v - 1] + hist[v];

    int cdf_min = 0;
    for (int v = 0; v < 256; v++) if (cdf[v] > 0) { cdf_min = cdf[v]; break; }

    int Ii = pts[i].I, m = cnt + 1;
    if (m == cdf_min || m == 0) { out_I[orig[i]] = Ii; return; }
    
    int val = (int)(((long long)(cdf[Ii] - cdf_min) * 255) / (m - cdf_min));
    out_I[orig[i]] = max(0, min(255, val));
}






__constant__ int4 dc_cent[128]; 

__global__ void asgn_cluster(
    const Point* __restrict__ pts,
    int n, int k,
    int* __restrict__ asgn,
    int* __restrict__ changed,
    int* __restrict__ acx,
    int* __restrict__ acy,
    int* __restrict__ acz,
    int* __restrict__ acnt)
{
    extern __shared__ int smem[];
    const int n_warp = blockDim.x / 32;
    const int w_id   = threadIdx.x / 32;

    int* sx = smem;
    int* sy = smem + n_warp * k;
    int* sz = smem + 2 * n_warp * k;
    int* sc = smem + 3 * n_warp * k;

    for (int t = threadIdx.x; t < n_warp * k; t += blockDim.x) 
        sx[t] = sy[t] = sz[t] = sc[t] = 0;
    
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Point p = pts[i];
        int bC = 0;
        long long bD = LLONG_MAX; 

        for (int c = 0; c < k; c++) {
            int4 cent = dc_cent[c];
            long long dx = p.x - cent.x, dy = p.y - cent.y, dz = p.z - cent.z;
            long long d2 = dx*dx + dy*dy + dz*dz;

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

__global__ void km_update(
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
        centroids[c] = { acx[c] / acnt[c],
                         acy[c] / acnt[c],
                         acz[c] / acnt[c], 
                         0 
                        };
    }
}

__global__ void km_hist(const Point* __restrict__ pts,
                                    const int*   __restrict__ assignments,
                                    int n,
                                    int* __restrict__ cluster_hist)
{
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;

    int cluster = 0, intensity = 0;
    bool valid = (i < n);
    if (valid) {
        cluster   = assignments[i];
        intensity = pts[i].I;
    }

    unsigned mask = __ballot_sync(0xFFFFFFFF, valid);
    if (!valid) return;

    int lane = threadIdx.x & 31;
    int key  = (cluster << 8) | intensity;

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

__global__ void cdf(const int* __restrict__ cluster_hist,
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

__global__ void km_equalize(const Point* __restrict__ pts,
                                        const int*   __restrict__ asgn,
                                        const int*   __restrict__ cdf,
                                        const int*   __restrict__ cdf_min,
                                        const int*   __restrict__ cluster_sz,
                                        int n,
                                        int* __restrict__ out_I)
{
    int i = blockIdx.x * BLOCK_SZ + threadIdx.x;
    if (i >= n) return;

    int c  = asgn[i];
    int Ii = pts[i].I;

    int cdf_self = cdf[c * HIST_BINS + Ii];
    int cm       = cdf_min[c];
    int m        = cluster_sz[c];

    if (m == cm) {
        out_I[i] = Ii;
    } else {
        int val = (int)(((long long)(cdf_self - cm) * 255) / (m - cm));
        out_I[i] = max(0, min(255, val));
    }
}












__global__ void knn(const Point* __restrict__ pts,
                                   const TreeNode* __restrict__ tree,
                                   int root, int n, int k,
                                   int* __restrict__ out_I) {
    int qi = blockIdx.x * KNN_BLOCK_SZ + threadIdx.x;
    if (qi >= n) return;

    Point q = pts[qi];
    HeapNode heap[MAX_K];
    int heap_sz = 0;

    int stack[32];
    int stack_sz = 0;
    stack[stack_sz++] = root;

    while (stack_sz > 0) {
        int curr = stack[--stack_sz];
        if (curr == -1) continue;

        TreeNode node = tree[curr];
        Point  p    = pts[node.pt_idx];

        if (node.pt_idx != qi) {
            int64_t dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            int64_t d2 = dx*dx + dy*dy + dz*dz;

            HeapNode e = { d2, p.x, p.y, p.z, node.pt_idx };

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

    out_I[qi] = equalize(pts, neighbors, heap_sz, q.I);
}









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
                   &(*h_pts)[i].z, &(*h_pts)[i].I) != 4) {
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

    // Setup AoS layout using Point struct directly
    Point* pts = (Point*)malloc((size_t)n * sizeof(Point));
    int* orig = (int*)malloc((size_t)n * sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int oi = items[i].idx;
        pts[i]  = h_pts[oi];
        orig[i] = oi;
    }

    // Build prefix sum for cell offsets
    int* cs = (int*)calloc((size_t)t_cells, sizeof(int));
    int* cc = (int*)calloc((size_t)t_cells, sizeof(int));
    
    for (int i = 0; i < n; i++) cc[items[i].cell]++;
    for (long long c = 1; c < t_cells; c++) cs[c] = cs[c - 1] + cc[c - 1];

    Point *d_pts;
    int *d_orig, *d_cs, *d_cc, *d_out;
    CUDA_CHECK(cudaMalloc(&d_pts,  (size_t)n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_orig, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cs,   (size_t)t_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cc,   (size_t)t_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pts,  pts,  (size_t)n * sizeof(Point),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig, orig, (size_t)n * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cs,   cs,   (size_t)t_cells * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cc,   cc,   (size_t)t_cells * sizeof(int), cudaMemcpyHostToDevice));

    int blk = BLOCK_SZ, grd = (n + blk - 1) / blk;
    Point grid_dim = {Gx, Gy, Gz, 0};
    Point min_pt = {xmn, ymn, zmn, 0};
    approx_knn<<<grd, blk>>>(
        d_pts, d_orig, n,
        d_cs, d_cc, grid_dim, min_pt, 1.0f / vs, k, d_out);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    write_output("approx_knn.txt", h_pts, h_out, n);

    cudaFree(d_pts);
    cudaFree(d_orig); cudaFree(d_cs); cudaFree(d_cc); cudaFree(d_out);

    free(items); free(pts);
    free(orig); free(cs); free(cc); free(h_out);
}








static void run_kmeans(const Point* h_pts, int n, int k, int T) {
    int4* h_centroids;
    CUDA_CHECK(cudaMallocHost(&h_centroids, (size_t)k * sizeof(int4)));
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
        asgn_cluster, cudaFuncCachePreferShared));

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

        asgn_cluster<<<grid, BLOCK_SZ, smem_size>>>(
            d_pts, n, k, d_assignments, d_changed,
            d_accum_cx, d_accum_cy, d_accum_cz, d_accum_cnt);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!h_changed && iter > 0) break;

        km_update<<<k_grid, BLOCK_SZ>>>(
            d_centroids, d_accum_cx, d_accum_cy, d_accum_cz, d_accum_cnt, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int* d_cdf;
    int* d_cdf_min;
    CUDA_CHECK(cudaMalloc(&d_cdf,     (size_t)k * HIST_BINS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf_min, (size_t)k             * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cluster_hist, 0, (size_t)k * HIST_BINS * sizeof(int)));

    km_hist<<<grid, BLOCK_SZ>>>(d_pts, d_assignments, n, d_cluster_hist);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cdf<<<k_grid, BLOCK_SZ>>>(
        d_cluster_hist, k, d_cdf, d_cdf_min, d_cluster_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    km_equalize<<<grid, BLOCK_SZ>>>(
        d_pts, d_assignments, d_cdf, d_cdf_min, d_cluster_size, n, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out;
    CUDA_CHECK(cudaMallocHost(&h_out, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    write_output("kmeans.txt", h_pts, h_out, n);

    cudaFree(d_pts);          cudaFree(d_centroids);   cudaFree(d_assignments);
    cudaFree(d_changed);      cudaFree(d_accum_cx);    cudaFree(d_accum_cy);
    cudaFree(d_accum_cz);     cudaFree(d_accum_cnt);   cudaFree(d_cluster_hist);
    cudaFree(d_cluster_size); cudaFree(d_out);
    cudaFree(d_cdf);          cudaFree(d_cdf_min);
    CUDA_CHECK(cudaFreeHost(h_centroids));
    CUDA_CHECK(cudaFreeHost(h_out));
}







static int build_tree(std::vector<int>& indices, const Point* pts,
                             int start, int end, int depth,
                             std::vector<TreeNode>& nodes) {
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

    nodes[node_idx].left  = build_tree(indices, pts, start, mid, depth+1, nodes);
    nodes[node_idx].right = build_tree(indices, pts, mid+1, end, depth+1, nodes);
    return node_idx;
}





static void run_knn(const Point* h_pts, int n, int k) {
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    std::vector<TreeNode> tree_nodes;
    tree_nodes.reserve(n);
    int root = build_tree(indices, h_pts, 0, n, 0, tree_nodes);

    Point*  d_pts;
    TreeNode* d_tree;
    int*    d_out;
    CUDA_CHECK(cudaMalloc(&d_pts,  (size_t)n * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_tree, (size_t)n * sizeof(TreeNode)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pts,  h_pts,             (size_t)n * sizeof(Point),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree, tree_nodes.data(), (size_t)n * sizeof(TreeNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaFuncSetCacheConfig(knn, cudaFuncCachePreferL1));

    int grid = (n + KNN_BLOCK_SZ - 1) / KNN_BLOCK_SZ;
    knn<<<grid, KNN_BLOCK_SZ>>>(d_pts, d_tree, root, n, k, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int* h_out = (int*)malloc((size_t)n * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

    write_output("knn.txt", h_pts, h_out, n);
    cudaFree(d_pts); cudaFree(d_tree); cudaFree(d_out); free(h_out);
}






int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.txt> <knn|approx_knn|kdtree|kmeans|eval>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    Point* h_pts = nullptr;
    int n, k, T;
    read_input(argv[1], &h_pts, &n, &k, &T);

    double start_time = omp_get_wtime();

    const char* mode = argv[2];
    if      (strcmp(mode, "knn")        == 0) run_knn(h_pts, n, k);
    else if (strcmp(mode, "approx_knn") == 0) run_approx_knn(h_pts, n, k);
    else if (strcmp(mode, "kmeans")     == 0) run_kmeans(h_pts, n, k, T);
    else {
        fprintf(stderr, "Unknown mode '%s'. Expected: knn|approx_knn|kdtree|kmeans|eval\n",
                mode);
        cudaFreeHost(h_pts);
        return EXIT_FAILURE;
    }

    double end_time = omp_get_wtime();
    printf("Time taken: %f seconds\n", end_time - start_time);

    cudaFreeHost(h_pts);
    return 0;
}
