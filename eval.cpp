/*
 * eval.cpp — Sequential reference + evaluator for a2
 *
 * COMPILE:
 *   g++ -O2 -std=c++11 -o eval eval.cpp
 *
 * USAGE:
 *   # Run GPU version first to produce output files, then:
 *   ./eval input.txt
 *
 *   The evaluator:
 *     1. Runs sequential KNN   → seq_knn.txt    + reports time
 *     2. Runs sequential KMeans → seq_kmeans.txt + reports time
 *     3. Compares seq_knn.txt   vs knn.txt       (exact match)
 *     4. Compares knn.txt       vs approx_knn.txt (MAE — approx correctness)
 *     5. Compares seq_kmeans.txt vs kmeans.txt    (exact match + MAE fallback)
 *     6. Prints speedup = seq_time / GPU_time
 *        (GPU time is read from gpu_times.txt if present; see below)
 *
 * GPU TIME FILE (optional):
 *   If you redirect GPU wall-time to a file:
 *     { time ./a2 input.txt knn ; } 2>&1 | grep real > gpu_times.txt
 *   the evaluator will parse it for speedup.  Otherwise supply it manually
 *   when prompted, or pass it as a third argument:
 *     ./eval input.txt gpu_times.txt
 */

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

// ─── Constants (must match main.cu) ──────────────────────────────────────────
#define HIST_BINS         256
#define MAX_K             128
#define APPROX_WIN_FACTOR  16

// ─── Timing ───────────────────────────────────────────────────────────────────
static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ─── Data structures ──────────────────────────────────────────────────────────
struct Point { int x, y, z, intensity; };

// Neighbor entry — same field layout and tiebreak as HeapEntry in main.cu.
struct Neighbor {
    int64_t dist2;
    int     x, y, z, idx;
};

// Same total order as is_worse() in main.cu:
//   worse = larger dist2; on tie, lexicographically larger (x, y, z).
static bool is_worse(const Neighbor& a, const Neighbor& b) {
    if (a.dist2 != b.dist2) return a.dist2 > b.dist2;
    if (a.x     != b.x)     return a.x     > b.x;
    if (a.y     != b.y)     return a.y     > b.y;
    return                         a.z     > b.z;
}

// Max-heap operations — mirrors heap_push / heapify_down in main.cu.
static void heap_push(Neighbor* heap, int& sz, const Neighbor& e) {
    int i = sz++;
    heap[i] = e;
    while (i > 0) {
        int par = (i - 1) >> 1;
        if (!is_worse(heap[i], heap[par])) break;
        std::swap(heap[i], heap[par]);
        i = par;
    }
}

static void heap_sift_down(Neighbor* heap, int sz) {
    int i = 0;
    while (true) {
        int worst = i, l = 2*i+1, r = 2*i+2;
        if (l < sz && is_worse(heap[l], heap[worst])) worst = l;
        if (r < sz && is_worse(heap[r], heap[worst])) worst = r;
        if (worst == i) break;
        std::swap(heap[i], heap[worst]);
        i = worst;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  I/O helpers
// ═══════════════════════════════════════════════════════════════════════════════

static void read_input(const char* path,
                        std::vector<Point>& pts, int& n, int& k, int& T) {
    FILE* f = fopen(path, "r");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }
    if (fscanf(f, "%d %d %d", &n, &k, &T) != 3) {
        fprintf(stderr, "Bad header in %s\n", path); exit(EXIT_FAILURE);
    }
    pts.resize(n);
    for (int i = 0; i < n; i++)
        if (fscanf(f, "%d %d %d %d",
                   &pts[i].x, &pts[i].y, &pts[i].z, &pts[i].intensity) != 4) {
            fprintf(stderr, "Truncated at point %d\n", i); exit(EXIT_FAILURE);
        }
    fclose(f);
}

// Reads the output format written by write_output() in main.cu.
// Returns false if the file does not exist.
static bool read_output(const char* path, std::vector<int>& intensities, int n) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    intensities.resize(n);
    int x, y, z, v;
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%d %d %d %d", &x, &y, &z, &v) != 4) {
            fprintf(stderr, "Truncated read from %s at line %d\n", path, i);
            fclose(f); return false;
        }
        intensities[i] = v;
    }
    fclose(f);
    return true;
}

static void write_output(const char* path,
                          const std::vector<Point>& pts,
                          const std::vector<int>& intensities) {
    FILE* f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(EXIT_FAILURE); }
    int n = (int)pts.size();
    for (int i = 0; i < n; i++)
        fprintf(f, "%d %d %d %d\n",
                pts[i].x, pts[i].y, pts[i].z, intensities[i]);
    fclose(f);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Comparison metrics
// ═══════════════════════════════════════════════════════════════════════════════

struct CmpResult {
    double mae;       // mean absolute error
    int    max_err;   // largest per-point difference
    int    mismatches;
    int    n;
};

static CmpResult compare(const std::vector<int>& ref,
                          const std::vector<int>& got) {
    CmpResult r = {0.0, 0, 0, (int)ref.size()};
    for (int i = 0; i < r.n; i++) {
        int d = abs(ref[i] - got[i]);
        r.mae += d;
        r.max_err = std::max(r.max_err, d);
        if (d) r.mismatches++;
    }
    r.mae /= r.n;
    return r;
}

static void print_cmp(const char* label, const CmpResult& c, bool exact_required) {
    bool pass = exact_required ? (c.mismatches == 0) : (c.mae < 10.0);
    printf("  %-28s  MAE=%6.3f  max_err=%3d  mismatches=%d/%d  [%s]\n",
           label, c.mae, c.max_err, c.mismatches, c.n,
           pass ? "PASS" : "FAIL");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SEQUENTIAL KNN
//
//  Matches knn_kernel in main.cu exactly:
//    - k nearest neighbors found with the same tiebreak comparator
//    - equalization includes self: histogram has k+1 entries (self + k neighbors)
//    - CDF computed over the combined histogram
// ═══════════════════════════════════════════════════════════════════════════════

static int equalize_knn(const std::vector<Point>& pts,
                         const Neighbor* neighbors, int cnt,
                         int self_intensity) {
    int hist[HIST_BINS] = {};
    hist[self_intensity]++;                          // self included
    for (int j = 0; j < cnt; j++)
        hist[pts[neighbors[j].idx].intensity]++;

    int total = cnt + 1;
    int cumsum = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; v++) {
        cumsum += hist[v];
        if (cdf_min == 0 && cumsum > 0) cdf_min  = cumsum;
        if (v == self_intensity)         cdf_self = cumsum;
    }
    if (total == cdf_min) return self_intensity;
    return (int)roundf((float)(cdf_self - cdf_min) /
                       (float)(total    - cdf_min) * 255.0f);
}

static void seq_knn(const std::vector<Point>& pts, int k,
                     std::vector<int>& out_intensity) {
    int n = (int)pts.size();
    out_intensity.resize(n);
    std::vector<Neighbor> heap(k);

    for (int qi = 0; qi < n; qi++) {
        const Point& q = pts[qi];
        int heap_sz = 0;

        for (int j = 0; j < n; j++) {
            if (j == qi) continue;
            int64_t dx = q.x - pts[j].x,
                    dy = q.y - pts[j].y,
                    dz = q.z - pts[j].z;
            Neighbor e = { dx*dx + dy*dy + dz*dz,
                           pts[j].x, pts[j].y, pts[j].z, j };

            if (heap_sz < k) {
                heap_push(heap.data(), heap_sz, e);
            } else if (is_worse(heap[0], e)) {
                heap[0] = e;
                heap_sift_down(heap.data(), heap_sz);
            }
        }
        out_intensity[qi] = equalize_knn(pts, heap.data(), heap_sz, q.intensity);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SEQUENTIAL APPROX KNN  (Morton Z-order, matches approx_knn_kernel)
//
//  equalize_from_neighbors() in main.cu does NOT add self to the histogram
//  but uses total = k+1.  We reproduce that behaviour here so the sequential
//  approx output matches the GPU approx output exactly (and the MAE is
//  computed by comparing both against the exact GPU knn.txt).
// ═══════════════════════════════════════════════════════════════════════════════

static uint32_t morton_expand(uint32_t v) {
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

static uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return morton_expand(x) | (morton_expand(y) << 1) | (morton_expand(z) << 2);
}

// Matches equalize_from_neighbors() in main.cu — self NOT added to histogram,
// but total = k+1.
static int equalize_approx(const std::vector<Point>& pts,
                             const Neighbor* neighbors, int cnt,
                             int self_intensity) {
    int hist[HIST_BINS] = {};
    for (int j = 0; j < cnt; j++)
        hist[pts[neighbors[j].idx].intensity]++;

    int total = cnt + 1;
    int cumsum = 0, cdf_min = 0, cdf_self = 0;
    for (int v = 0; v < HIST_BINS; v++) {
        cumsum += hist[v];
        if (cdf_min == 0 && cumsum > 0) cdf_min  = cumsum;
        if (v == self_intensity)         cdf_self = cumsum;
    }
    if (total == cdf_min) return self_intensity;
    return (int)roundf((float)(cdf_self - cdf_min) /
                       (float)(total    - cdf_min) * 255.0f);
}

static void seq_approx_knn(const std::vector<Point>& pts, int k,
                             std::vector<int>& out_intensity) {
    int n = (int)pts.size();
    out_intensity.resize(n);

    // Bounding box
    int xmin = INT_MAX, xmax = INT_MIN;
    int ymin = INT_MAX, ymax = INT_MIN;
    int zmin = INT_MAX, zmax = INT_MIN;
    for (const auto& p : pts) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
    }
    int xr = xmax - xmin, yr = ymax - ymin, zr = zmax - zmin;

    // Compute and sort (morton_code, orig_idx)
    std::vector<std::pair<uint32_t, int>> pairs(n);
    for (int i = 0; i < n; i++) {
        uint32_t nx = xr ? (uint32_t)((int64_t)(pts[i].x - xmin) * 1023 / xr) : 0u;
        uint32_t ny = yr ? (uint32_t)((int64_t)(pts[i].y - ymin) * 1023 / yr) : 0u;
        uint32_t nz = zr ? (uint32_t)((int64_t)(pts[i].z - zmin) * 1023 / zr) : 0u;
        pairs[i] = { morton3D(nx, ny, nz), i };
    }
    std::sort(pairs.begin(), pairs.end());

    // Inverse rank map
    std::vector<int> rank(n);
    for (int pos = 0; pos < n; pos++) rank[pairs[pos].second] = pos;

    int window = std::min(n, std::max(APPROX_WIN_FACTOR * k, k + 1));
    std::vector<Neighbor> heap(k);

    for (int i = 0; i < n; i++) {
        const Point& q = pts[i];
        int p  = rank[i];
        int lo = std::max(0, p - window / 2);
        int hi = std::min(n - 1, lo + window - 1);
        lo     = std::max(0, hi - window + 1);

        int heap_sz = 0;
        for (int pos = lo; pos <= hi; pos++) {
            int j = pairs[pos].second;
            if (j == i) continue;
            int64_t dx = q.x - pts[j].x,
                    dy = q.y - pts[j].y,
                    dz = q.z - pts[j].z;
            Neighbor e = { dx*dx + dy*dy + dz*dz,
                           pts[j].x, pts[j].y, pts[j].z, j };
            if (heap_sz < k) {
                heap_push(heap.data(), heap_sz, e);
            } else if (is_worse(heap[0], e)) {
                heap[0] = e;
                heap_sift_down(heap.data(), heap_sz);
            }
        }
        out_intensity[i] = equalize_approx(pts, heap.data(), heap_sz, q.intensity);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SEQUENTIAL K-MEANS
//
//  Matches the GPU pipeline in run_kmeans() / associated kernels exactly:
//    - Init      : centroids[c] = pts[c]   (first k points)
//    - Assign    : min float sq-dist; first cluster wins on exact tie (c loop forward)
//    - Converge  : skip early-exit on iter == 0 (mirrors "iter > 0" guard in GPU)
//    - Update    : float mean; empty clusters frozen
//    - Histogram : every point counted in its assigned cluster (includes "center")
//    - Equalize  : CDF lookup, same formula as equalize_kmeans_kernel
// ═══════════════════════════════════════════════════════════════════════════════

static void seq_kmeans(const std::vector<Point>& pts, int k, int T,
                        std::vector<int>& out_intensity) {
    int n = (int)pts.size();
    out_intensity.resize(n);

    // Centroid positions as float (matches float4 on GPU)
    std::vector<float> cx(k), cy(k), cz(k);
    for (int c = 0; c < k; c++) {
        cx[c] = (float)pts[c].x;
        cy[c] = (float)pts[c].y;
        cz[c] = (float)pts[c].z;
    }

    std::vector<int> asgn(n, 0);

    for (int iter = 0; iter < T; iter++) {
        bool changed = false;

        // Assignment step
        for (int i = 0; i < n; i++) {
            float px = (float)pts[i].x, py = (float)pts[i].y, pz = (float)pts[i].z;
            int   best_c = 0;
            float best_d = FLT_MAX;
            for (int c = 0; c < k; c++) {
                float dx = px - cx[c], dy = py - cy[c], dz = pz - cz[c];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best_d) { best_d = d2; best_c = c; }
            }
            if (asgn[i] != best_c) { asgn[i] = best_c; changed = true; }
        }

        // Mirror GPU guard: skip convergence check on first iteration
        if (!changed && iter > 0) break;

        // Update step — float accumulation (matches GPU float atomics)
        std::vector<float> ncx(k, 0.f), ncy(k, 0.f), ncz(k, 0.f);
        std::vector<int>   cnt(k, 0);
        for (int i = 0; i < n; i++) {
            int c = asgn[i];
            ncx[c] += (float)pts[i].x;
            ncy[c] += (float)pts[i].y;
            ncz[c] += (float)pts[i].z;
            cnt[c]++;
        }
        for (int c = 0; c < k; c++) {
            if (cnt[c] > 0) {
                cx[c] = ncx[c] / cnt[c];
                cy[c] = ncy[c] / cnt[c];
                cz[c] = ncz[c] / cnt[c];
            }
            // Empty clusters: frozen (same as GPU kmeans_finalize_kernel)
        }
    }

    // Build per-cluster histogram (all assigned points, including "center")
    std::vector<int> hist(k * HIST_BINS, 0);
    for (int i = 0; i < n; i++)
        hist[asgn[i] * HIST_BINS + pts[i].intensity]++;

    // Compute CDF and cluster sizes (matches compute_cdf_kernel)
    std::vector<int> cdf(k * HIST_BINS, 0);
    std::vector<int> cdf_min(k, 0);
    std::vector<int> cluster_sz(k, 0);
    for (int c = 0; c < k; c++) {
        int sum = 0, first_nonzero = 0;
        for (int v = 0; v < HIST_BINS; v++) {
            sum += hist[c * HIST_BINS + v];
            cdf[c * HIST_BINS + v] = sum;
            if (first_nonzero == 0 && sum > 0) first_nonzero = sum;
        }
        cdf_min[c]    = first_nonzero;
        cluster_sz[c] = sum;
    }

    // Equalize (matches equalize_kmeans_kernel)
    for (int i = 0; i < n; i++) {
        int c    = asgn[i];
        int Ii   = pts[i].intensity;
        int cs   = cdf[c * HIST_BINS + Ii];
        int cm   = cdf_min[c];
        int m    = cluster_sz[c];
        if (m == cm) {
            out_intensity[i] = Ii;
        } else {
            out_intensity[i] = (int)roundf(
                (float)(cs - cm) / (float)(m - cm) * 255.0f);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s input.txt [gpu_time_knn_s gpu_time_kmeans_s]\n"
            "  gpu_time_*_s  optional: GPU wall-time in seconds for speedup\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    double gpu_time_knn    = 0.0;
    double gpu_time_kmeans = 0.0;
    if (argc >= 4) {
        gpu_time_knn    = atof(argv[2]);
        gpu_time_kmeans = atof(argv[3]);
    }

    std::vector<Point> pts;
    int n, k, T;
    read_input(argv[1], pts, n, k, T);
    printf("Input: n=%d  k=%d  T=%d\n\n", n, k, T);

    // ── Sequential KNN ────────────────────────────────────────────────────────
    printf("[ Sequential KNN ]\n");
    std::vector<int> seq_knn_out;
    double t0 = now_sec();
    seq_knn(pts, k, seq_knn_out);
    double seq_knn_time = now_sec() - t0;
    printf("  Time : %.3f s\n", seq_knn_time);
    write_output("seq_knn.txt", pts, seq_knn_out);
    printf("  Written: seq_knn.txt\n");

    // Compare against GPU knn.txt
    std::vector<int> gpu_knn_out;
    if (read_output("knn.txt", gpu_knn_out, n)) {
        CmpResult r = compare(seq_knn_out, gpu_knn_out);
        print_cmp("seq_knn vs knn.txt", r, true);
        if (gpu_time_knn > 0.0)
            printf("  Speedup (KNN): %.2fx\n", seq_knn_time / gpu_time_knn);
    } else {
        printf("  knn.txt not found — run ./a2 input.txt knn first\n");
    }

    // ── Approx KNN correctness ────────────────────────────────────────────────
    // Correctness = MAE between approx output and exact KNN output.
    // Both files should be GPU outputs (knn.txt and approx_knn.txt).
    printf("\n[ Approx KNN correctness vs Exact KNN ]\n");

    // Sequential approx (for verifying the GPU approx is consistent)
    std::vector<int> seq_approx_out;
    t0 = now_sec();
    seq_approx_knn(pts, k, seq_approx_out);
    double seq_approx_time = now_sec() - t0;
    printf("  Sequential approx time: %.3f s\n", seq_approx_time);
    write_output("seq_approx_knn.txt", pts, seq_approx_out);

    // Primary metric: GPU approx vs GPU exact
    std::vector<int> gpu_approx_out;
    if (read_output("approx_knn.txt", gpu_approx_out, n)) {
        if (!gpu_knn_out.empty()) {
            CmpResult r = compare(gpu_knn_out, gpu_approx_out);
            print_cmp("knn.txt vs approx_knn.txt", r, false);
            printf("  (Threshold TBD by grader — lower MAE is better)\n");

            // Speedup of GPU approx over GPU exact KNN
            // (requires both GPU times; printed only if both supplied)
            double gpu_approx_time = 0.0;
            if (argc >= 5) {
                gpu_approx_time = atof(argv[4]);
                if (gpu_time_knn > 0.0 && gpu_approx_time > 0.0)
                    printf("  Speedup (approx vs knn, GPU): %.2fx\n",
                           gpu_time_knn / gpu_approx_time);
            }
        } else {
            printf("  knn.txt not found — cannot compute MAE\n");
        }
        // Also show sequential approx vs sequential exact (sanity check)
        CmpResult r2 = compare(seq_knn_out, seq_approx_out);
        print_cmp("seq_knn vs seq_approx_knn", r2, false);
        // GPU approx vs sequential approx (algorithm consistency)
        CmpResult r3 = compare(seq_approx_out, gpu_approx_out);
        print_cmp("seq_approx vs approx_knn.txt", r3, true);
    } else {
        printf("  approx_knn.txt not found — run ./a2 input.txt approx_knn first\n");
    }

    // ── Sequential K-means ────────────────────────────────────────────────────
    printf("\n[ Sequential K-Means ]\n");
    std::vector<int> seq_km_out;
    t0 = now_sec();
    seq_kmeans(pts, k, T, seq_km_out);
    double seq_km_time = now_sec() - t0;
    printf("  Time : %.3f s\n", seq_km_time);
    write_output("seq_kmeans.txt", pts, seq_km_out);
    printf("  Written: seq_kmeans.txt\n");

    std::vector<int> gpu_km_out;
    if (read_output("kmeans.txt", gpu_km_out, n)) {
        CmpResult r = compare(seq_km_out, gpu_km_out);
        // K-means: float accumulation order can differ between CPU and GPU,
        // so we check exact match first and fall back to MAE tolerance.
        print_cmp("seq_kmeans vs kmeans.txt", r, false);
        if (r.mismatches == 0)
            printf("  (Exact match)\n");
        else
            printf("  (Float rounding in centroid updates may cause small differences)\n");
        if (gpu_time_kmeans > 0.0)
            printf("  Speedup (K-Means): %.2fx\n", seq_km_time / gpu_time_kmeans);
    } else {
        printf("  kmeans.txt not found — run ./a2 input.txt kmeans first\n");
    }

    printf("\n[ Sequential reference times summary ]\n");
    printf("  KNN    : %.3f s\n", seq_knn_time);
    printf("  Approx : %.3f s\n", seq_approx_time);
    printf("  KMeans : %.3f s\n", seq_km_time);
    if (gpu_time_knn > 0.0 || gpu_time_kmeans > 0.0) {
        printf("\n[ Speedup summary ]\n");
        if (gpu_time_knn    > 0.0)
            printf("  KNN    speedup : %.2fx\n", seq_knn_time    / gpu_time_knn);
        if (gpu_time_kmeans > 0.0)
            printf("  KMeans speedup : %.2fx\n", seq_km_time / gpu_time_kmeans);
    }

    return 0;
}