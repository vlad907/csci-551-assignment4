/* Exercise 4: CUDA trapezoidal rule for sin(x) on [a,b].
 *
 * Original starter (cuda_trap1 from class) used:
 *   ./cuda_trap1 <n> <a> <b> <blk_ct> <th_per_blk>
 *   n        — number of trapezoids
 *   a, b     — interval endpoints
 *   blk_ct   — CUDA grid: number of thread blocks
 *   th_per_blk — threads per block; must satisfy blk_ct * th_per_blk >= n
 *              so each interior index i in 1..n-1 is handled by some thread.
 *
 * This version only requires steps (n) and optional a, b; it picks a 1-D grid
 * with threads_per_block = min(1024, n) and enough blocks to cover n threads.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#ifndef ITERS
#define ITERS 50
#endif

__host__ __device__ float f(float x) { return sinf(x); }

__global__ void Dev_trap(const float a, const float b, const float h, const int n,
                         float *trap_p) {
   int my_i = blockDim.x * blockIdx.x + threadIdx.x;
   if (0 < my_i && my_i < n) {
      float my_x = a + my_i * h;
      atomicAdd(trap_p, f(my_x));
   }
}

static void compute_grid(int n, int *blk_ct, int *th_per_blk) {
   const int max_tpb = 1024;
   *th_per_blk = n < max_tpb ? n : max_tpb;
   if (*th_per_blk < 1) *th_per_blk = 1;
   *blk_ct = (n + *th_per_blk - 1) / *th_per_blk;
}

static void trap_wrapper(const float a, const float b, const int n, float *trap_p,
                         int blk_ct, int th_per_blk) {
   *trap_p = 0.5f * (f(a) + f(b));
   float h = (b - a) / n;
   Dev_trap<<<blk_ct, th_per_blk>>>(a, b, h, n, trap_p);
   cudaDeviceSynchronize();
   *trap_p = h * (*trap_p);
}

/* Reference sum in double; GPU path uses float + atomicAdd (order-dependent). */
static double serial_trap_double(double a, double b, int n) {
   double h = (b - a) / n;
   double trap = 0.5 * (sin(a) + sin(b));
   for (int i = 1; i <= n - 1; i++) {
      double x = a + i * h;
      trap += sin(x);
   }
   return trap * h;
}

static void update_stats(double start, double finish, double *min_p, double *max_p,
                         double *total_p) {
   double elapsed = finish - start;
   if (elapsed < *min_p) *min_p = elapsed;
   if (elapsed > *max_p) *max_p = elapsed;
   *total_p += elapsed;
}

static void usage(const char *prog) {
   fprintf(stderr, "usage: %s <n> [a] [b]\n", prog);
   fprintf(stderr, "  n: number of trapezoids (required)\n");
   fprintf(stderr, "  a, b: interval (default 0 and pi); integral of sin is 2 on [0,pi]\n");
}

int main(int argc, char *argv[]) {
   int n;
   float a, b;
   int blk_ct, th_per_blk;

   if (argc < 2 || argc > 4) {
      usage(argv[0]);
      return 1;
   }
   n = (int)strtol(argv[1], NULL, 10);
   if (argc >= 4) {
      a = (float)strtod(argv[2], NULL);
      b = (float)strtod(argv[3], NULL);
   } else {
      a = 0.0f;
      b = (float)M_PI;
   }

   compute_grid(n, &blk_ct, &th_per_blk);
   printf("Auto grid: %d blocks x %d threads/block (>= n=%d indices)\n", blk_ct,
          th_per_blk, n);

   float *trap_p = NULL;
   cudaError_t err = cudaMallocManaged(&trap_p, sizeof(float));
   if (err != cudaSuccess) {
      fprintf(stderr, "cudaMallocManaged: %s\n", cudaGetErrorString(err));
      return 1;
   }

   double dmin = 1.0e6, dmax = 0.0, dtotal = 0.0;
   double hmin = 1.0e6, hmax = 0.0, htotal = 0.0;
   double trap_cpu;
   double start, finish;

   for (int iter = 0; iter < ITERS; iter++) {
      *trap_p = 0.0f;
      GET_TIME(start);
      trap_wrapper(a, b, n, trap_p, blk_ct, th_per_blk);
      GET_TIME(finish);
      update_stats(start, finish, &dmin, &dmax, &dtotal);

      GET_TIME(start);
      trap_cpu = serial_trap_double((double)a, (double)b, n);
      GET_TIME(finish);
      update_stats(start, finish, &hmin, &hmax, &htotal);
   }

   printf("CUDA trapezoidal estimate: %e\n", (double)*trap_p);
   printf("CPU verification (double trap sum): %.15e\n", trap_cpu);
   printf("Device times:  min = %e, max = %e, avg = %e\n", dmin, dmax,
          dtotal / (double)ITERS);
   printf("  Host times:  min = %e, max = %e, avg = %e\n", hmin, hmax,
          htotal / (double)ITERS);

   double t_cuda = dtotal / (double)ITERS;
   double t_cpu = htotal / (double)ITERS;
   if (t_cuda > 0.0) {
      double speedup = t_cpu / t_cuda;
      printf("Speedup (avg CPU trap / avg CUDA trap): %.4f\n", speedup);
      /* Parallel fraction estimate from measured speedup (Amdahl, large N): P ≈ 1 - 1/S */
      if (speedup > 1.0) {
         double p_frac = 1.0 - 1.0 / speedup;
         printf("Estimated parallel fraction from speedup: %.4f (%.2f %%)\n", p_frac,
                p_frac * 100.0);
      }
   }

   cudaFree(trap_p);
   return 0;
}
