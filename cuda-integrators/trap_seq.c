/* Sequential trapezoidal rule with CLI args for timing vs CUDA (Exercise 4).
 * Usage: ./trap_seq <n> <a> <b>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#ifndef ITERS
#define ITERS 50
#endif

/* Double accumulation matches ∫sin on [0,π]≈2 for large n; float sum drifts ~1e-3 over 1e6 terms. */
static double serial_trap(double a, double b, int n) {
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

int main(int argc, char *argv[]) {
   if (argc != 4) {
      fprintf(stderr, "usage: %s <n> <a> <b>\n", argv[0]);
      return 1;
   }
   int n = (int)strtol(argv[1], NULL, 10);
   double a = strtod(argv[2], NULL);
   double b = strtod(argv[3], NULL);

   double hmin = 1.0e6, hmax = 0.0, htotal = 0.0;
   double result = 0.0;
   double start, finish;

   for (int iter = 0; iter < ITERS; iter++) {
      GET_TIME(start);
      result = serial_trap(a, b, n);
      GET_TIME(finish);
      update_stats(start, finish, &hmin, &hmax, &htotal);
   }

   printf("Sequential trap integral: %.15e\n", result);
   printf("Host times:  min = %e, max = %e, avg = %e\n", hmin, hmax,
          htotal / (double)ITERS);
   return 0;
}
