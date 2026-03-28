/*
 * CSCI 551 Exercise 4 — train simulation (numerical integration only).
 *
 * Integrates a(t) -> v(t) -> x(t) with explicit Euler and linear interpolation
 * of the 1 Hz acceleration table in ex4.h (no closed-form antiderivatives).
 *
 * MPI: time index range is split across ranks; each rank receives (v,x) from
 * rank-1 and sends final (v,x) to rank+1 (pipeline; see README for scaling).
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ex4.h"

static const double TARGET_M = 122000.0;
static const double V_MAX_KMH = 320.0;

static int tab_len(void) {
   return (int)(sizeof(DefaultProfile) / sizeof(DefaultProfile[0]));
}

/* Linear interpolation between integer-second samples; table index i = floor(t) for t in [0,1800]. */
static double accel_interp(double t) {
   const int n = tab_len();
   if (t <= 0.0) {
      return DefaultProfile[0];
   }
   if (t >= (double)(n - 1)) {
      return DefaultProfile[n - 1];
   }
   int i = (int)floor(t);
   if (i >= n - 1) {
      return DefaultProfile[n - 1];
   }
   double frac = t - (double)i;
   return DefaultProfile[i] * (1.0 - frac) + DefaultProfile[i + 1] * frac;
}

static void split_range(long long N, int rank, int P, long long *lo, long long *hi) {
   const long long base = N / (long long)P;
   const long long extra = N % (long long)P;
   const long long before = (long long)rank < extra ? (long long)rank : extra;
   const long long my_count = base + ((long long)rank < extra ? 1LL : 0LL);
   *lo = (long long)rank * base + before;
   *hi = *lo + my_count;
}

static void usage(int rank) {
   if (rank != 0) {
      return;
   }
   fprintf(stderr,
           "Usage: mpirun -np P ./train_ex4 --dt <seconds> [--T <seconds>]\n"
           "  --dt   time step (e.g. 0.001 or 0.0001)\n"
           "  --T    duration seconds (default 1800)\n"
           "Prints: wall time, final x (m), final v (m/s), peak |v| (m/s), distance error (m).\n");
}

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank = 0;
   int P = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &P);

   double dt = -1.0;
   double T = 1800.0;
   for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
         dt = strtod(argv[++i], NULL);
      } else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
         T = strtod(argv[++i], NULL);
      } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
         usage(rank);
         MPI_Finalize();
         return 0;
      }
   }

   if (!(dt > 0.0) || !(T > 0.0)) {
      usage(rank);
      MPI_Finalize();
      return 1;
   }

   const long long N = (long long)llround(T / dt);
   if (N < 1) {
      if (rank == 0) {
         fprintf(stderr, "Invalid N steps.\n");
      }
      MPI_Finalize();
      return 1;
   }

   long long lo = 0;
   long long hi = N;
   split_range(N, rank, P, &lo, &hi);

   MPI_Barrier(MPI_COMM_WORLD);
   const double t_wall0 = MPI_Wtime();

   double v = 0.0;
   double x = 0.0;
   double vmax = 0.0;

   if (rank > 0) {
      MPI_Recv(&v, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&x, 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   }

   for (long long k = lo; k < hi; ++k) {
      double t = (double)k * dt;
      double a = accel_interp(t);
      v += a * dt;
      x += v * dt;
      const double av = fabs(v);
      if (av > vmax) {
         vmax = av;
      }
   }

   if (rank < P - 1) {
      MPI_Send(&v, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
      MPI_Send(&x, 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   const double t_wall1 = MPI_Wtime();
   const double elapsed = t_wall1 - t_wall0;

   double gmax = 0.0;
   MPI_Reduce(&vmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   double final_v = 0.0;
   double final_x = 0.0;
   if (rank == P - 1) {
      final_v = v;
      final_x = x;
   }
   if (P > 1) {
      MPI_Bcast(&final_v, 1, MPI_DOUBLE, P - 1, MPI_COMM_WORLD);
      MPI_Bcast(&final_x, 1, MPI_DOUBLE, P - 1, MPI_COMM_WORLD);
   }

   if (rank == 0) {
      const double v_max_ms = V_MAX_KMH * (1000.0 / 3600.0);
      const double err_m = fabs(final_x - TARGET_M);
      printf("train_ex4: dt=%.6g s, T=%.3f s, N=%lld steps, MPI ranks=%d\n", dt, T,
             (long long)N, P);
      printf("  wall_time_s=%.6f\n", elapsed);
      printf("  final_position_m=%.3f  (target %.3f m, error %.3f m)\n", final_x, TARGET_M,
             err_m);
      printf("  final_velocity_m_s=%.6f  (%.3f km/h)\n", final_v, final_v * 3.6);
      printf("  peak_abs_velocity_m_s=%.6f  (%.3f km/h)  [limit %.0f km/h = %.3f m/s]\n",
             gmax, gmax * 3.6, V_MAX_KMH, v_max_ms);
   }

   MPI_Finalize();
   return 0;
}
