#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DT 0.0001
#define NUM_THREADS 4
#define TABLE_SIZE 1801

#include "ex3.h"

// ChatGPT goofed
// Function profile
//static double function_profile[TABLE_SIZE] = {
//    122000.000000, 121999.989005, 121999.956018, 121999.900041, 121999.821078, 
    // truncated for brevity
//};

// Integrate the function profile using the left Riemann sum method
void integrate(double dt, int num_threads, double* velocity, double* position) {
    int i, tid;
    double vel = 0.0, pos = 0.0;
    #pragma omp parallel for num_threads(num_threads) \
        reduction(+: vel) reduction(+: pos)
    for (i = 0; i < TABLE_SIZE - 1; i++) {
        tid = omp_get_thread_num();
        vel += (DefaultProfile[i+1] - DefaultProfile[i]) / dt;
        pos += DefaultProfile[i];
    }
    *velocity = vel;
    *position = pos;
}

int main(int argc, char** argv) {
    double velocity, position;
    double start_time, end_time;
    
    // Set the number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Integrate the function profile
    start_time = omp_get_wtime();
    integrate(DT, NUM_THREADS, &velocity, &position);
    end_time = omp_get_wtime();
    
    // Print results
    printf("Velocity: %f\n", velocity);
    printf("Position: %f\n", position);
    printf("Elapsed time: %f seconds\n", end_time - start_time);
    
    return 0;
}
