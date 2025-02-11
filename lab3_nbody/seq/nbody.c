#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(Body *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].vx = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].vy = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    data[i].vz = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


void bodyForce(Body *p, float dt, int n) {
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; i++) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  const float dt = 0.01f; // time step

  int nIters = 10;  // simulation iterations
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);
  if (argc > 2) nIters = atoi(argv[2]);


  int bytes = nBodies * sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  randomizeBodies(p, nBodies); // Init pos / vel data

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    bodyForce(p, dt, nBodies); // compute interbody forces

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed;
    }
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }

  // for (int i = 0 ; i < nBodies; i++) { // integrate position
  //   printf("i: %d vx: %f vy: %f vz: %f start %d  \n",
  //         i, p[i].x, p[i].y, p[i].z, 0);
  // }

  double avgTime = (nIters - 1) / totalTime;
  float rate = (nIters - 1) / totalTime;
  printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
          nIters, rate);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);


  free(buf);
}
