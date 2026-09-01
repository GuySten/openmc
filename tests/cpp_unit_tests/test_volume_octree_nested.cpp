#include "openmc/volume_octree_math.h"
#include <cstdio>
using namespace openmc;
static int fails = 0;
#define CK(c, ...)                                                             \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("FAIL ");                                                         \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      ++fails;                                                                 \
    }                                                                          \
  } while (0)
int main()
{
  // TRISO-like concentric spheres. Shell volumes are 4/3 pi (ra^3 - rb^3).
  double R[5] = {
    0.02125, 0.03125, 0.03525, 0.03875, 0.04275}; // kernel..OPyC, cm
  QuadricForm q;
  q.A[0] = q.A[1] = q.A[2] = 1;
  q.c = -1;
  AxisSymProfile p;
  as_axis_symmetric(q, p);
  double box = 0.05;
  vector<double> px {-box, box, box, -box}, py {-box, -box, box, box};
  printf("TRISO concentric shells (one node each, no subdivision)\n");
  for (int i = 0; i < 5; ++i) {
    bool has_b = i > 0;
    double Ka = -R[i] * R[i], Kb = has_b ? -R[i - 1] * R[i - 1] : 0.0;
    double lo, hi;
    axis_sym_annulus_bounds(
      p, true, Ka, has_b, Kb, -box, box, px, py, 1 << 16, lo, hi);
    double exact =
      4.0 / 3.0 * M_PI *
      (R[i] * R[i] * R[i] - (has_b ? R[i - 1] * R[i - 1] * R[i - 1] : 0.0));
    bool ok = lo <= exact + 1e-15 && exact <= hi + 1e-15;
    printf("  layer %d  [%.12f, %.12f] exact %.12f  rel width %.2e  %s\n", i,
      lo, hi, exact, (hi - lo) / exact, ok ? "brackets" : "*** VIOLATION ***");
    CK(ok, "layer %d", i);
  }
  // the shells plus the surrounding matrix must fill the box
  {
    double lo, hi;
    axis_sym_annulus_bounds(
      p, false, 0.0, true, -R[4] * R[4], -box, box, px, py, 1 << 16, lo, hi);
    double exact = 8 * box * box * box - 4.0 / 3.0 * M_PI * R[4] * R[4] * R[4];
    printf("  matrix   [%.12f, %.12f] exact %.12f  %s\n", lo, hi, exact,
      (lo <= exact + 1e-15 && exact <= hi + 1e-15) ? "brackets"
                                                   : "*** VIOLATION ***");
    CK(lo <= exact + 1e-15 && exact <= hi + 1e-15, "matrix");
  }
  // a shell clipped by the box, against a dense grid
  {
    double b = 0.03;
    vector<double> qx {-b, b, b, -b}, qy {-b, -b, b, b};
    double lo, hi;
    axis_sym_annulus_bounds(p, true, -R[2] * R[2], true, -R[1] * R[1], -b, b,
      qx, qy, 1 << 16, lo, hi);
    long in = 0;
    int M = 460;
    double vol = 8 * b * b * b;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < M; ++k) {
          double x = -b + 2 * b * (i + 0.5) / M, y = -b + 2 * b * (j + 0.5) / M,
                 z = -b + 2 * b * (k + 0.5) / M;
          double d = x * x + y * y + z * z;
          if (d < R[2] * R[2] && d > R[1] * R[1])
            ++in;
        }
    double ref = vol * in / ((double)M * M * M);
    printf("  clipped shell [%.10f, %.10f] grid ref %.10f  %s\n", lo, hi, ref,
      (lo - 2e-6 <= ref && ref <= hi + 2e-6) ? "consistent"
                                             : "*** MISMATCH ***");
    CK(lo - 2e-6 <= ref && ref <= hi + 2e-6, "clipped shell");
  }
  printf(fails ? "\n%d FAILURES\n" : "\nnested-shell bounds ok\n", fails);
  return fails ? 1 : 0;
}
