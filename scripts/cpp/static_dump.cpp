/* Dumps one BDF-1 step of the C++ fluid solver's static mode (a fixed surface; the
   static_stokes example: the unit sphere with one toroidal velocity mode), for
   test/stokesChecks.ts: the preconditioned operator applied to a random vector, the
   preconditioned right-hand side, and the step's solution, in the solver's ref_dump format.
   Packed vectors are the solver's real embedding (FluidOpPack); spectra are on the working
   band.

     static_dump <out_dir> <lmax> <factor> <nlat_f> <nphi_f> <dt> <degree> <re> <be> */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "backend_init.h"
#include "backend_mem.h"
#include "fluid_examples.h"
#include "fluid_grid.h"
#include "fluid_op.h"
#include "fluid_stepper.h"
#include "kernels.h"
#include "ref_dump.h"

static std::vector<double> Down(const breal *d, long n)
{
  std::vector<double> v(n);
  BufDownload(v.data(), d, n, NULL);
  return v;
}

int main(int argc, char **argv)
{
  if (argc != 10) {
    std::fprintf(stderr, "usage: %s out_dir lmax factor nlat_f nphi_f dt degree re be\n", argv[0]);
    return 1;
  }
  const std::string dir  = argv[1];
  const int         lmax = std::atoi(argv[2]), factor = std::atoi(argv[3]);
  const int         nlat_f = std::atoi(argv[4]), nphi_f = std::atoi(argv[5]);
  const double      dt = std::atof(argv[6]);
  const int         degree = std::atoi(argv[7]);
  const double      re = std::atof(argv[8]), be = std::atof(argv[9]);
  int               bargc = 1;
  if (BackendInit(&bargc, &argv)) return 1;

  FluidGrid g;
  if (FluidGridCreate(&g, lmax, factor, nlat_f, nphi_f, 0)) return 1;
  FluidStepParams prm;
  prm.re        = re;
  prm.be        = be;
  prm.is_static = 1;
  prm.tol       = 1e-12;
  FluidState s0, s1;
  FluidStateCreate(&s0, &g);
  FluidStateCreate(&s1, &g);
  if (FluidExampleInit(&g, "static_stokes", 0.0, degree, &s0)) return 1;
  FluidStepper st;
  if (FluidStepperCreate(&st, &g, &prm)) return 1;

  /* The operator as FluidStep sets it up for a static BDF-1 step. */
  const long nc = 2 * g.nlm;
  FluidOpSetGeometry(&st.op, s0.d_x);
  FluidOpParams op;
  op.mass      = 1.0 / dt;
  op.disp      = dt;
  op.re        = re;
  op.be        = be;
  op.bending   = prm.bending;
  op.is_static = 1;
  op.precond   = prm.precond;
  FluidOpSetTransport(&st.op, s0.d_u, nullptr, 1);
  FluidOpSetupSolver(&st.op, &op);
  const long n = st.op.n;

  /* A random vector: spectra decaying with degree, m = 0 real, packed. */
  std::mt19937_64                  rng(12345);
  std::normal_distribution<double> N(0.0, 1.0);
  std::vector<double>              spec(4 * nc, 0.0);
  for (int f = 0; f < 4; f++)
    for (int m = 0; m <= lmax; m++)
      for (int l = m; l <= lmax; l++) {
        const long k = FluidLmIndex(lmax, l, m);
        spec[f * nc + 2 * k]     = N(rng) / (1.0 + l);
        spec[f * nc + 2 * k + 1] = m == 0 ? 0.0 : N(rng) / (1.0 + l);
      }
  breal *d_spec = BufAlloc(4 * nc), *d_z = BufAlloc(n), *d_az = BufAlloc(n), *d_b = BufAlloc(n);
  breal *d_hist = BufAlloc(3 * nc), *d_yk = BufAlloc(3 * nc);
  BufUpload(d_spec, spec.data(), 4 * nc, NULL);
  FluidOpPack(&st.op, d_spec, d_spec + 3 * nc, nullptr, d_z);
  FluidOpMatvec(&st.op, d_z, d_az);
  BufCopy(d_hist, s0.d_u, 3 * nc, NULL);
  KScale(d_hist, 1.0 / dt, 3 * nc, NULL);
  KFFill(d_yk, 0.0, 3 * nc, NULL);
  FluidOpRhs(&st.op, d_hist, d_yk, d_b);
  BackendDeviceSync();
  const std::vector<double> z = Down(d_z, n), az = Down(d_az, n), b = Down(d_b, n);
  const double area = st.op.pc.area, mean_h2 = st.op.pc.mean_H2;

  FluidStepInfo info;
  if (FluidStep(&st, &s0, nullptr, dt, 1, &s1, &info)) return 1;

  RefWriter wr(dir);
  wr.scalar("lmax", g.lmax);
  wr.scalar("lmax_f", g.lmax_f);
  wr.scalar("nlat_f", g.nlat_f);
  wr.scalar("nphi_f", g.nphi_f);
  wr.scalar("dt", dt);
  wr.scalar("degree", degree);
  wr.scalar("re", re);
  wr.scalar("be", be);
  wr.scalar("area", area);
  wr.scalar("mean_H2", mean_h2);
  wr.scalar("iterations", info.iterations);
  wr.scalar("rel_resid", info.rel_resid);
  const std::string sp = std::to_string(g.nlm), sn = std::to_string(n);
  wr.array("X", Down(s0.d_x, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  wr.array("u0", Down(s0.d_u, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  wr.array("z", z.data(), n, false, sn);
  wr.array("Az", az.data(), n, false, sn);
  wr.array("b", b.data(), n, false, sn);
  wr.array("u1", Down(s1.d_u, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  wr.array("p1", Down(s1.d_p, nc).data(), g.nlm, true, sp);
  if (!wr.finish()) return 1;
  std::printf("wrote %s: lmax %d, product band %d on %d x %d, n %ld, %d GMRES its, residual %.2e\n",
              dir.c_str(), g.lmax, g.lmax_f, g.nlat_f, g.nphi_f, n, info.iterations, info.rel_resid);

  for (breal *d : {d_spec, d_z, d_az, d_b, d_hist, d_yk}) BufFree(d);
  FluidStepperDestroy(&st);
  FluidStateDestroy(&s1);
  FluidStateDestroy(&s0);
  FluidGridDestroy(&g);
  BackendFinalize();
  return 0;
}
