/* Dumps BDF-1 steps of the C++ fluid solver for one example on a moving surface (not the
   static mode), for test/movingChecks.ts: the conversion of a random Cartesian field to
   radial, spheroidal and toroidal potentials and back (FluidCartesianToQst,
   FluidQstToCartesian), the preconditioned operator on a random vector, the preconditioned
   right-hand side, the state after one step and after nsteps steps, in the solver's ref_dump
   format. With bending, also the bending force at the initial surface and its derivative along
   a random displacement (FluidBendingJvp). Packed vectors are the solver's real embedding (FluidOpPack); spectra are on the
   working band.

     step_dump <out_dir> <example> <lmax> <factor> <nlat_f> <nphi_f> <dt> <re> <be> <bending>
               <nsteps> [r0]

   With r0 > 0 the example's velocity starts on Krause & Voigt's perturbed sphere of that
   amplitude instead of its own surface. */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "backend_init.h"
#include "backend_mem.h"
#include "fluid_bending.h"
#include "fluid_examples.h"
#include "fluid_grid.h"
#include "fluid_op.h"
#include "fluid_precond.h"
#include "fluid_stepper.h"
#include "kernels.h"
#include "ref_dump.h"

static std::vector<double> Down(const breal *d, long n)
{
  std::vector<double> v(n);
  BufDownload(v.data(), d, n, NULL);
  return v;
}

/* Random spectra on the working band, decaying with degree, m = 0 real. */
static std::vector<double> RandomSpectra(std::mt19937_64 &rng, int lmax, int nfields)
{
  std::normal_distribution<double> N(0.0, 1.0);
  const long                       nlm = FluidLmIndex(lmax, lmax, lmax) + 1, nc = 2 * nlm;
  std::vector<double>              s(nfields * nc, 0.0);
  for (int f = 0; f < nfields; f++)
    for (int m = 0; m <= lmax; m++)
      for (int l = m; l <= lmax; l++) {
        const long k     = FluidLmIndex(lmax, l, m);
        s[f * nc + 2 * k]     = N(rng) / (1.0 + l);
        s[f * nc + 2 * k + 1] = m == 0 ? 0.0 : N(rng) / (1.0 + l);
      }
  return s;
}

int main(int argc, char **argv)
{
  if (argc != 12 && argc != 13) {
    std::fprintf(stderr,
                 "usage: %s out_dir example lmax factor nlat_f nphi_f dt re be bending nsteps [r0]\n",
                 argv[0]);
    return 1;
  }
  const std::string dir = argv[1], example = argv[2];
  const int         lmax = std::atoi(argv[3]), factor = std::atoi(argv[4]);
  const int         nlat_f = std::atoi(argv[5]), nphi_f = std::atoi(argv[6]);
  const double      dt = std::atof(argv[7]), re = std::atof(argv[8]), be = std::atof(argv[9]);
  const int         bending = std::atoi(argv[10]), nsteps = std::atoi(argv[11]);
  const double      r0 = argc == 13 ? std::atof(argv[12]) : 0.0;
  int               bargc = 1;
  if (BackendInit(&bargc, &argv)) return 1;

  FluidGrid g;
  if (FluidGridCreate(&g, lmax, factor, nlat_f, nphi_f, 0)) return 1;
  FluidStepParams prm;
  prm.re      = re;
  prm.be      = be;
  prm.bending = bending;
  prm.volume  = FluidExampleVolume(example);
  prm.tol     = 1e-12;
  FluidState s[3];
  for (auto &x : s) FluidStateCreate(&x, &g);
  if (FluidExampleInit(&g, example, 0.4, 2, &s[0])) return 1;
  if (r0 > 0.0) {
    if (FluidExampleInit(&g, "perturbed_sphere", r0, 2, &s[1])) return 1;
    BufCopy(s[0].d_x, s[1].d_x, 3 * 2 * g.nlm, NULL);
  }
  FluidStepper st;
  if (FluidStepperCreate(&st, &g, &prm)) return 1;

  /* The operator as FluidStep sets it up for a BDF-1 step without a gauge velocity. */
  const long nc = 2 * g.nlm;
  FluidOpSetGeometry(&st.op, s[0].d_x);
  FluidOpParams op;
  op.mass    = 1.0 / dt;
  op.disp    = dt;
  op.re      = re;
  op.be      = be;
  op.bending = bending;
  op.volume  = prm.volume;
  op.precond = prm.precond;
  FluidOpSetTransport(&st.op, s[0].d_u, nullptr, 0);
  FluidOpSetupSolver(&st.op, &op);
  const long n = st.op.n;

  std::mt19937_64           rng(12345);
  const std::vector<double> cart = RandomSpectra(rng, lmax, 3), spec = RandomSpectra(rng, lmax, 4);
  breal *d_cart = BufAlloc(3 * nc), *d_qst = BufAlloc(3 * nc), *d_back = BufAlloc(3 * nc);
  breal *d_spec = BufAlloc(4 * nc), *d_z = BufAlloc(n), *d_az = BufAlloc(n), *d_b = BufAlloc(n);
  breal *d_hist = BufAlloc(3 * nc), *d_yk = BufAlloc(3 * nc);
  BufUpload(d_cart, cart.data(), 3 * nc, NULL);
  FluidCartesianToQst(&st.op.pc.qst, &g, d_cart, d_qst);
  FluidQstToCartesian(&st.op.pc.qst, &g, d_qst, d_back);
  BufUpload(d_spec, spec.data(), 4 * nc, NULL);
  KFFill(d_z, 0.0, n, NULL);
  FluidOpPack(&st.op, d_spec, d_spec + 3 * nc, nullptr, d_z);
  FluidOpMatvec(&st.op, d_z, d_az);
  BufCopy(d_hist, s[0].d_u, 3 * nc, NULL);
  KScale(d_hist, 1.0 / dt, 3 * nc, NULL);
  KFFill(d_yk, 0.0, 3 * nc, NULL);
  FluidOpRhs(&st.op, d_hist, d_yk, d_b);
  BackendDeviceSync();

  RefWriter         wr(dir);
  const std::string sp = std::to_string(g.nlm), sn = std::to_string(n);
  wr.scalar("lmax", g.lmax);
  wr.scalar("lmax_f", g.lmax_f);
  wr.scalar("nlat_f", g.nlat_f);
  wr.scalar("nphi_f", g.nphi_f);
  wr.scalar("dt", dt);
  wr.scalar("re", re);
  wr.scalar("be", be);
  wr.scalar("bending", bending);
  wr.scalar("nsteps", nsteps);
  wr.scalar("r0", r0);
  wr.scalar("area", st.op.pc.area);
  wr.array("cart", cart.data(), 3 * g.nlm, true, "3," + sp);
  wr.array("qst", Down(d_qst, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  wr.array("cart_back", Down(d_back, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  wr.array("z", Down(d_z, n).data(), n, false, sn);
  wr.array("Az", Down(d_az, n).data(), n, false, sn);
  wr.array("b", Down(d_b, n).data(), n, false, sn);
  wr.array("X0", Down(s[0].d_x, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  if (bending) {
    const std::vector<double> y = RandomSpectra(rng, lmax, 3);
    breal *d_y = BufAlloc(3 * nc), *d_jy = BufAlloc(3 * nc);
    BufUpload(d_y, y.data(), 3 * nc, NULL);
    FluidBendingJvp(&g, &st.op.geo, &st.op.w, &st.op.lin, d_y, d_jy);
    BackendDeviceSync();
    wr.array("bforce", Down(st.op.d_bforce, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
    wr.array("y", y.data(), 3 * g.nlm, true, "3," + sp);
    wr.array("jy", Down(d_jy, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
    BufFree(d_y);
    BufFree(d_jy);
  }
  wr.array("u0", Down(s[0].d_u, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);

  int cur = 0, nxt = 1, its1 = 0, its_total = 0;
  for (int k = 1; k <= nsteps; k++) {
    FluidStepInfo info;
    if (FluidStep(&st, &s[cur], nullptr, dt, 1, &s[nxt], &info)) return 1;
    its_total += info.iterations;
    std::swap(cur, nxt);
    if (k == 1) {
      its1 = info.iterations;
      wr.array("X1", Down(s[cur].d_x, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
      wr.array("u1", Down(s[cur].d_u, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
      wr.array("p1", Down(s[cur].d_p, nc).data(), g.nlm, true, sp);
    }
  }
  wr.scalar("iterations1", its1);
  wr.scalar("iterations_mean", (double)its_total / nsteps);
  wr.array("XN", Down(s[cur].d_x, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  wr.array("uN", Down(s[cur].d_u, 3 * nc).data(), 3 * g.nlm, true, "3," + sp);
  wr.array("pN", Down(s[cur].d_p, nc).data(), g.nlm, true, sp);
  if (!wr.finish()) return 1;
  std::printf("wrote %s: %s, lmax %d, product band %d on %d x %d, n %ld, %d steps, %d its at step "
              "1, %.1f mean\n",
              dir.c_str(), example.c_str(), g.lmax, g.lmax_f, g.nlat_f, g.nphi_f, n, nsteps, its1,
              (double)its_total / nsteps);

  for (breal *d : {d_cart, d_qst, d_back, d_spec, d_z, d_az, d_b, d_hist, d_yk}) BufFree(d);
  FluidStepperDestroy(&st);
  for (auto &x : s) FluidStateDestroy(&x);
  FluidGridDestroy(&g);
  BackendFinalize();
  return 0;
}
