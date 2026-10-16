/* Dumps the C++ fluid solver's surface geometry for one example, for
   test/geometryChecks.ts: the example's position coefficients on the working
   band, the product grid's cos(theta), and FluidGeometryBuild's 16 output
   fields on the product grid (theta-contiguous, as the C++ stores them), in
   the solver's own ref_dump format.

   Builds against a cpu tree of meliao/evolving_surface (PR #10); see build.sh.

     geometry_dump <out_dir> <example> <lmax> <factor> <nlat_f> <nphi_f> <r0> */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backend_init.h"
#include "backend_mem.h"
#include "fluid_examples.h"
#include "fluid_geometry.h"
#include "fluid_grid.h"
#include "fluid_stepper.h"
#include "ref_dump.h"

int main(int argc, char **argv)
{
  if (argc != 8) {
    std::fprintf(stderr, "usage: %s out_dir example lmax factor nlat_f nphi_f r0\n", argv[0]);
    return 1;
  }
  const std::string dir = argv[1], example = argv[2];
  const int    lmax = std::atoi(argv[3]), factor = std::atoi(argv[4]);
  const int    nlat_f = std::atoi(argv[5]), nphi_f = std::atoi(argv[6]);
  const double r0 = std::atof(argv[7]);
  int          bargc = 1;
  if (BackendInit(&bargc, &argv)) return 1;

  FluidGrid g;
  if (FluidGridCreate(&g, lmax, factor, nlat_f, nphi_f, 0)) return 1;
  FluidState s;
  FluidStateCreate(&s, &g);
  if (FluidExampleInit(&g, example, r0, 0, &s)) return 1;
  FluidGeometry geo;
  FluidGeometryCreate(&geo, &g);
  FluidGeometryBuild(&geo, &g, s.d_x);
  BackendDeviceSync();

  const long          ns = g.nspat_f;
  std::vector<double> x(3 * 2 * g.nlm), out(FGEO_OUT_N * ns), ct(g.nlat_f);
  BufDownload(x.data(), s.d_x, (long)x.size(), NULL);
  BufDownload(out.data(), geo.d_out, (long)out.size(), NULL);
  for (int i = 0; i < g.nlat_f; i++) ct[i] = g.op.cfg->ct[i];

  RefWriter wr(dir);
  wr.scalar("lmax", g.lmax);
  wr.scalar("lmax_f", g.lmax_f);
  wr.scalar("nlat_f", g.nlat_f);
  wr.scalar("nphi_f", g.nphi_f);
  wr.scalar("r0", r0);
  wr.array("X", x.data(), 3 * g.nlm, true, "3," + std::to_string(g.nlm));
  wr.array("ct", ct.data(), g.nlat_f, false, std::to_string(g.nlat_f));
  wr.array("geometry", out.data(), (long)out.size(), false,
           std::to_string(FGEO_OUT_N) + "," + std::to_string(g.nphi_f) + "," +
               std::to_string(g.nlat_f));
  if (!wr.finish()) return 1;
  std::printf("wrote %s: lmax %d, product band %d, %d x %d\n", dir.c_str(), g.lmax, g.lmax_f,
              g.nlat_f, g.nphi_f);

  FluidGeometryDestroy(&geo);
  FluidStateDestroy(&s);
  FluidGridDestroy(&g);
  BackendFinalize();
  return 0;
}
