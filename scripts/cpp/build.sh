#!/bin/sh
# Builds the dump tools (geometry_dump, static_dump, step_dump) against a configured and built cpu tree of
# meliao/evolving_surface (PR #10), e.g.
#   cmake -S cpp -B cpp/build-cpu -DES_CPU_PETSC=OFF -DES_BUILD_TESTS=OFF
#   cmake --build cpp/build-cpu -j
# then: scripts/cpp/build.sh <evolving_surface>/cpp <evolving_surface>/cpp/build-cpu
# It reuses the flags CMake chose for the fluid_killing driver.
set -e
SRC=$1
BUILD=$2
HERE=$(cd "$(dirname "$0")" && pwd)
FLAGS=$(sed -n 's/^CXX_\(DEFINES\|INCLUDES\|FLAGS\) = //p' "$BUILD/CMakeFiles/fluid_killing.dir/flags.make" | tr '\n' ' ')
LINK=$(sed 's/^[^ ]* .*-o fluid_killing //' "$BUILD/CMakeFiles/fluid_killing.dir/link.txt")
cd "$BUILD"
for t in geometry_dump static_dump step_dump; do
  c++ $FLAGS -c "$HERE/$t.cpp" -o $t.o
  c++ -O3 $t.o -o $t $LINK
  echo "built $BUILD/$t"
done
