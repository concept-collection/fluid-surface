#!/bin/sh
# Builds geometry_dump against a configured and built cpu tree of
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
c++ $FLAGS -c "$HERE/geometry_dump.cpp" -o geometry_dump.o
c++ -O3 geometry_dump.o -o geometry_dump $LINK
echo "built $BUILD/geometry_dump"
