#! /bin/sh
# Under the BSD 3-Clause License by Teng Zhang (zhteg4@gmail.com)
: '
Compile the lammps binary.

Usage:
  install.sh
'
[ -d build ] && rm -rf build

[ -d lammps ] || git submodule add -b release https://github.com/lammps/lammps.git
set -- git submodule update
[ ! -z $SHALLOW ] && set -- "$@" --depth 1
"$@" --init lammps

set -- cmake
# OpenMP packages (https://docs.lammps.org/Speed_omp.html)
set -- "$@" -D PKG_OPENMP=yes -D PKG_PYTHON=on -D PKG_MANYBODY=on \
  -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RIGID=on -D BUILD_LIB=on
# OS - dependent settings
case $(uname) in
  Darwin*)
    # Mac OS X enables OpenMP libraries (https://iscinumpy.gitlab.io/post/omp-on-high-sierra/)
    set -- "$@" -DOpenMP_CXX_LIB_NAMES=omp \
      -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.a \
      -DOpenMP_CXX_FLAGS="'-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include'"
    ;;
  Linux*)
    # https://freezing.cool/notes/lammps-on-wsl-with-openmp-and-gpu
    (nvidia-smi 2>/dev/null) && set -- "$@" -D PKG_GPU=on -D GPU_API=cuda
    ;;
esac

set -- "$@" -S ./lammps/cmake/ -B build
echo "$@"; "$@"
cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
