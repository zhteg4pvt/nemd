#! /bin/sh

# Python3 executable (which python3 -> Mac: /usr/local/bin; Linux: /usr/bin/)
set -- "$@" -D PYTHON_EXECUTABLE=$(which python3) -D CMAKE_INSTALL_PREFIX=/usr/local
# OpenMP packages (https://docs.lammps.org/Speed_omp.html)
set -- "$@" -D PKG_OPENMP=yes -D PKG_PYTHON=on -D PKG_MANYBODY=on \
  -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RIGID=on -D BUILD_LIB=on \
  -D BUILD_SHARED_LIBS=yes
# OS - dependent settings
case $(uname) in
  Darwin*)
    # Darwin (Mac OS X)
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

# Add (or initialize) git, rm previous directory, cmake configure, and build binaries
[ -d lammps ] && git submodule update --init lammps || git submodule add -b release https://github.com/lammps/lammps.git
[ -d build ] && rm -rf build
set -- "$@" -S ./lammps/cmake/ -B build
set -x;
cmake "$@"
cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
set +x;
# Create soft links
# Though '-D LAMMPS_MACHINE=serial' creates lmp_serial executable,
# it generates liblammps_serial.dylib instead of liblammps.dylib which is required by
# import lammps; lammps.lammps(). (/usr/local/lib/python3.10/site-packages/lammps)
ln build/lmp build/lmp_serial
