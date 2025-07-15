#! /bin/sh
# Under the BSD 3-Clause License by Teng Zhang (zhteg4@gmail.com)
: '
Compile the lammps binary.

Usage:
  install.sh
'
[ -z $SHALLOW ] && rm -rf build

[ -d lammps ] || git submodule add -b release https://github.com/lammps/lammps.git
. ../../sh/nemd_func
git_update lammps

set -- "$@" -D PKG_OPENMP=yes -D PKG_MANYBODY=on -D PKG_MOLECULE=on -D PKG_KSPACE=on \
-D PKG_RIGID=on -D PKG_PYTHON=yes -D PKG_EXTRA-DUMP=yes \
-D Python_EXECUTABLE=$(python -c "import sys; print(sys.executable)")

# OS - dependent settings
case $(uname) in
  Darwin*)
    llvm; C_COMPILER=clang; CXX_COMPILER=clang++
    ;;
  Linux*)
    C_COMPILER=gcc; CXX_COMPILER=g++
    # https://freezing.cool/notes/lammps-on-wsl-with-openmp-and-gpu
    (nvidia-smi 2>/dev/null) && set -- "$@" -D PKG_GPU=on -D GPU_API=cuda
    ;;
esac

cmake_build "$@" -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER -S ./lammps/cmake/
