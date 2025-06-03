#! /bin/sh
# Under the BSD 3-Clause License by Teng Zhang (zhteg4@gmail.com)
: '
Compile the alamode binary.

Usage:
  install.sh
'
[ -z $SHALLOW ] && rm -rf build

[ -d alamode ] || git submodule add -b master https://github.com/ttadano/alamode.git
. ../../sh/nemd_func
git_update alamode

# OS - dependent settings
case $(uname) in
  Darwin*)
    PREFIX=$(brew --prefix); set -- "$@" -DSPGLIB_ROOT=$PREFIX -DFFTW3_ROOT=$PREFIX
    llvm; C_COMPILER=clang; CXX_COMPILER=clang++
    ;;
  Linux*)
    C_COMPILER=gcc; CXX_COMPILER=g++
    ;;
esac

cmake_build "$@" -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER -S alamode -DCMAKE_POLICY_VERSION_MINIMUM=3.5
