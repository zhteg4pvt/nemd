#! /bin/sh

C_COMPILER=gcc
CXX_COMPILER=g++

# OS - dependent settings
case $(uname) in
  Darwin*)
    set -- "$@" -DSPGLIB_ROOT=/usr/local -DFFTW3_ROOT=/usr/local
    # llvm compilers
    LLVM=$(brew info llvm | grep 'export PATH' | sed 's/^.*export PATH=//; s/:$PATH.*$//; s/\"//')
    echo $PATH | grep -q $$LLVM || export PATH="$LLVM${PATH:+":$PATH"}"
    C_COMPILER=clang; CXX_COMPILER=clang++
    ;;
esac

set -- "$@" -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER  -S alamode -B build

# Add (or initialize) git, rm previous directory, cmake configure, and build binaries
[ -d alamode ] && git submodule update --init alamode || git submodule add -b master https://github.com/ttadano/alamode.git
[ -d build ] && rm -rf build
set -x;
cmake "$@"
cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
set +x;