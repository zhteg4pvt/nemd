#! /bin/sh
# Under the BSD 3-Clause License by Teng Zhang (zhteg4@gmail.com)
: '
Compile the alamode binary.

Usage:
  install.sh
'
[ -d build ] && rm -rf build

[ -d alamode ] || git submodule add -b master https://github.com/ttadano/alamode.git
set -- git submodule update
[ ! -z $SHALLOW ] && set -- "$@" --depth 1
"$@" --init alamode

set -- cmake
C_COMPILER=gcc
CXX_COMPILER=g++
# OS - dependent settings
case $(uname) in
  Darwin*)
    PREFIX=$(brew config | grep HOMEBREW_PREFIX | awk '{print $2}')
    set -- "$@" -DSPGLIB_ROOT=$PREFIX -DFFTW3_ROOT=$PREFIX
    # llvm compilers
    LLVM=$(brew info llvm | grep 'export PATH' | sed 's/^.*export PATH=//; s/:$PATH.*$//; s/\"//')
    echo $PATH | grep -q $$LLVM || export PATH="$LLVM${PATH:+":$PATH"}"
    C_COMPILER=clang; CXX_COMPILER=clang++
    ;;
esac

set -- "$@" -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER  -S alamode -B build
echo "$@"; "$@"
cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
