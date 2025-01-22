#! /bin/sh

# OS - dependent settings
case $(uname) in
  Darwin*)
    # Darwin (Mac OS X)
    # /usr/local/Cellar/spglib/2.4.0/ from "/usr/local/Cellar/spglib"
    SPGLIB_ROOT=$(echo $(brew --cellar spglib)/*/lib* | sed 's/lib$//')
    # /usr/local/opt/llvm/bin from "echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc"
    LLVM_PATH=$(brew info llvm | grep 'export PATH' | sed 's/^.*export PATH=//; s/:$PATH.*$//; s/\"//')
    [[ ":$PATH:" != *":$LLVM_PATH:"* ]] && export PATH="$LLVM_PATH${PATH:+":$PATH"}"
    CMAKE_C_COMPILER=$(which clang)
    CMAKE_CXX_COMPILER=$(which clang++)
    ;;
  Linux*)
    CMAKE_C_COMPILER=$(which gcc)
    CMAKE_CXX_COMPILER=$(which g++)
    ;;
esac
set -- "$@" -DCMAKE_C_COMPILER=$CMAKE_C_COMPILER -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER

# Add (or initialize) git, rm previous directory, cmake configure, and build binaries
[ -d alamode ] && git submodule update --init alamode || git submodule add -b master https://github.com/ttadano/alamode.git
[ -d build ] && rm -rf build
set -- -S alamode -B build "$@" -DFFTW3_ROOT=/usr/local
set -x;
cmake "$@"
cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
set +x;
