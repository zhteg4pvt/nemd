#! /bin/zsh
# git submodule add -b develop (https://github.com/ttadano/alamode.git)
# Initialize any uninitialized submodules
git submodule update --init alamode
# Remove previous build
[ -d build ] && rm -rf build
# Set source and target
set -- -S alamode -B build "$@" -DFFTW3_ROOT=/usr/local
# OS - dependent settings ($OSTYPE sh is empty)
case "$OSTYPE" in
  darwin*)
    # Darwin (Mac OS X)
    # /usr/local/Cellar/spglib/2.4.0/ from "/usr/local/Cellar/spglib"
    SPGLIB_ROOT=$(echo $(brew --cellar spglib)/*/lib* | sed 's/lib$//')
    # /usr/local/opt/llvm/bin from "echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc"
    LLVM_PATH=$(brew info llvm | grep 'export PATH' | sed 's/^.*export PATH=//; s/:$PATH.*$//; s/\"//')
    [[ ":$PATH:" != *":$LLVM_PATH:"* ]] && export PATH="$LLVM_PATH${PATH:+":$PATH"}"
    CMAKE_C_COMPILER=$(which clang)
    CMAKE_CXX_COMPILER=$(which clang++)
    ;;
  linux*)
    SPGLIB_ROOT=`pip3 show spglib | grep 'Location:' | sed 's/^.*: //'`/spglib
    CMAKE_C_COMPILER=$(which gcc)
    CMAKE_CXX_COMPILER=$(which g++)
    ;;
esac
set -- "$@" -DCMAKE_C_COMPILER=$CMAKE_C_COMPILER -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER -DSPGLIB_ROOT=$SPGLIB_ROOT
# cmake configure and build
set -x;
cmake "$@"
cmake --build build -j $(nproc 2>/dev/null || sysctl -n hw.logicalcpu)
set +x;
