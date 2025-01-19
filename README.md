## Nonequilibrium Molecular Dynamics (nemd)

### Installation
Clone the repository with submodules downloaded and the master branch checked out by
```
git clone --recurse-submodules -j8 git://github.com/zhteg4pvt/nemd --branch master
```
Change the directory, install dependencies, and compile the source codes
```
cd nemd; pip install .[dev] -v
```
### Testing
Set environmental variables (recommended)
```
source premake
```
Run unit testing to check the smallest functional units
```
run_test
```
In a clean directory, run integration testing to check the combined entity
```
run_itest
```