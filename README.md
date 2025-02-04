## Nonequilibrium Molecular Dynamics (nemd)

### Installation
Prepare the machine and clone the repository
```
bash -c "$(curl -fsSL 'https://raw.githubusercontent.com/zhteg4/nemd/main/setup')"
```
Install dependencies, compile the binaries, and distribute scripts
```
cd ~/git/nemd; pip3.10 install . -v
```
### Test
Set environmental variables
```
source premake
```
Run unit test
```
nemd_test
```
Run integration test
```
nemd_itest
