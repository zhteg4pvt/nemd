# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Compile binaries, install packages, and distribute scripts. (Python 3.10.12)

pip3 install . -v
python3 setup.py install
"""
import functools
import glob
import os
import pathlib
import subprocess
import sys

import setuptools

PKGS = ('cmake', )


class Darwin:
    """
    macOS installer.
    """
    MODULE = 'module'
    NEMD = 'nemd'
    LAMMPS = 'lammps'
    ALAMODE = 'alamode'
    PACKAGES = [NEMD, LAMMPS, ALAMODE]
    BUILD = 'build'

    LMP = 'lmp'
    ALM = 'alm'
    ANPHON = 'anphon'
    ALM_EXES = [ALM, ANPHON]

    INSTALL = ('brew', 'install', '-q')
    PKGS = PKGS + ('gcc', 'libomp', 'fftw')
    LMP_PKGS = ('clang-format', )
    ALM_PKGS = ('llvm', 'spglib', 'eigen', 'boost', 'lapack', 'symspg')

    def __init__(self):
        self.dir = pathlib.Path(__file__).parent
        # Python modules to install under site-packages
        self.pkg_dir = {x: os.path.join(self.MODULE, x) for x in self.PACKAGES}

    def install(self):
        """
        Main method to install.
        """
        self.prereq()
        self.compile()

    def prereq(self, *pkgs):
        """
        Install the packages required by compilation.

        :param pkgs tuple: the packages to be installed.
        """
        if not self.lmp_exe:
            pkgs += self.LMP_PKGS
        if len(self.alm_exes) != 2:
            pkgs += self.ALM_PKGS
        if not pkgs:
            return
        self.subprocess(*self.PKGS, *pkgs)

    def subprocess(self, *pkgs):
        """
        Run subprocess to install packages.

        :param pkgs tuple: package names to install.
        """
        cmd = ' '.join(self.INSTALL + pkgs)
        subprocess.run(f"echo {cmd}; {cmd}", shell=True)

    @property
    @functools.lru_cache
    def lmp_exe(self):
        """
        Whether lammps executable with python package is found.

        :return 'PosixPath': lammps executable with python package.
        """
        lmp = self.locate(self.LMP, is_abs=True)
        if not lmp.is_file():
            return
        cmd = f'{lmp} -h | grep PYTHON'
        lmp = subprocess.run(cmd, capture_output=True, shell=True)
        if not lmp.returncode and lmp.stdout:
            return lmp
        print(f'{lmp} with python not found for {self.LMP}.')

    def locate(self, name, is_abs=False):
        """
        Locate the executable in the build.

        :param nam str: the executable name.
        :param is_abs bool: whether the path is an absolute path.
        :return 'PosixPath': the executable path absolute or relative to the
            module dir.
        """
        match name:
            case self.LMP:
                pkg_dir = self.pkg_dir[self.LAMMPS]
                target = (self.BUILD, name)
            case self.ALM | self.ANPHON:
                pkg_dir = self.pkg_dir[self.ALAMODE]
                target = (self.BUILD, name, name)
            case _:
                pkg_dir = self.pkg_dir[self.NEMD]
                target = (self.BUILD, name)
        print(f"{name}: {pkg_dir}: {target}")
        return self.dir.joinpath(pkg_dir, *target) if is_abs else pathlib.Path(
            *target)

    @property
    @functools.lru_cache
    def alm_exes(self):
        """
        Check whether alamode executables can be found.

        :return list of 'PosixPath': alamode executables.
        """
        alm_exes = [self.locate(x, is_abs=True) for x in self.ALM_EXES]
        return [x for x in alm_exes if x.is_file()]

    def compile(self):
        """
        Compile the binaries.
        """
        if not self.lmp_exe:
            print(f'Installing {self.LAMMPS}...')
            cwd = self.dir.joinpath(self.pkg_dir[self.LAMMPS])
            subprocess.run('bash install.sh', shell=True, cwd=cwd)
        if len(self.alm_exes) != 2:
            print(f'Installing {self.ALAMODE}...')
            cwd = self.dir.joinpath(self.pkg_dir[self.ALAMODE])
            subprocess.run('bash install.sh', shell=True, cwd=cwd)


class Linux(Darwin):
    """
    Linux installer.
    """
    INSTALL = ('sudo', 'apt-get', 'install', '-y')
    PKGS = PKGS + (
        'build-essential',
        'fftw3-dev',
    )
    LMP_PKGS = ('libopenmpi-dev', 'ffmpeg')
    ALM_PKGS = ('libsymspg-dev', 'libeigen3-dev', 'libboost-all-dev',
                'libblas-dev', 'liblapack-dev')

    def install(self):
        """
        Main method to install.
        """
        super().install()
        self.term()

    def prereq(self, *pkgs):
        """
        Install the packages required by compilation.

        :param pkgs tuple: the packages to be installed.
        """
        info = subprocess.run('nvidia-smi | grep NVIDIA-SMI', shell=True)
        if not info.returncode:
            pkgs += ('nvidia-cuda-toolkit', )
        super().prereq(*pkgs)


Installer = {'darwin': Darwin, 'linux': Linux}[sys.platform]


class Distribution(Installer):
    """
    Distribution the scripts, packages and data.
    """

    PARQUET = 'parquet'
    # Scripts to install under site-packages/bin
    SCRIPTS = ['sh', 'driver', 'workflow']
    SCRIPTS = [y for x in SCRIPTS for y in glob.glob(os.path.join(x, '*'))]
    SCRIPTS = [x for x in SCRIPTS if os.path.isfile(x)]
    INSTALL_REQUIRES = [
        'numpy<2.0', 'scipy>=1.14.1', 'networkx>=3.3', 'pandas>=2.2.2',
        'chemparse', 'mendeleev', 'rdkit', 'signac', 'signac-flow',
        'matplotlib', 'plotly', 'crystals', 'numba', 'wurlitzer',
        'methodtools', 'fastparquet', 'lazy_import', 'tabulate', 'psutil',
        'yapf', 'isort', 'snakeviz', 'tuna', 'pytest'
    ]
    CLASSIFIERS = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD-3-Clause',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.10'
    ]

    def run(self):
        """
        Main method to install and distribute.
        """
        self.install()
        self.setup()

    def setup(self):
        """
        Setup the distributions.
        """
        setuptools.setup(name=self.NEMD,
                         version='1.0.0',
                         description='A molecular simulation toolkit',
                         url='https://github.com/zhteg4pvt/nemd',
                         author='Teng Zhang',
                         author_email='zhteg4@gmail.com',
                         license='BSD 3-clause',
                         packages=self.PACKAGES,
                         package_dir=self.pkg_dir,
                         package_data=self.package_data,
                         scripts=self.SCRIPTS,
                         install_requires=self.INSTALL_REQUIRES,
                         classifiers=self.CLASSIFIERS)

    @property
    def package_data(self):
        """
        Non-python files to install under site-packages/{package name}

        return dict: key is package name, value is list of namepaths.
        """
        # Non-python files to install under site-packages/{package name}
        # NEMD
        data_dir = pathlib.Path('data')
        nemd = [data_dir.joinpath('table', f'*.{self.PARQUET}')]
        oplsua_dir = data_dir.joinpath('ff', 'oplsua')
        nemd += [oplsua_dir.joinpath(f'*.{x}') for x in ['npy', self.PARQUET]]
        # LAMMPS
        lammps = [str(self.locate(self.LMP))]
        # ALAMODE
        alamode = [str(self.locate(x)) for x in self.ALM_EXES]
        tools = pathlib.Path(self.ALAMODE, 'tools')
        alamode += [tools.joinpath(x, '*.py') for x in ['', 'interface']]
        package_data = {
            self.NEMD: nemd,
            self.LAMMPS: lammps,
            self.ALAMODE: alamode
        }
        return {x: list(map(str, y)) for x, y in package_data.items()}


dist = Distribution()
dist.run()
