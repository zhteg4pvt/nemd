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

MODULE = 'module'
NEMD = 'nemd'
LAMMPS = 'lammps'
ALAMODE = 'alamode'
PKGS = ('cmake', )


class Darwin:
    """
    macOS installer.
    """

    LMP = 'lmp'
    ALM = 'alm'
    ANPHON = 'anphon'
    INSTALL = ('brew', 'install', '-q')
    PKGS = PKGS + ('gcc', 'libomp')
    LMP_PKGS = ('clang-format', )
    ALM_PKGS = ('llvm', 'open-mpi', 'spglib', 'fftw', 'eigen', 'boost',
                'lapack')

    def install(self):
        """
        Main method to install.
        """
        self.prereq()
        self.compile()
        self.qt()

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
        print(cmd)
        subprocess.run(cmd, shell=True)

    @property
    @functools.lru_cache
    def lmp_exe(self):
        """
        Whether lammps executable with python package is found.

        :return 'PosixPath': lammps executable with python package.
        """
        bin_exe = self.locate(self.LMP)
        if not bin_exe:
            return
        cmd = f'{bin_exe} -h | grep PYTHON'
        lmp = subprocess.run(cmd, capture_output=True, shell=True)
        if not lmp.returncode and lmp.stdout:
            return bin_exe
        print(f'{bin_exe} with python not found for {self.LMP}.')

    def locate(self, name):
        """
        Locate the executable in bin.

        :param name: the executable name.
        :return 'PosixPath': the executable pathname in bin.
        """
        # name in build
        match name:
            case self.LMP:
                target = f'{MODULE}/{LAMMPS}/build/{name}'
            case self.ALM | self.ANPHON:
                target = f'{MODULE}/{ALAMODE}/build/{name}/{name}'
        target = pathlib.Path(target)
        if target.is_file():
            return target
        print(f'{target} not found.')

    @property
    @functools.lru_cache
    def alm_exes(self):
        """
        Check whether alamode executables can be found.

        :return list of 'PosixPath': alamode executables.
        """
        return [x for x in map(self.locate, [self.ALM, self.ANPHON]) if x]

    def compile(self):
        """
        Compile the binaries.
        """
        if not self.lmp_exe:
            print(f'Installing {LAMMPS}...')
            cwd = os.path.join(MODULE, LAMMPS)
            subprocess.run('bash install.sh', shell=True, cwd=cwd)
        if len(self.alm_exes) != 2:
            print(f'Installing {ALAMODE}...')
            cwd = os.path.join(MODULE, ALAMODE)
            subprocess.run('bash install.sh', shell=True, cwd=cwd)

    def qt(self):
        """
        Install the qt, a C++framework for developing graphical user interfaces
        and cross-platform applications, both desktop and embedded.
        """
        qt = subprocess.run('brew list qt5', capture_output=True, shell=True)
        if qt.stdout:
            print('qt installation found.')
            return
        print('qt installation not found. Installing...')
        self.subprocess('qt5')


class Linux(Darwin):
    """
    Linux installer.
    """
    INSTALL = ('sudo', 'apt-get', 'install', '-y')
    PKGS = PKGS + ('zsh', 'build-essential')
    LMP_PKGS = ('python3-venv', 'python3-apt', 'python3-setuptools',
                'openmpi-bin', 'openmpi-common', 'libopenmpi-dev',
                'libgtk2.0-dev', 'fftw3', 'fftw3-dev', 'ffmpeg')
    ALM_PKGS = ('libsymspg-dev', 'libeigen3-dev', 'fftw3-dev',
                'libboost-all-dev', 'libblas-dev', 'liblapack-dev')

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
        super().subprocess(*pkgs)

    def qt(self):
        """
        Install the qt, a C++framework for developing graphical user interfaces
        and cross-platform applications, both desktop and embedded.
        """
        self.subprocess('qt5-base-dev', 'libgl1-mesa-dev', '^libxcb.*-dev',
                        'libx11-xcb-dev', 'libglu1-mesa-dev')

    def term(self):
        """
        Install terminal supporting split view.
        """
        self.subprocess('tilix')


Installer = {'darwin': Darwin, 'linux': Linux}[sys.platform]


class Distribution(Installer):
    """
    Distribution the scripts, packages and data.
    """

    PARQUET = 'parquet'
    PACKAGES = [NEMD, LAMMPS, ALAMODE]
    # Python modules to install under site-packages
    PACKAGE_DIR = {x: os.path.join(MODULE, x) for x in PACKAGES}
    # Scripts to install under site-packages/bin
    SCRIPTS = ['sh', 'driver', 'workflow']
    SCRIPTS = [y for x in SCRIPTS for y in glob.glob(os.path.join(x, '*'))]
    SCRIPTS = [x for x in SCRIPTS if os.path.isfile(x)]
    # 'pyqt5==5.15.4', 'pyqt5-sip==12.12.1' removes the DeprecationWarning:
    # sipPyTypeDict() is deprecated, xxx sipPyTypeDictRef() instead
    INSTALL_REQUIRES = [
        'numpy', 'scipy>=1.14.1', 'networkx>=3.3', 'pandas>=2.2.2',
        'chemparse', 'mendeleev', 'rdkit', 'signac', 'signac-flow',
        'matplotlib', 'plotly', 'crystals', 'numba', 'wurlitzer',
        'methodtools', 'fastparquet', 'lazy_import', 'tabulate', 'psutil',
        'yapf', 'isort', 'snakeviz', 'tuna', 'pytest', 'dash[testing]',
        'pyqt5==5.15.4', 'pyqt5-sip==12.12.1', 'dash_bootstrap_components'
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
        setuptools.setup(name=NEMD,
                         version='1.0.0',
                         description='A molecular simulation toolkit',
                         url='https://github.com/zhteg4pvt/nemd',
                         author='Teng Zhang',
                         author_email='zhteg4@gmail.com',
                         license='BSD 3-clause',
                         packages=self.PACKAGES,
                         package_dir=self.PACKAGE_DIR,
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
        lammps = [str(self.lmp_exe.relative_to(self.PACKAGE_DIR[LAMMPS]))]
        # ALAMODE
        alamode_dir = self.PACKAGE_DIR[ALAMODE]
        alamode = [str(x.relative_to(alamode_dir)) for x in self.alm_exes]
        tools = pathlib.Path(ALAMODE, 'tools')
        alamode += [tools.joinpath(x, '*.py') for x in ['', 'interface']]
        package_data = {NEMD: nemd, LAMMPS: lammps, ALAMODE: alamode}
        return {x: list(map(str, y)) for x, y in package_data.items()}


dist = Distribution()
dist.run()
