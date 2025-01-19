"""
pip3 install setuptools openvino-telemetry
pip3 install .[dev] -v --user
python3 setup.py install

Python 3.10.12 is supported and tested.
request 'pyqt5==5.15.4', 'pyqt5-sip==12.12.1' to remove the DeprecationWarning:
sipPyTypeDict() is deprecated, xxx sipPyTypeDictRef() instead
"""
import functools
import glob
import os
import pathlib
import subprocess
import sys

import setuptools
from setuptools.command.install import install

MODULE = 'module'
NEMD = 'nemd'
LAMMPS = 'lammps'
ALAMODE = 'alamode'
# Python modules to install under site-packages
PACKAGE_DIR = {x: os.path.join(MODULE, x) for x in [NEMD, ALAMODE]}
# Non-python files to install under site-packages/{package name}
DATA = pathlib.Path('data')
NPY = 'npy'
PARQUET = 'parquet'
NEMD_DATA = [DATA.joinpath('ff', 'oplsua', f'*.{x}') for x in [NPY, PARQUET]]
NEMD_DATA.append(DATA.joinpath('table', f'*.{PARQUET}'))
ALAMODE_TOOLS = pathlib.Path(ALAMODE, 'tools')
ALAMODE_DATA = [ALAMODE_TOOLS.joinpath(x, '*.py') for x in ['', 'interface']]
PACKAGE_DATA = {NEMD: NEMD_DATA, ALAMODE: ALAMODE_DATA}
PACKAGE_DATA = {x: list(map(str, y)) for x, y in PACKAGE_DATA.items()}
# Scripts to install under site-packages/bin
SCRIPTS = ['sh', 'driver', 'workflow']
SCRIPTS = [y for x in SCRIPTS for y in glob.glob(os.path.join(x, '*'))]
SCRIPTS = [x for x in SCRIPTS if os.path.isfile(x)]
PKGS = ('cmake', )


class Darwin:
    """
    macOS installer.
    """

    LMP = 'lmp_serial'
    ALM = 'alm'
    ANPHON = 'anphon'
    BIN = pathlib.Path('/usr/local/bin')
    INSTALL = ('brew', 'install', '-q')
    PKGS = PKGS + ('gcc', 'libomp')
    LMP_PKGS = ('clang-format', )
    ALM_PKGS = ('llvm', 'open-mpi', 'spglib', 'fftw', 'eigen', 'boost',
                'lapack')

    def run(self):
        """
        Main method to run.
        """
        self.install()
        self.compile()
        self.installQt()

    def install(self, *pkgs):
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
        subprocess.run(' '.join(self.INSTALL + pkgs), shell=True)

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
        # bin/name
        bin_exe = self.BIN.joinpath(name)
        if os.access(bin_exe, os.X_OK):
            print(f'{bin_exe} found for {name}.')
            return bin_exe
        # name in build
        match name:
            case self.LMP:
                target = f'{MODULE}/{LAMMPS}/build/{name}'
            case self.ALM | self.ANPHON:
                target = f'{MODULE}/{ALAMODE}/build/{name}/{name}'
        target = pathlib.Path(target)
        if not target.is_file():
            print(f'{target} not found.')
            return
        # bin/name --> name in build
        try:
            bin_exe.unlink()
        except FileNotFoundError:
            pass
        bin_exe.symlink_to(target.resolve())
        print(f'Soft link to {target} updated. ({bin_exe})')
        return bin_exe

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
        Install the lammps with specific packages if not available.
        """
        if not self.lmp_exe:
            print(f'Installing {LAMMPS}...')
            cwd = os.path.join(MODULE, LAMMPS)
            subprocess.run('bash install.sh', shell=True, cwd=cwd)
            self.locate(self.LMP)
        if len(self.alm_exes) != 2:
            print(f'Installing {ALAMODE}...')
            cwd = os.path.join(MODULE, ALAMODE)
            subprocess.run('bash install.sh', shell=True, cwd=cwd)
            self.locate(self.ALM)
            self.locate(self.ANPHON)

    def installQt(self):
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
    # Permission denied: 'xxx' -> '/usr/local/bin/lmp_serial'
    BIN = pathlib.Path.home().joinpath(".local/bin")
    INSTALL = ('sudo', 'apt-get', 'install', '-y')
    PKGS = PKGS + ('zsh', 'build-essential')
    LMP_PKGS = ('python3-venv', 'python3-apt', 'python3-setuptools',
                'openmpi-bin', 'openmpi-common', 'libopenmpi-dev',
                'libgtk2.0-dev', 'fftw3', 'fftw3-dev', 'ffmpeg')
    ALM_PKGS = ('libeigen3-dev', 'libboost-all-dev', 'libblas-dev',
                'liblapack-dev')

    def run(self):
        """
        Main method to run.
        """
        super().run()
        self.installTerm()

    def install(self, *pkgs):
        """
        Install the packages required by compilation.

        :param pkgs tuple: the packages to be installed.
        """
        info = subprocess.run('nvidia-smi | grep NVIDIA-SMI', shell=True)
        if not info.returncode:
            pkgs += ('nvidia-cuda-toolkit', )
        super().install(*pkgs)

    def installQt(self):
        """
        Install the qt, a C++framework for developing graphical user interfaces
        and cross-platform applications, both desktop and embedded.
        """
        self.subprocess('qt5-base-dev', 'libgl1-mesa-dev', '^libxcb.*-dev',
                        'libx11-xcb-dev', 'libglu1-mesa-dev')

    def installTerm(self):
        """
        Install terminal supporting split view.
        """
        self.subprocess('tilix')


class Install(install):
    """
    Installation on different platforms.
    """

    def run(self):
        """
        Install packages outside regular install_requires.
        """
        print(f"***** Platform: {sys.platform} *****")
        match sys.platform:
            case 'darwin':
                Darwin().run()
            case 'linux':
                Linux().run()
        super().run()


setuptools.setup(
    name=NEMD,
    version='1.0.0',
    description='A molecular simulation toolkit',
    url='https://github.com/zhteg4pvt/nemd',
    author='Teng Zhang',
    author_email='zhteg4@gmail.com',
    license='BSD 3-clause',
    packages=[NEMD, ALAMODE],
    package_dir=PACKAGE_DIR,
    package_data=PACKAGE_DATA,
    scripts=SCRIPTS,
    install_requires=[
        'numpy', 'scipy>=1.14.1', 'networkx>=3.3', 'pandas>=2.2.2',
        'chemparse', 'mendeleev', 'rdkit', 'signac', 'signac-flow',
        'matplotlib', 'plotly', 'crystals', 'spglib', 'numba', 'tbb',
        'wurlitzer', 'methodtools', 'fastparquet', 'lazy_import', 'tabulate',
        'psutil', 'pyqt5==5.15.4', 'pyqt5-sip==12.12.1',
        'dash_bootstrap_components'
    ],
    extras_require={
        'dev':
        ['yapf', 'isort', 'snakeviz', 'tuna', 'pytest', 'dash[testing]']
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD-3-Clause',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.10'
    ],
    cmdclass={'install': Install})
