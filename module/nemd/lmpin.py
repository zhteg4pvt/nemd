# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module writes Lammps in script.

https://docs.lammps.org/commands_list.html
"""
import contextlib
import os
import re
import string

import numpy as np
import scipy

from nemd import constants
from nemd import symbols


class Base(list):
    """
    Base class for in-script and log analyzer.
    """
    UNITS = 'units'
    REAL = 'real'
    METAL = 'metal'
    TIMESTEP = 'timestep'

    def __init__(self, unit=METAL, options=None):
        """
        :param unit str: the unit.
        :param options `namedtuple`: command line options.
        """
        super().__init__()
        self.unit = unit
        self.options = options

    def getTimestep(self, timestep=None, backend=False):
        """
        Return the timestep.

        :param timestep float: the timestep in self.unit.
        :param backend bool: the timestep is in ps if True.
        :raise ValueError: when the unit is unknown.
        """
        if timestep:
            timestep *= self.time_unit
        if timestep is None:
            timestep = getattr(self.options, 'timestep', None)
            if timestep:
                timestep *= scipy.constants.femto
        if timestep is None:
            match self.unit:
                case self.REAL | self.METAL:
                    timestep = scipy.constants.femto
        return timestep / (scipy.constants.pico if backend else self.time_unit)

    @property
    def time_unit(self):
        """
        Get the front-end time unit.
        """
        match self.unit:
            case self.REAL:
                return scipy.constants.femto
            case self.METAL:
                return scipy.constants.pico


class SinglePoint(Base):
    """
    LAMMPS in-script writer for configuration and single point simulation.
    """
    ATOMIC = 'atomic'

    PAIR_STYLE = 'pair_style'
    SW = 'sw'

    READ_DATA = 'read_data'
    READ_DATA_RE = re.compile(rf'{READ_DATA}\s*(\S*)')
    PAIR_COEFF = 'pair_coeff'
    CUSTOM_EXT = '.custom'
    XTC_EXT = symbols.XTC_EXT
    DUMP_ID, DUMP_Q = 1, 1000

    V_ATOM_STYLE = ATOMIC
    V_PAIR_STYLE = SW

    def __init__(self, struct=None, isblock=False, **kwargs):
        """
        :param options 'argparse.Namespace': command line options.
        :param isblock bool: whether self is already a block.
        """
        kwargs.setdefault('options', struct.options)
        super().__init__(**kwargs)
        self.struct = struct
        self.isblock = isblock
        self.outfile = f"{self.options.JOBNAME}.in"
        # Segmentation fault: 11 (11) when 1 atom dumps xtc trajectory
        ext = self.CUSTOM_EXT if self.struct.atom_total == 1 else self.XTC_EXT
        self.dump_file = f"{self.options.JOBNAME}{ext}"

    def write(self):
        """
        Write the command to the file.
        """
        self.setUp()
        self.finalize()
        with open(self.outfile, 'w') as fh:
            for line in self:
                fh.write(line + '\n')

    def setUp(self):
        """
        Write out LAMMPS in script.
        """
        self.setup()
        self.pair()
        self.data()
        self.coeff()
        self.traj()
        self.thermo()
        self.minimize()
        self.timestep()
        self.simulation()

    def setup(self):
        """
        Write the setup section including unit and atom styles.
        """
        self.join(self.UNITS, self.unit)
        self.join('atom_style', self.V_ATOM_STYLE)

    def join(self, *args, newline=False):
        """
        Join the arguments to form a lammps command.

        :param newline: add a newline.
        """
        cmd = " ".join(map(str, args))
        self.append(cmd + '\n' if newline else cmd)

    def pair(self):
        """
        Write pair style.
        """
        self.join(self.PAIR_STYLE, self.V_PAIR_STYLE)

    def data(self):
        """
        Write data file related information.
        """
        self.join(self.READ_DATA, f"{self.options.JOBNAME}.data")

    def coeff(self):
        """
        Write pair coefficient.
        """
        self.join(self.PAIR_COEFF, '*', '*', self.struct.ff,
                  *self.struct.masses.element)

    def traj(self):
        """
        Dump out trajectory.
        """
        if self.dump_file.endswith(self.XTC_EXT):
            self.dump(self.DUMP_ID,
                      'all',
                      'xtc',
                      self.DUMP_Q,
                      self.dump_file,
                      xyz=False)
            self.dump_modify(self.DUMP_ID, 'unwrap', 'yes')
            return
        self.dump(self.DUMP_ID, 'all', 'custom', self.DUMP_Q, self.dump_file,
                  'id')
        self.dump_modify(self.DUMP_ID, sort=True)

    def thermo(self):
        """
        Set thermo.
        """
        self.append('thermo 1000')
        self.append('thermo_modify flush yes')

    def dump(self, idx, *args, xyz=True, force=False):
        """
        Dump out trajectory.

        :param idx int: the dump id.
        :param xyz bool: write xyz coordinates if Truef
        :param force bool: write force on each atom if True
        """
        if xyz:
            args += ('xu', 'yu', 'zu')
        if force:
            args += ('fx', 'fy', 'fz')
        if not args:
            return
        self.join('dump', idx, *args)

    def dump_modify(self, idx, *args, sort=False, fmt=None):
        """
        Modify the trajectory dump.

        :param idx int: the dump id.
        :param sort bool: sort by atom id if True
        :param fmt str: the float format
        """
        if sort:
            args += ('sort', 'id')
        if fmt:
            args += ('format', fmt)
        if not args:
            return
        self.join('dump_modify', idx, *args)

    def minimize(self, min_style='fire', geo=None):
        """
        Write commands related to minimization.

        :param min_style str: cg, fire, spin, etc.
        :param geo str: the geometry to restrain (e.g., dihedral 1 2 3 4).
        """
        if self.options.no_minimize:
            return
        restrain = geo and (len(self.options.substruct) > 1)
        if restrain:
            self.append(
                f'fix rest all restrain {geo} -2000.0 -2000.0 {self.options.substruct[1]}'
            )
        self.join('min_style', min_style)
        self.append(f"minimize 1.0e-6 1.0e-8 1000000 10000000")
        if restrain:
            self.append('unfix rest')

    def timestep(self):
        """
        Write timestep-related commands.
        """
        if not self.options.temp:
            return
        self.join(self.TIMESTEP, self.getTimestep())

    def simulation(self):
        """
        Single point energy calculation.
        """
        self.run_step()

    def run_step(self, nstep=0):
        """
        Write run.
        """
        self.join('run', int(nstep))

    def finalize(self, block_id=0):
        """
        Finalize by balancing the fixes with unfixes, flattening the sublist,
        write restart, and terminating with the quit command.

        :param block_id int: the starting fix id.
        """
        for idx, cmds in enumerate(self):
            if isinstance(cmds, str):
                continue
            ascii_id = 0
            for sub_idx, cmd in enumerate(cmds[:]):
                if not cmd.startswith("fix %s"):
                    continue
                fid = f'{block_id}{string.ascii_lowercase[ascii_id]}'
                cmds[sub_idx] = cmd % fid
                cmds.append("unfix %s" % fid)
                ascii_id += 1
            self[idx:idx + 1] = cmds
            block_id += 1
        self.join('write_restart', f"{self.options.JOBNAME}.rst")
        self.join('quit', 0)

    def equal(self, name, expr, bracked=False, quoted=False, **kwargs):
        """
        Define variable with equal style.

        :param name str: the variable name.
        :param expr str: the variable expression.
        :param bracked bool: curly bracket the expression.
        :param quoted bool: quote the expression.
        """
        if bracked:
            expr = f'${{{expr}}}'
        if quoted:
            expr = f'"{expr}"'
        self.variable(name, 'equal', expr, **kwargs)

    def variable(self, *args, **kwargs):
        """
        Define variable.
        """
        self.join('variable', *args, **kwargs)

    @contextlib.contextmanager
    def block(self, newline=True):
        """
        Create, yield, and append a block.

        :param newline: add an empty line after the block.
        :return `cls`: block if self or a new block.
        """
        if self.isblock:
            yield self
            return
        block = self.__class__(struct=self.struct, isblock=True)
        try:
            yield block
        finally:
            self.append(block)
            if newline:
                self.append('')

    def __str__(self, fmt='\033[36m%s\033[0m'):
        """
        Get a human-readable string representation.

        :param fmt str: format of the block.
        :return str: the formatted printing.
        """
        flat = [x if isinstance(x, str) else fmt % '\n'.join(x) for x in self]
        return '\n'.join(flat)


class RampUp(SinglePoint):
    """
    Customized for low temp NVT, NPT ramp up, NPT relaxation and production.
    """
    NVT = 'NVT'
    NPT = 'NPT'
    NVE = 'NVE'
    MODULUS = 'modulus'
    ENSEMBLES = [NVE, NVT, NPT]
    BERENDSEN = 'berendsen'

    def __init__(self, *args, **kwargs):
        """
        :param atom_total int: the total number of atoms.
        """
        super().__init__(*args, **kwargs)
        self.relax_step = self.options.relax_time * constants.NANO_TO_FEMTO / self.options.timestep
        if self.relax_step:
            self.relax_step = round(self.relax_step, -3) or 1E3

    def simulation(self):
        """
        Main method to run the writer.
        """
        if not self.options.temp or self.struct.atom_total == 1:
            self.run_step()
            return
        self.velocity()
        self.startLow()
        self.rampUp()
        self.relaxation()
        self.production()

    def velocity(self):
        """
        Create initial velocity.
        """
        temp = self.options.stemp if self.relax_step else self.options.temp
        seed = np.random.randint(1, high=symbols.MAX_INT32)
        self.join('velocity all create', temp, seed)

    def startLow(self):
        """
        Start simulation from low temperature and constant volume.
        """
        self.nvt(nstep=self.relax_step / 1E3,
                 stemp=self.options.stemp,
                 temp=self.options.stemp)

    def nvt(self, nstep=1E4, stemp=None, temp=300, style=BERENDSEN):
        """
        Append command for constant volume and temperature.

        :param nstep int: run this number of steps.
        :param stemp float: starting temperature.
        :param temp float: target temperature.
        :param style str: thermostat style.
        """
        if not nstep:
            return
        if stemp is None:
            stemp = temp
        with self.block() as blk:
            if style == self.BERENDSEN:
                blk.fix_all('temp/berendsen', stemp, temp, self.options.tdamp)
                blk.nve(nstep=nstep)
            # FIXME: support thermostat more than berendsen.

    def fix_all(self, *args):
        """
        Fix all command.
        """
        self.join('fix %s all', *args)

    def rampUp(self, spress=1):
        """
        Ramp up temperature to the targe value.

        :param spress float: starting pressure.
        """
        self.npt(nstep=self.relax_step / 1E1,
                 stemp=self.options.stemp,
                 spress=spress,
                 temp=self.options.temp,
                 press=self.options.press)

    def npt(self,
            nstep=20000,
            spress=None,
            press=1,
            style=BERENDSEN,
            modulus=10,
            **kwargs):
        """
        Append command for constant pressure and temperature.

        :param nstep int: run this number of steps.
        :param spress float: starting pressure.
        :param press float: target pressure.
        :param style str: the barostat style.
        :param modulus float: the modulus for the barostat.
        """
        if not nstep:
            return
        if spress is None:
            spress = press
        with self.block() as blk:
            if style == self.BERENDSEN:
                blk.fix_all('press/berendsen', 'iso', spress, press,
                            self.options.pdamp, self.MODULUS, modulus)
                blk.nvt(nstep=nstep, **kwargs)
        # FIXME: support thermostat more than berendsen.

    def relaxation(self, modulus=10):
        """
        Relaxation at constant temperature.

        :param modulus float: the modulus for the barostat.
        """
        self.npt(nstep=self.relax_step,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 press=self.options.press,
                 modulus=modulus)

    def production(self, **kwargs):
        """
        Production run.

        NOTE: NVE is good for all, specially transport properties, but requires
        good energy conservation during the time integration. NVT and NPT help
        with properties non-sensitive to disturbance.
        """
        nstep = self.options.prod_time * constants.NANO_TO_FEMTO / self.options.timestep
        match self.options.prod_ens:
            case self.NVE:
                self.nve(nstep=nstep)
            case self.NVT:
                self.nvt(nstep=nstep, temp=self.options.temp)
            case self.NPT:
                self.npt(nstep=nstep,
                         temp=self.options.temp,
                         press=self.options.press,
                         **kwargs)

    def nve(self, nstep=1E3):
        """
        Constant energy and volume.

        NOTE: NVT works with single module while NVT yields nan xyz

        :param nstep int: run this number of steps.
        """
        if not nstep:
            return
        with self.block() as blk:
            blk.fix_all('nve')
            blk.run_step(nstep=nstep)


class Ave(RampUp):
    """
    Customized with PBC averaging for NVT relaxation.
    """
    XYZ = 'xyz'

    def relaxation(self, **kwargs):
        """
        Customized with cell averaging.
        """
        if self.options.prod_ens == self.NPT:
            super().relaxation(**kwargs)
            return
        # NVE and NVT production runs use averaged cell
        self.average(**kwargs)
        self.nvt(nstep=self.relax_step / 1E2,
                 stemp=self.options.temp,
                 temp=self.options.temp)

    def average(self, modulus=10, rec_num=10):
        """
        Change the box to the average boundary of a NPT simulation.

        :param modulus float: the modules in berendsen barostat.
        :param rec_num int: the number of records.
        """
        if not self.relax_step:
            return
        # Run NPT with boundary recorded.
        spans = [f'{i}l' for i in self.XYZ]
        for name, dim in zip(spans, self.XYZ):
            self.equal(name, f'{dim}hi - {dim}lo', quoted=True)
        with self.block() as blk:
            blk.ave_time(*[f'v_{i}' for i in spans],
                         file=self.XYZ,
                         nstep=self.relax_step,
                         num=rec_num)
            blk.npt(nstep=self.relax_step,
                    stemp=self.options.temp,
                    temp=self.options.temp,
                    press=self.options.press,
                    modulus=modulus)
        self.print(*spans, label='Final')
        # Calculate the aves span, ratio to remap, and change the box.
        aves = [f'ave_{i}' for i in self.XYZ]
        for name, dim in zip(aves, self.XYZ):
            self.python(name, f'get{dim.upper()}L', 'sf', self.XYZ)
        self.print(*aves, label='Averaged')
        ratios = [f'ratio_{i}' for i in self.XYZ]
        for ratio, ave, span in zip(ratios, aves, spans):
            self.equal(ratio, f'v_{ave} / v_{span}', quoted=True)
        scales = [f'{i} scale ${{{r}}}' for i, r in zip(self.XYZ, ratios)]
        self.join('change_box', 'all', *scales, 'remap')
        # Delete used variables.
        self.delete(*spans, *aves, *ratios)

    def ave_time(self, *args, file=None, nstep=None, num=None):
        """
        Fix ave/time command.

        :param file str: save the data into this file.
        :param nstep int: simulation step.
        :param num int: average this number of steps.
        """
        if nstep and num:
            per = int(nstep / num)
            args = (1, per, per) + args
        if file:
            args += ('file', file)
        if not args:
            return
        self.fix_all('ave/time', *args)

    def python(self, name, func, fmt, *inputs, imp='from nemd.lmpfunc import'):
        """
        Construct a python command.

        :param name str: the variable name.
        :param func str: the function name.
        :param fmt str: the type of the input and output variables.
        :param inputs tuple: the input variables of the function.
        :param imp str: the python command to import function.
        """
        self.variable(name, 'python', func)
        self.join('python', func, 'input', len(inputs), *inputs, 'return',
                  f'v_{name}', 'format', fmt, 'here', f'"{imp} {func}"')

    def print(self, *args, label=None):
        """
        Print variables.

        :param args tuple of str: the variables to be printed.
        :param label str: the label of the variables.
        """
        to_print = ' '.join(f'{i}=${{{i}}}' for i in args)
        if label:
            to_print = f'{label}: {to_print}'
        self.join('print', f'"{to_print}"')

    def delete(self, *args):
        """
        Delete variables.

        :param args tuple of str: the variables to delete.
        """
        for arg in args:
            self.variable(arg, 'delete')


class Script(Ave):
    """
    Customized with relaxation cycles.
    """
    MODULUS_VAR = f'${{{RampUp.MODULUS}}}'
    FACT = 'fact'
    EVERY = 'every'

    def __init__(self, *args, loop_num=1000, wnum=3, rec_num=100, **kwargs):
        """
        :param loop_num int: the maximum number of cycles.
        :param wnum int: the number of sinusoidal waves in each cycle.
        :param rec_num int: each sinusoidal wave records this number of data.
        """
        super().__init__(*args, **kwargs)
        self.loop_num = loop_num
        self.wnum = wnum
        self.rec_num = rec_num
        # Maximum Total Cycle Steps: self.relax_steps * 10.
        # Deformation and relaxation cost one more additional self.wstep
        self.wstep = int(self.relax_step * 10 / loop_num / (self.wnum + 1))
        self.wstep = max([int(self.wstep / self.rec_num), 10]) * self.rec_num

    def rampUp(self):
        """
        Ramp up temperature to the targe value.
        """
        if self.options.prod_ens == self.NPT:
            super().rampUp()
            return
        if not self.relax_step:
            return
        # NVT at low temperature
        self.nvt(nstep=self.relax_step / 2E1,
                 stemp=self.options.stemp,
                 temp=self.options.temp)
        # Ramp up with constant volume
        self.nvt(nstep=self.relax_step / 2E1,
                 stemp=self.options.temp,
                 temp=self.options.temp)
        # Change the volume to approach the target pressure (1 frame per cycle)
        with self.tmp_dump(self.DUMP_ID, self.EVERY,
                           {self.wstep * (self.wnum + 1): self.DUMP_Q}):
            self.cycle()
        # NVT and NPT relaxation to reach the exact target pressure
        self.nvt(nstep=self.relax_step / 1E1, temp=self.options.temp)
        self.equal('press', 'press')
        self.npt(nstep=self.relax_step / 1E1,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 spress='${press}',
                 press=self.options.press,
                 modulus=self.MODULUS_VAR)

    @contextlib.contextmanager
    def tmp_dump(self, dump_id, key, kwargs):
        """
        Temporarily change the trajectory dump.

        :param dump_id int: the dump id to modify.
        :param key str: keyword.
        :param kwargs dict: temporarily values and values to restore
        """
        if self.dump_file.endswith(self.XTC_EXT) and key == self.EVERY:
            # ERROR: Cannot change dump_modify every for dump xtc (src/EXTRA-DUMP/dump_xtc.cpp:144)
            yield
            return
        self.dump_modify(dump_id, key, *kwargs.keys())
        try:
            yield
        finally:
            self.dump_modify(dump_id, key, *kwargs.values())

    def cycle(self,
              defm_id='defm_id',
              defm_start='defm_start',
              defm_break='defm_break',
              imodulus='imodulus'):
        """
        Deform the box to get close to the target pressure by cycles, in which
        the system wiggle, adjust, and relax.

        :param defm_id str: deformation id loop from 0 to loop_num - 1
        :param defm_start str: label to start the deformation loop
        :param defm_break str: terminated deformation goes to this label
        """
        self.variable(defm_id, 'loop', 0, self.loop_num - 1, 'pad')
        # Loop start
        self.join('label', defm_start)
        self.print(defm_id)
        with self.tmp_dir(f"defm_${{{defm_id}}}") as cdw:
            # Wiggle for factor
            file = self.wiggle()
            self.python(self.FACT, 'getBdryFact', 'fsf', self.options.press,
                        file)
            self.print(self.FACT)
            self.if_then(f"${{{self.FACT}}} == 1", f'jump SELF {defm_break}')
            # Adjust and continue
            self.adjust()
            self.shell(*cdw)
            self.join('next', defm_id)
            self.join('jump', 'SELF', defm_start, newline=True)
            # Break with the modulus evaluation as (files in subdir).
            self.join('label', defm_break)
            self.python(imodulus, 'getModulus', 'sif', file, self.rec_num)
            self.equal(self.MODULUS, imodulus, bracked=True)
            self.delete(imodulus, defm_id)

    @contextlib.contextmanager
    def tmp_dir(self, dirname, cdw=('cd', os.pardir)):
        """
        Temporarily change the trajectory dump.

        :param dirname str: the temporary dirname to cd into.
        :return cdw tuple: the shell command to go back to the working directory.
        """
        self.shell('mkdir', dirname)
        self.shell('cd', dirname, newline=True)
        try:
            yield cdw
        finally:
            self.shell(*cdw)

    def wiggle(self, file='press_vol', vol='vol', amp='amp'):
        """
        Wiggle simulation.

        :param file str: the filename to record pressure and volume.
        :param vol str: the volume variable name.
        :param amp str: the wiggle amplitude.
        :return filename: the pressure and volume filename.
        """
        self.equal(vol, vol)
        self.equal(amp, f"0.01*v_{vol}^(1/3)")
        with self.block() as blk:
            parm = f"%s wiggle ${{{amp}}} {self.wstep * self.options.timestep}"
            blk.deform(self.rec_num, parm=parm)
            blk.ave_time('c_thermo_press',
                         f'v_{vol}',
                         file=file,
                         nstep=self.wstep,
                         num=self.rec_num)
            blk.nvt(nstep=self.wstep * self.wnum,
                    stemp=self.options.temp,
                    temp=self.options.temp)
        self.delete(vol, amp)
        return file

    def deform(self, period, *args, parm=None):
        """
        Deform command.

        :param period int: perform box deformation every this many timesteps.
        :param parm str: the parameter in one dimension.
        """
        if parm:
            args = tuple(parm % i for i in self.XYZ) + args
        self.fix_all('deform', period, *args)

    def adjust(self):
        """
        Adjust the simulation box by deformation and NPT.
        """
        with self.block() as blk:
            blk.deform(100, 'remap', 'v', parm=f"%s scale ${{{self.FACT}}}")
            blk.nvt(nstep=self.wstep / 2,
                    stemp=self.options.temp,
                    temp=self.options.temp)
        self.delete(self.FACT)
        self.nvt(nstep=self.wstep / 2,
                 stemp=self.options.temp,
                 temp=self.options.temp)

    def if_then(self, cond, action, **kwargs):
        """
        If command.

        :param cond str: the condition to trigger the action.
        :param action str: the ation to take.
        """
        self.join('if', f'"{cond}"', 'then', f'"{action}"', **kwargs)

    def shell(self, *args, **kwargs):
        """
        Shell command.
        """
        self.join('shell', *args, **kwargs)

    def relaxation(self, modulus=MODULUS_VAR):
        """
        See parent.
        """
        super().relaxation(modulus=modulus)

    def production(self, modulus=MODULUS_VAR):
        """
        See parent.
        """
        super().production(modulus=modulus)
