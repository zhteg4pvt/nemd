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


class SinglePoint(list):
    """
    LAMMPS in-script writer for configuration and single point simulation.
    """
    UNITS = 'units'
    LJ = 'lj'
    REAL = 'real'
    METAL = 'metal'

    ATOMIC = 'atomic'

    PAIR_STYLE = 'pair_style'
    SW = 'sw'

    READ_DATA = 'read_data'
    READ_DATA_RE = re.compile(rf'{READ_DATA}\s*([\w.]*)')
    PAIR_COEFF = 'pair_coeff'
    TIMESTEP = 'timestep'
    CUSTOM_EXT = '.custom'
    DUMP_ID, DUMP_Q = 1, 1000

    V_UNITS = METAL
    V_ATOM_STYLE = ATOMIC
    V_PAIR_STYLE = SW

    def __init__(self, struct=None, isblock=False):
        """
        :param options 'argparse.Namespace': command line options.
        :param isblock bool: whether self is already a block.
        """
        super().__init__()
        self.struct = struct
        self.isblock = isblock
        self.options = self.struct.options
        self.outfile = f"{self.options.JOBNAME}.in"

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
        self.traj()
        self.minimize()
        self.timestep()
        self.simulation()

    def setup(self):
        """
        Write the setup section including unit and atom styles.
        """
        self.join(self.UNITS, self.V_UNITS)
        self.join('atom_style', self.V_ATOM_STYLE)

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

    def traj(self, xyz=True, force=False, sort=True, fmt=None):
        """
        Dump out trajectory.

        :param xyz bool: write xyz coordinates if Truef
        :param force bool: write force on each atom if True
        :param sort bool: sort by atom id if True
        :param fmt str: the float format
        """
        args = []
        if xyz:
            args += ['xu', 'yu', 'zu']
        if force:
            args += ['fx', 'fy', 'fz']
        if not args:
            return
        self.join('dump', self.DUMP_ID, 'all', 'custom', self.DUMP_Q,
                  f"{self.options.JOBNAME}{self.CUSTOM_EXT}", 'id', *args)
        # Dumpy modify
        args = []
        if sort:
            args += ['sort', 'id']
        if fmt:
            args += ['format', fmt]
        if not args:
            return
        self.dump_modify(self.DUMP_ID, *args)

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

        :param newline: new a newline when appending the block.
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

    def join(self, *args, newline=False):
        """
        Join the arguments to form a lammps command.

        :param newline: add newline after the line.
        """
        cmd = " ".join(map(str, args))
        self.append(cmd + '\n' if newline else cmd)

    def dump_modify(self, *args):
        """
        Modify the trajectory dump.
        """
        self.join('dump_modify', *args)

    def minimize(self, min_style='fire', geo=None):
        """
        Write commands related to minimization.

        :param min_style str: cg, fire, spin, etc.
        :param geo str: the geometry to restrain (e.g., dihedral 1 2 3 4).
        """
        if self.options.no_minimize:
            return
        if val := geo and self.options.substruct[1]:
            self.append(f'fix rest all restrain {geo} -2000.0 -2000.0 {val}')
        self.join('min_style', min_style)
        self.append(f"minimize 1.0e-6 1.0e-8 1000000 10000000")
        if val:
            self.append('unfix rest')

    def timestep(self):
        """
        Write timestep-related commands.
        """
        if not self.options.temp:
            return
        self.join(
            self.TIMESTEP,
            self.options.timestep * scipy.constants.femto / self.time_unit())
        self.append('thermo_modify flush yes')
        self.append('thermo 1000')

    @classmethod
    def time_unit(cls, unit=None):
        """
        Return the time unit in the LAMMPS input file.

        :param unit str: the unit of the input script.
        :return float: the time unit in the LAMMPS input file.
        :raise ValueError: when the unit is unknown.
        """
        match unit if unit else cls.V_UNITS:
            case cls.REAL:
                return scipy.constants.femto
            case cls.METAL:
                return scipy.constants.pico
            case _:
                raise ValueError(f"Unknown unit: {unit}")

    def simulation(self):
        """
        Single point energy calculation.
        """
        self.runStep()

    def runStep(self, nstep=0):
        """
        Write run.
        """
        self.join('run', int(nstep))

    def finalize(self, block_id=0):
        """
        Finalize by balancing the fixes with unfixes, flattening the sublist,
        and terminating with the quit command.
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
        self.join('quit 0')


class RampUp(SinglePoint):
    """
    Customized for low temp NVT, NPT ramp up, NPT relaxation and production.
    """
    NVT = 'NVT'
    NPT = 'NPT'
    NVE = 'NVE'
    MODULUS = 'modulus'
    ENSEMBLES = [NVE, NVT, NPT]
    CUSTOM_EXT = f"{SinglePoint.CUSTOM_EXT}.gz"
    BERENDSEN = 'berendsen'

    def __init__(self, *args, **kwargs):
        """
        :param atom_total int: the total number of atoms.
        """
        super().__init__(*args, **kwargs)
        self.tdamp = self.options.timestep * self.options.tdamp
        self.pdamp = self.options.timestep * self.options.pdamp
        timestep = self.options.timestep / constants.NANO_TO_FEMTO
        self.prod_step = int(self.options.prod_time / timestep)
        self.relax_step = int(self.options.relax_time / timestep)
        if self.relax_step:
            self.relax_step = int(round(self.relax_step, -3) or 1E3)

    def simulation(self):
        """
        Main method to run the writer.
        """
        if not self.options.temp or self.struct.atom_total == 1:
            self.runStep()
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

        :param nstep int: run this steps for time integration.
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
                blk.fixAll('temp/berendsen', stemp, temp, self.tdamp)
                blk.nve(nstep=nstep)
            # FIXME: support thermostat more than berendsen.

    def fixAll(self, *args):
        """
        Fix all command.
        """
        self.join('fix %s all', *args)

    def rampUp(self):
        """
        Ramp up temperature to the targe value.
        """
        self.npt(nstep=self.relax_step / 1E1,
                 stemp=self.options.stemp,
                 temp=self.options.temp,
                 press=self.options.press)

    def npt(self,
            nstep=20000,
            spress=1.,
            press=1.,
            style=BERENDSEN,
            modulus=10,
            **kwargs):
        """
        Append command for constant pressure and temperature.

        :param nstep int: run this steps for time integration.
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
                blk.fixAll('press/berendsen', 'iso', spress, press, self.pdamp,
                           self.MODULUS, modulus)
                blk.nvt(nstep=nstep, **kwargs)
        # FIXME: support thermostat more than berendsen.

    def relaxation(self, modulus=10):
        """
        Longer relaxation at constant temperature.

        :param modulus float: the modulus for the barostat.
        """
        self.npt(nstep=self.relax_step,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 press=self.options.press,
                 modulus=modulus)

    def production(self, modulus=10):
        """
        Production run.

        NOTE: NVE is good for all, specially transport properties, but requires
        good energy conservation during the time integration. NVT and NPT help
        with properties non-sensitive to disturbance.
        """
        match self.options.prod_ens:
            case self.NVE:
                self.nve(nstep=self.prod_step)
            case self.NVT:
                self.nvt(nstep=self.prod_step,
                         stemp=self.options.temp,
                         temp=self.options.temp)
            case self.NPT:
                self.npt(nstep=self.prod_step,
                         stemp=self.options.temp,
                         temp=self.options.temp,
                         press=self.options.press,
                         modulus=modulus)

    def nve(self, nstep=1E3):
        """
        Constant energy and volume.

        NOTE: NVT works with single module while NVT yields nan xyz

        :param nstep int: run this steps for time integration.
        """
        with self.block() as blk:
            blk.fixAll('nve')
            blk.runStep(nstep=nstep)


class Ave(RampUp):
    """
    Customized with PBC averaging for NVT relaxation.
    """

    def relaxation(self, modulus=10):
        """
        Customized with cell averaging.
        """
        if self.options.prod_ens == self.NPT:
            super().relaxation(modulus=modulus)
            return
        # NVE and NVT production runs use averaged cell
        self.average(modulus=modulus)
        self.nvt(nstep=self.relax_step / 1E2,
                 stemp=self.options.temp,
                 temp=self.options.temp)

    def average(self, modulus=10, xyz='xyz', rec_num=10):
        """
        Change the box to the average boundary of a NPT simulation.

        :param modulus float: the modules in berendsen barostat.
        :param xyz str: the filename to record boundary.
        :param rec_num int: the number of records.
        """
        # Run NPT with boundary recorded
        spans = [f'{i}l' for i in xyz]
        for name, dim in zip(spans, xyz):
            self.equal(name, f'{dim}hi - {dim}lo', quoted=True)
        with self.block() as blk:
            blk.aveTime(*[f'v_{i}' for i in spans],
                        'file',
                        xyz,
                        nstep=self.relax_step,
                        num=rec_num)
            blk.npt(nstep=self.relax_step,
                    stemp=self.options.temp,
                    temp=self.options.temp,
                    press=self.options.press,
                    modulus=modulus)
        self.print(*spans, label='Final')
        # Calculate the aves span, ratio to remap, and change the box.
        aves = [f'ave_{i}' for i in xyz]
        for name, dim in zip(aves, xyz):
            func = f'get{dim.upper()}L'
            self.python(name, func, 'sf', xyz)
        self.print(*aves, label='Averaged')
        ratios = [f'ratio_{i}' for i in xyz]
        for ratio, ave, span in zip(ratios, aves, spans):
            self.equal(ratio, f'v_{ave} / v_{span}', quoted=True)
        scales = [f'{i} scale ${{{r}}}' for i, r in zip(xyz, ratios)]
        self.join('change_box', 'all', *scales, 'remap')
        # Delete used variables
        self.delete(*spans, *aves, *ratios)

    def aveTime(self, *args, nstep=None, num=None):
        if nstep and num:
            per = int(nstep / num)
            args = (1, per, per) + args
        self.fixAll('ave/time', *args)

    def python(self, name, func, fmt, *inputs):
        """
        Construct a python command.

        :param func str: the function name.
        :param iargs list: the input variables of the function.
        :param rargs list: the output variables of the function.
        :param farg str: the type of the input and output variables.
        """
        self.variable(name, 'python', func)
        args = [func]
        args += ['input', len(inputs), *inputs]
        args += ['return', f'v_{name}']
        args += ['format', fmt]
        args += ['here', f'"from nemd.lmpfunc import {func}"']
        self.join('python', *args)

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
        for arg in args:
            self.variable(arg, 'delete')


class Script(Ave):
    """
    Customized with relaxation cycles.
    """
    MODULUS_VAR = f'${{{RampUp.MODULUS}}}'
    FACT = 'fact'

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
        # Maximum Total Cycle Steps (cyc_nstep): self.relax_steps * 10
        # The deformation and relaxation cost one additional self.wstep
        self.wstep = int(self.relax_step * 10 / loop_num / (self.wnum + 1))
        self.wstep = max([int(self.wstep / self.rec_num), 10]) * self.rec_num

    def rampUp(self, ensemble=None):
        """
        Ramp up temperature to the targe value.

        :param ensemble str: the ensemble to ramp up temperature.

        NOTE: ensemble=None runs NVT at low temperature and ramp up with constant
        volume, calculate the averaged pressure at high temperature, and changes
        volume to reach the target pressure.
        """
        if ensemble == self.NPT:
            super().rampUp()
            return
        self.nvt(nstep=self.relax_step / 2E1,
                 stemp=self.options.stemp,
                 temp=self.options.temp)
        self.nvt(nstep=self.relax_step / 2E1,
                 stemp=self.options.temp,
                 temp=self.options.temp)
        self.cycle()
        self.nvt(nstep=self.relax_step / 1E1, temp=self.options.temp)
        self.equal('press', 'press')
        self.npt(nstep=self.relax_step / 1E1,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 spress='${press}',
                 press=self.options.press,
                 modulus=self.MODULUS_VAR)

    def cycle(self,
              defm_id='defm_id',
              defm_start='defm_start',
              defm_break='defm_break',
              imodulus='imodulus'):
        """
        Deform the box by cycles to get close to the target pressure.
        One cycle: sinusoidal wave, printing, deformation, and relaxation.

        :param defm_id str: deformation id loop from 0 to loop_num - 1
        :param defm_start str: label to start the deformation loop
        :param defm_break str: terminated deformation goes to this label
        """
        # Each cycle dumps one trajectory frame
        self.dump_modify(self.DUMP_ID, 'every', self.wstep * (self.wnum + 1))
        # Set variables used in the loop
        self.variable(defm_id, 'loop', 0, self.loop_num - 1, 'pad')
        self.join('label', defm_start)
        self.print(defm_id)
        # Run in a subdirectory as some output files are of the same names
        dirname = f"defm_${{{defm_id}}}"
        self.shell('mkdir', dirname)
        self.shell('cd', dirname, newline=True)
        file = self.wiggleBox()
        self.python(self.FACT, 'getBdryFact', 'fsf', self.options.press, file)
        self.print(self.FACT)
        self.if_then(f"${{{self.FACT}}} == 1", f'jump SELF {defm_break}')
        self.deformBox()
        self.shell('cd', os.pardir)
        self.join('next', defm_id)
        self.join('jump', 'SELF', defm_start, newline=True)
        self.join('label', defm_break)
        self.python(imodulus, 'getModulus', 'sif', file, self.rec_num)
        # Record modulus as immediate variable evaluation uses files in subdir.
        self.equal(self.MODULUS, imodulus, bracked=True)
        self.delete(imodulus, defm_id)
        self.shell('cd', os.pardir)
        # Restore dump defaults
        self.dump_modify(self.DUMP_ID, 'every', self.DUMP_Q)

    def wiggleBox(self, file='press_vol', vol='vol', amp='amp'):
        self.equal(vol, vol)
        self.equal(amp, f"0.01*v_{vol}^(1/3)")
        with self.block() as blk:
            parm = f"%s wiggle ${{{amp}}} {self.wstep * self.options.timestep}"
            blk.deform(self.rec_num, parm=parm)
            blk.aveTime('c_thermo_press',
                        f'v_{vol}',
                        'file',
                        file,
                        nstep=self.wstep,
                        num=self.rec_num)
            blk.nvt(nstep=self.wstep * self.wnum,
                    stemp=self.options.temp,
                    temp=self.options.temp)
        self.delete(vol, amp)
        return file

    def deform(self, per, *args, dim='xyz', parm=None):
        if parm:
            args = tuple(parm % i for i in dim) + args
        self.fixAll('deform', per, *args)

    def deformBox(self):
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
        self.join('if', f'"{cond}"', 'then', f'"{action}"', **kwargs)

    def shell(self, *args, **kwargs):
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
