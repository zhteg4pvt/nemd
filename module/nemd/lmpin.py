# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module writes Lammps in script.

https://docs.lammps.org/commands_list.html
"""
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

    RUN_STEP = "run %i"

    V_UNITS = METAL
    V_ATOM_STYLE = ATOMIC
    V_PAIR_STYLE = SW

    def __init__(self, struct=None):
        """
        :param options 'argparse.Namespace': command line options.
        """
        super().__init__()
        self.struct = struct
        self.options = self.struct.options
        self.outfile = f"{self.options.JOBNAME}.in"

    def write(self):
        """
        Write the command to the file.
        """
        self.setUp()
        with open(self.outfile, 'w') as fh:
            for idx, cmd in enumerate(self, 1):
                num = round(cmd.count('%s') / 2)
                ids = [f'{idx}{string.ascii_lowercase[x]}' for x in range(num)]
                ids += [x for x in reversed(ids)]
                cmd = cmd % tuple(ids) if ids else cmd
                fh.write(cmd + '\n')
            fh.write('quit 0\n')

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
        self.append(f"{self.UNITS} {self.V_UNITS}")
        self.append(f"atom_style {self.V_ATOM_STYLE}")

    def pair(self):
        """
        Write pair style.
        """
        self.append(f"{self.PAIR_STYLE} {self.V_PAIR_STYLE}")

    def data(self):
        """
        Write data file related information.
        """
        self.append(f"{self.READ_DATA} {self.options.JOBNAME}.data")

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
        file = f"{self.options.JOBNAME}{self.CUSTOM_EXT}"
        args = [self.DUMP_ID, 'all', 'custom', self.DUMP_Q, file, 'id'] + args
        self.join('dump', *args)
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
        if bracked:
            expr = f'${{{expr}}}'
        if quoted:
            expr = f'"{expr}"'
        self.variable(name, 'equal', expr, **kwargs)

    def variable(self, *args, **kwargs):
        self.join('variable', *args, **kwargs)

    def join(self, *args, newline=False):
        """
        Join the arguments to form a lammps command.
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
        :param val float: the value of the restraint.
        """
        if self.options.no_minimize:
            return
        if val := geo and self.options.substruct[1]:
            self.append(f'fix rest all restrain {geo} -2000.0 -2000.0 {val}')
        self.append(f"min_style {min_style}")
        self.append(f"minimize 1.0e-6 1.0e-8 1000000 10000000")
        if val:
            self.append('unfix rest')

    def timestep(self):
        """
        Write commands related to timestep.
        """
        if not self.options.temp:
            return
        time = self.options.timestep * scipy.constants.femto
        self.append(f'\n{self.TIMESTEP} {time / self.time_unit() }')
        self.append(f'thermo_modify flush yes')
        self.append(f'thermo 1000')

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
        self.append(self.RUN_STEP % 0)


class RampUp(SinglePoint):
    """
    Customized for low temp NVT, NPT ramp up, NPT relaxation and production.
    """
    NVT = 'NVT'
    NPT = 'NPT'
    NVE = 'NVE'
    ENSEMBLES = [NVE, NVT, NPT]
    CUSTOM_EXT = f"{SinglePoint.CUSTOM_EXT}.gz"
    FIX_ID = f"fix %s"
    FIX_ALL = f"{FIX_ID} all"
    FIX_NVE = f"{FIX_ALL} nve"
    BERENDSEN = 'berendsen'
    TEMP_BERENDSEN = f"{FIX_ALL} temp/berendsen {{stemp}} {{temp}} {{tdamp}}"
    UNFIX = "unfix %s"

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
            super().simulation()
            return
        self.velocity()
        self.startLow()
        self.rampUp()
        self.relaxation()
        self.production()

    def velocity(self):
        """
        Create initial velocity for the system.
        """
        temp = self.options.stemp if self.relax_step else self.options.temp
        seed = np.random.randint(1, high=symbols.MAX_INT32)
        self.append(f"velocity all create {temp} {seed}")

    def startLow(self):
        """
        Start simulation from low temperature and constant volume.
        """
        self.nvt(nstep=self.relax_step / 1E3,
                 stemp=self.options.stemp,
                 temp=self.options.stemp)

    def nvt(self, nstep=1E4, stemp=None, temp=300, style=BERENDSEN, pre=None):
        """
        Append command for constant volume and temperature.

        :param nstep int: run this steps for time integration.
        :param stemp float: starting temperature.
        :param temp float: target temperature.
        :param style str: the style for the command.
        :param pre str: additional pre-conditions.
        """
        if not nstep:
            return
        if stemp is None:
            stemp = temp
        cmds = pre if pre else []
        if style == self.BERENDSEN:
            cmds.append(
                self.TEMP_BERENDSEN.format(stemp=stemp,
                                           temp=temp,
                                           tdamp=self.tdamp))
            cmds.append(self.FIX_NVE)
        # FIXME: support thermostat more than berendsen.
        unfixes = [self.UNFIX for x in cmds if x.startswith(self.FIX_ID)]
        self.append('\n'.join(cmds + [self.RUN_STEP % nstep] + unfixes) + '\n')

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
            stemp=300,
            temp=300,
            spress=1.,
            press=1.,
            style=BERENDSEN,
            modulus=10,
            pre=None):
        """
        Append command for constant pressure and temperature.

        :param nstep int: run this steps for time integration
        :param stemp int: starting temperature
        :param temp float: target temperature
        :param spress float: starting pressure
        :param press float: target pressure
        :param style str: the style for the command
        :param pre list: additional pre-conditions
        """
        if not nstep:
            return
        if spress is None:
            spress = press
        cmds = pre if pre else []
        if style == self.BERENDSEN:
            cmd = f"fix %s all press/berendsen iso {spress} {press} {self.pdamp} modulus {modulus}"
            cmds.append(cmd)
            cmd = self.TEMP_BERENDSEN.format(stemp=stemp,
                                             temp=temp,
                                             tdamp=self.tdamp)
            cmds.append(cmd)
            cmds.append(self.FIX_NVE)
        # FIXME: support thermostat more than berendsen.
        unfixes = [self.UNFIX for x in cmds if x.startswith(self.FIX_ID)]
        self.append('\n'.join(cmds + [self.RUN_STEP % nstep] + unfixes) + '\n')

    def relaxation(self, modulus=10):
        """
        Longer relaxation at constant temperature and deform to the mean size.
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

        :param nstep int: run this steps for time integration.
        """
        # NVT on single molecule -> nan xyz (guess due to translation)
        self.append('\n'.join(
            [self.FIX_NVE, self.RUN_STEP % nstep, self.UNFIX]))


class Ave(RampUp):
    """
    Customized with PBC averaging for NVT relaxation.
    """

    def relaxation(self, modulus=10):
        """
        Relaxation simulation.
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
        Average the boundary based on NPT ensemble.

        :param modulus float: the modules in berendsen barostat.
        :param xyz str: the filename to record boundary.
        :param rec_num int: the number of records.
        """
        # Record the xyz span durning NPT
        spans = [f'{i}l' for i in xyz]
        for name, dim in zip(spans, xyz):
            self.equal(name, f'{dim}hi - {dim}lo', quoted=True)
        num = int(self.relax_step / rec_num)
        args = ' '.join(f'v_{i}' for i in spans)
        record = f"fix %s all ave/time 1 {num} {num} {args} file {xyz}"
        self.npt(nstep=self.relax_step,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 press=self.options.press,
                 modulus=modulus,
                 pre=[record])
        self.print(*spans, label='Final')
        # Calculate the aves span
        aves = [f'ave_{i}' for i in xyz]
        for name, dim in zip(aves, xyz):
            func = f'get{dim.upper()}L'
            self.python(name, func, 'sf', xyz)
        self.print(*aves, label='Averaged')
        # Calculate the ratio and change the box
        ratios = [f'ratio_{i}' for i in xyz]
        for ratio, ave, span in zip(ratios, aves, spans):
            self.equal(ratio, f'v_{ave} / v_{span}', quoted=True)
        scales = [f'{i} scale ${{{r}}}' for i, r in zip(xyz, ratios)]
        self.join('change_box', 'all', *scales, 'remap')
        # Delete used variables
        self.delete(*spans, *aves, *ratios)

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
    MODULUS = 'modulus'
    MODULUS_VAR = f'${{{MODULUS}}}'
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
              imodulus=f'i{MODULUS}'):
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
        file = self.wiggle()
        self.python(self.FACT, 'getBdryFact', 'fsf', self.options.press, file)
        self.print(self.FACT)
        self.if_then(f"${{{self.FACT}}} == 1", f'jump SELF {defm_break}')
        self.deform()
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

    def wiggle(self, file='press_vol', vol='vol', amp='amp'):
        self.equal(vol, vol)
        self.equal(amp, f"0.01*v_{vol}^(1/3)")
        period = self.wstep * self.options.timestep
        args = [f"{i} wiggle ${{{amp}}} {period}" for i in 'xyz']
        deform = f"fix %s all deform {self.rec_num} {' '.join(args)}"
        per = int(self.wstep / self.rec_num)
        record = f"fix %s all ave/time 1 {per} {per} c_thermo_press v_{vol} file {file}"
        self.nvt(nstep=self.wstep * self.wnum,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 pre=[deform, record])
        self.delete(vol, amp)
        return file

    def deform(self):
        args = [f"{i} scale ${{{self.FACT}}}" for i in 'xyz']
        cmd = f"fix %s all deform 100 {' '.join(args)} remap v"
        self.nvt(nstep=self.wstep / 2,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 pre=[cmd])
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
