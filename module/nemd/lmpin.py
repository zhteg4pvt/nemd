# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module writes Lammps in script.
"""
import functools
import os
import string

import numpy as np
import scipy

from nemd import builtinsutils
from nemd import constants
from nemd import lmpfix
from nemd import symbols


class In(builtinsutils.Object):
    """
    Class to write a LAMMPS in script.

    https://docs.lammps.org/commands_list.html
    """

    IN_EXT = '.in'
    DATA_EXT = '.data'
    CUSTOM_EXT = f"{lmpfix.CUSTOM_EXT}.gz"

    UNITS = 'units'
    REAL = 'real'
    METAL = 'metal'

    ATOM_STYLE = 'atom_style'
    LJ = 'LJ'
    FULL = 'full'
    ATOMIC = 'atomic'

    PAIR_STYLE = 'pair_style'
    SW = 'sw'
    LJ_CUT = 'lj/cut'
    LJ_CUT_COUL_LONG = 'lj/cut/coul/long'
    DEFAULT_CUT = symbols.DEFAULT_CUT
    DEFAULT_LJ_CUT = DEFAULT_CUT
    DEFAULT_COUL_CUT = f"{DEFAULT_CUT} {DEFAULT_CUT}"

    PAIR_COEFF = 'pair_coeff'
    DUMP_ID, DUMP_Q = lmpfix.DUMP_ID, lmpfix.DUMP_Q

    FIX_RESTRAIN = lmpfix.FIX_RESTRAIN
    UNFIX_RESTRAIN = lmpfix.UNFIX_RESTRAIN

    MIN_STYLE = 'min_style'
    FIRE = 'fire'
    MINIMIZE = 'minimize'

    FIX_RIGID_SHAKE = lmpfix.FIX_RIGID_SHAKE

    TIMESTEP = 'timestep'
    THERMO_MODIFY = 'thermo_modify'
    THERMO = 'thermo'

    V_UNITS = METAL
    V_ATOM_STYLE = ATOMIC
    V_PAIR_STYLE = SW

    def __init__(self, options=None):
        """
        :param options 'argparse.Namespace': command line options
        """
        self.options = options
        self.fh = None
        jobname = self.options.JOBNAME if self.options else self.name
        self.inscript = jobname + self.IN_EXT
        self.datafile = jobname + self.DATA_EXT
        self.dumpfile = jobname + self.CUSTOM_EXT

    def writeIn(self):
        """
        Write out LAMMPS in script.
        """
        with open(self.inscript, 'w') as self.fh:
            self.setup()
            self.pair()
            self.data()
            self.coeff()
            self.traj()
            self.minimize()
            self.shake()
            self.timestep()
            self.simulation()

    def setup(self):
        """
        Write the setup section including unit and atom styles.
        """
        self.fh.write(f"{self.UNITS} {self.V_UNITS}\n")
        self.fh.write(f"{self.ATOM_STYLE} {self.V_ATOM_STYLE}\n")

    def pair(self):
        """
        Write pair style.
        """
        self.fh.write(f"{self.PAIR_STYLE} {self.V_PAIR_STYLE}\n")

    def data(self):
        """
        Write data file related information.
        """
        self.fh.write(f"{lmpfix.READ_DATA} {self.datafile}\n\n")

    def coeff(self):
        """
        Write pair coefficients when data file doesn't contain the coefficients.
        """
        pass

    def traj(self, xyz=True, force=False, sort=True, fmt=None):
        """
        Dump out trajectory.

        :param xyz bool: write xyz coordinates if True
        :param force bool: write force on each atom if True
        :param sort bool: sort by atom id if True
        :param fmt str: the float format
        """
        attrib = ['id']
        if xyz:
            attrib += ['xu', 'yu', 'zu']
        if force:
            attrib += ['fx', 'fy', 'fz']
        attrib = ' '.join(attrib)
        cmd = lmpfix.DUMP_CUSTOM.format(file=self.dumpfile, attrib=attrib)
        self.fh.write(cmd)
        # Dumpy modify
        attrib = []
        if sort:
            attrib += ['sort', 'id']
        if fmt:
            attrib += ['format', fmt]
        if not attrib:
            return
        self.fh.write(lmpfix.DUMP_MODIFY.format(attrib=' '.join(attrib)))

    def minimize(self, min_style=FIRE):
        """
        Write commands related to minimization.

        :param min_style str: cg, fire, spin, etc.
        """
        if self.options is None or self.options.no_minimize:
            return
        if self.rest:
            self.fh.write(self.rest)
        self.fh.write(f"{self.MIN_STYLE} {min_style}\n")
        self.fh.write(f"{self.MINIMIZE} 1.0e-6 1.0e-8 1000000 10000000\n")
        if self.rest:
            self.fh.write(self.UNFIX_RESTRAIN)

    @property
    @functools.cache
    def rest(self):
        """
        Return the command to enforce specified restrain on the geometry.

        :return str: the command to fix the restrained geometry
        """
        pass

    def shake(self, bonds=None, angles=None):
        """
        Write fix shake command to enforce constant bond length and angel values.

        :param bonds list: the rigid bond type ids.
        :param angles list: the rigid angle type ids.
        """
        fixed_types = ''
        if bonds:
            fixed_types += f' b {bonds}'
        if angles:
            fixed_types += f' a {angles}'
        if not fixed_types:
            return
        self.fh.write(self.FIX_RIGID_SHAKE.format(types=fixed_types))

    def timestep(self):
        """
        Write commands related to timestep.
        """
        if self.options is None or not self.options.temp:
            return
        time = self.options.timestep * scipy.constants.femto
        self.fh.write(f'\n{self.TIMESTEP} {time / self.time_unit() }\n')
        self.fh.write(f'{self.THERMO_MODIFY} flush yes\n')
        self.fh.write(f'{self.THERMO} 1000\n')

    def simulation(self, atom_total=1):
        """
        Write command to further equilibration and production

        :param atom_total int: total atom number.
        """
        if self.options is None:
            return
        fwriter = FixWriter(self.fh,
                            options=self.options,
                            atom_total=atom_total)
        fwriter.run()

    @classmethod
    def time_unit(cls, unit=None):
        """
        Return the time unit in the LAMMPS input file.

        :unit str: the unit of the input script.
        :return float: the time unit in the LAMMPS input file.
        """
        match unit if unit else cls.V_UNITS:
            case cls.REAL:
                return scipy.constants.femto
            case cls.METAL:
                return scipy.constants.pico
            case _:
                raise ValueError(f"Invalid unit: {unit}")


class FixWriter:
    """
    This the wrapper for LAMMPS fix command writer. which usually includes an
    "unfix" after the run command.
    """
    VELOCITY = lmpfix.VELOCITY
    SET_VAR = lmpfix.SET_VAR
    NVE = lmpfix.NVE
    NVT = lmpfix.NVT
    NPT = lmpfix.NPT
    FIX = lmpfix.FIX
    FIX_NVE = lmpfix.FIX_NVE
    BERENDSEN = lmpfix.BERENDSEN
    FIX_TEMP_BERENDSEN = lmpfix.FIX_TEMP_BERENDSEN
    FIX_PRESS_BERENDSEN = lmpfix.FIX_PRESS_BERENDSEN
    RUN_STEP = lmpfix.RUN_STEP
    UNFIX = lmpfix.UNFIX
    RECORD_BDRY = lmpfix.RECORD_BDRY
    DUMP_EVERY = lmpfix.DUMP_EVERY
    DUMP_ID = lmpfix.DUMP_ID
    DUMP_Q = lmpfix.DUMP_Q
    VOL = lmpfix.VOL
    AMP = lmpfix.AMP
    IMMED_PRESS = lmpfix.IMMED_PRESS
    SET_IMMED_PRESS = lmpfix.SET_IMMED_PRESS
    PRESS = lmpfix.PRESS
    SET_PRESS = lmpfix.SET_PRESS
    IMMED_MODULUS = lmpfix.IMMED_MODULUS
    SET_IMMED_MODULUS = lmpfix.SET_IMMED_MODULUS
    MODULUS = lmpfix.MODULUS
    SET_MODULUS = lmpfix.SET_MODULUS
    FACTOR = lmpfix.FACTOR
    SET_FACTOR = lmpfix.SET_FACTOR
    SET_LABEL = lmpfix.SET_LABEL
    FIX_DEFORM = lmpfix.FIX_DEFORM
    WIGGLE_VOL = lmpfix.WIGGLE_VOL
    RECORD_PRESS_VOL = lmpfix.RECORD_PRESS_VOL
    CHANGE_BDRY = lmpfix.CHANGE_BDRY
    SET_LOOP = lmpfix.SET_LOOP
    MKDIR = lmpfix.MKDIR
    CD = lmpfix.CD
    JUMP = lmpfix.JUMP
    IF_JUMP = lmpfix.IF_JUMP
    PRINT = lmpfix.PRINT
    NEXT = lmpfix.NEXT
    DEL_VAR = lmpfix.DEL_VAR
    PRESS_VAR = f'${{{PRESS}}}'
    MODULUS_VAR = f'${{{MODULUS}}}'

    def __init__(self, fh, options=None, atom_total=1):
        """
        :param fh '_io.TextIOWrapper': file handler to write fix commands
        :param options 'types.Namespace': command line options and structure
            information such as bond types, angle types, and testing flag.
        :param atom_total int: total number of atoms.
        """
        self.fh = fh
        self.options = options
        self.atom_total = atom_total
        self.single_point = atom_total == 1 or not self.options.temp
        self.cmd = []
        if not self.options.temp:
            return
        self.timestep = self.options.timestep
        self.relax_time = self.options.relax_time
        self.prod_time = self.options.prod_time
        self.stemp = self.options.stemp
        self.temp = self.options.temp
        self.tdamp = self.options.timestep * self.options.tdamp
        self.press = self.options.press
        self.pdamp = self.options.timestep * self.options.pdamp
        timestep = self.options.timestep / constants.NANO_TO_FEMTO
        self.prod_step = int(self.prod_time / timestep)
        self.relax_step = int(self.relax_time / timestep)
        if self.relax_step:
            self.relax_step = min(round(self.relax_time / 1E3), 1) * 1E3

    def run(self):
        """
        Main method to run the writer.
        """
        self.singlePoint()
        self.velocity()
        self.startLow()
        self.rampUp()
        self.relaxAndDefrom()
        self.production()
        self.write()

    def singlePoint(self):
        """
        Single point energy calculation.

        :nstep int: run this steps for time integration.
        """
        if not self.single_point:
            return
        self.fh.write(self.RUN_STEP % 0)

    def velocity(self):
        """
        Create initial velocity for the system.

        https://docs.lammps.org/velocity.html
        """
        if self.single_point:
            return
        seed = np.random.randint(1, high=symbols.MAX_INT32)
        temp = self.options.stemp if self.relax_step else self.options.temp
        cmd = f"{self.VELOCITY} all create {temp} {seed}"
        self.cmd.append(cmd)

    def startLow(self):
        """
        Start simulation from low temperature and constant volume.
        """
        if self.single_point or not self.relax_step:
            return
        self.nvt(nstep=self.relax_step / 1E3,
                 stemp=self.stemp,
                 temp=self.stemp)

    def nvt(self, nstep=1E4, stemp=None, temp=300, style=BERENDSEN, pre=''):
        """
        Append command for constant volume and temperature.

        :nstep int: run this steps for time integration
        :stemp float: starting temperature
        :temp float: target temperature
        :style str: the style for the command
        :pre str: additional pre-conditions
        """
        if stemp is None:
            stemp = temp
        if style == self.BERENDSEN:
            cmd1 = self.FIX_TEMP_BERENDSEN.format(stemp=stemp,
                                                  temp=temp,
                                                  tdamp=self.tdamp)
            cmd2 = self.FIX_NVE
        cmd = pre + cmd1 + cmd2
        fix = [x for x in cmd.split(symbols.RETURN) if x.startswith(self.FIX)]
        self.cmd.append(cmd + self.RUN_STEP % nstep + self.UNFIX * len(fix))

    def rampUp(self, ensemble=None):
        """
        Ramp up temperature to the targe value.

        :ensemble str: the ensemble to ramp up temperature.

        NOTE: ensemble=None runs NVT at low temperature and ramp up with constant
        volume, calculate the averaged pressure at high temperature, and changes
        volume to reach the target pressure.
        """
        if self.single_point or not self.relax_step:
            return
        if ensemble == self.NPT:
            self.npt(nstep=self.relax_step / 1E1,
                     stemp=self.stemp,
                     temp=self.temp,
                     press=self.press)
            return

        self.nvt(nstep=self.relax_step / 2E1, stemp=self.stemp, temp=self.temp)
        self.nvt(nstep=self.relax_step / 2E1, stemp=self.temp, temp=self.temp)
        self.cycleToPress()
        self.nvt(nstep=self.relax_step / 1E1, temp=self.temp)
        self.npt(nstep=self.relax_step / 1E1,
                 stemp=self.temp,
                 temp=self.temp,
                 spress=self.PRESS_VAR,
                 press=self.press,
                 modulus=self.MODULUS_VAR)

    def npt(self,
            nstep=20000,
            stemp=300,
            temp=300,
            spress=1.,
            press=1.,
            style=BERENDSEN,
            modulus=10,
            pre=''):
        """
        Append command for constant pressure and temperature.

        :nstep int: run this steps for time integration
        :stemp int: starting temperature
        :temp float: target temperature
        :spress float: starting pressure
        :press float: target pressure
        :style str: the style for the command
        :pre str: additional pre-conditions
        """
        if spress is None:
            spress = press
        if style == self.BERENDSEN:
            cmd1 = self.FIX_PRESS_BERENDSEN.format(spress=spress,
                                                   press=press,
                                                   pdamp=self.pdamp,
                                                   modulus=modulus)
            cmd2 = self.FIX_TEMP_BERENDSEN.format(stemp=stemp,
                                                  temp=temp,
                                                  tdamp=self.tdamp)
            cmd3 = self.FIX_NVE
        cmd = pre + cmd1 + cmd2 + cmd3
        fix = [x for x in cmd.split(symbols.RETURN) if x.startswith(self.FIX)]
        self.cmd.append(cmd + self.RUN_STEP % nstep + self.UNFIX * len(fix))

    def cycleToPress(self,
                     max_loop=1000,
                     num=3,
                     record_num=100,
                     defm_id='defm_id',
                     defm_start='defm_start',
                     defm_break='defm_break'):
        """
        Deform the box by cycles to get close to the target pressure.
        One cycle consists of sinusoidal wave, print properties, deformation,
        and relaxation. The max total simulation time for the all cycles is the
        regular relaxation simulation time.

        :param max_loop int: the maximum number of big cycle loops.
        :param num int: the number of sinusoidal cycles.
        :param record_num int: each sinusoidal wave records this number of data.
        :param defm_id str: Deformation id loop from 0 to max_loop - 1
        :param defm_start str: Each deformation loop starts with this label
        :param defm_break str: Terminate the loop by go to this label
        """
        nstep = int(self.relax_step * 10 / max_loop / (num + 1))
        # One sinusoidal cycle that yields at least 10 records
        nstep = max([int(nstep / record_num), 10]) * record_num
        # Maximum Total Cycle Steps (cyc_nstep): self.relax_steps * 10
        cyc_nstep = nstep * (num + 1)
        # Each cycle dumps one trajectory frame
        self.cmd.append(self.DUMP_EVERY.format(id=self.DUMP_ID, arg=cyc_nstep))
        # Set variables used in the loop
        self.cmd.append(self.SET_VAR.format(var=self.VOL, expr=self.VOL))
        expr = f'0.01*v_{self.VOL}^(1/3)'
        self.cmd.append(self.SET_VAR.format(var=self.AMP, expr=expr))
        self.cmd.append(self.SET_IMMED_PRESS)
        self.cmd.append(self.SET_IMMED_MODULUS.format(record_num=record_num))
        self.cmd.append(self.SET_FACTOR.format(press=self.options.press))
        self.cmd.append(self.SET_LOOP.format(id=defm_id, end=max_loop - 1))
        self.cmd.append(self.SET_LABEL.format(label=defm_start))
        self.cmd.append(self.PRINT.format(var=defm_id))
        # Run in a subdirectory as some output files are of the same names
        dirname = f"defm_${{{defm_id}}}"
        self.cmd.append(self.MKDIR.format(dir=dirname))
        self.cmd.append(self.CD.format(dir=dirname))
        self.cmd.append("")
        pre = self.getCyclePre(nstep, record_num=record_num)
        self.nvt(nstep=nstep * num, stemp=self.temp, temp=self.temp, pre=pre)
        self.cmd.append(self.PRINT.format(var=self.IMMED_PRESS))
        self.cmd.append(self.PRINT.format(var=self.IMMED_MODULUS))
        self.cmd.append(self.PRINT.format(var=self.FACTOR))
        cond = f"${{{defm_id}}} == {max_loop - 1} || ${{{self.FACTOR}}} == 1"
        self.cmd.append(self.IF_JUMP.format(cond=cond, label=defm_break))
        self.cmd.append("")
        self.nvt(nstep=nstep / 2,
                 stemp=self.temp,
                 temp=self.temp,
                 pre=self.FIX_DEFORM)
        self.nvt(nstep=nstep / 2, stemp=self.temp, temp=self.temp)
        self.cmd.append(self.CD.format(dir=os.pardir))
        self.cmd.append(self.NEXT.format(id=defm_id))
        self.cmd.append(self.JUMP.format(label=defm_start))
        self.cmd.append("")
        self.cmd.append(self.SET_LABEL.format(label=defm_break))
        # Record press and modulus as immediate variable evaluation uses files
        self.cmd.append(self.SET_MODULUS)
        self.cmd.append(self.SET_PRESS)
        self.cmd.append(self.CD.format(dir=os.pardir))
        # Delete variables used in the loop
        self.cmd.append(self.DEL_VAR.format(var=self.VOL))
        self.cmd.append(self.DEL_VAR.format(var=self.AMP))
        self.cmd.append(self.DEL_VAR.format(var=self.IMMED_PRESS))
        self.cmd.append(self.DEL_VAR.format(var=self.IMMED_MODULUS))
        self.cmd.append(self.DEL_VAR.format(var=self.FACTOR))
        self.cmd.append(self.DEL_VAR.format(var=defm_id))
        # Restore dump defaults
        cmd = '\n' + self.DUMP_EVERY.format(id=self.DUMP_ID, arg=self.DUMP_Q)
        self.cmd.append(cmd)

    def getCyclePre(self, nstep, record_num=100):
        """
        Get the pre-stage str for the cycle simulation.

        :param nstep int: the simulation steps of the one cycles
        :param record_num int: each cycle records this number of data
        :return str: the prefix string of the cycle stage.
        """

        wiggle = self.WIGGLE_VOL.format(period=nstep * self.timestep)
        record_period = int(nstep / record_num)
        record_press = self.RECORD_PRESS_VOL.format(period=record_period)
        return record_press + wiggle

    def relaxAndDefrom(self):
        """
        Longer relaxation at constant temperature and deform to the mean size.
        """
        if self.single_point or not self.relax_step:
            return
        if self.options.prod_ens == self.NPT:
            self.npt(nstep=self.relax_step,
                     stemp=self.temp,
                     temp=self.temp,
                     press=self.press,
                     modulus=self.MODULUS_VAR)
            return
        # NVE and NVT production runs use averaged cell
        pre = self.RECORD_BDRY.format(num=self.relax_step / 1E1)
        self.npt(nstep=self.relax_step,
                 stemp=self.temp,
                 temp=self.temp,
                 press=self.press,
                 modulus=self.MODULUS_VAR,
                 pre=pre)
        self.cmd.append(self.CHANGE_BDRY)
        self.nvt(nstep=self.relax_step / 1E2, stemp=self.temp, temp=self.temp)

    def production(self):
        """
        Production run. NVE is good for all, specially transport properties, but
        requires for good energy conservation in time integration. NVT and NPT
        may help for disturbance non-sensitive property.
        """
        if self.single_point:
            return
        if self.options.prod_ens == self.NVE:
            self.nve(nstep=self.prod_step)
        elif self.options.prod_ens == self.NVT:
            self.nvt(nstep=self.prod_step, stemp=self.temp, temp=self.temp)
        else:
            self.npt(nstep=self.prod_step,
                     stemp=self.temp,
                     temp=self.temp,
                     press=self.press,
                     modulus=self.MODULUS_VAR)

    def nve(self, nstep=1E3):
        """
        Append command for constant energy and volume.

        :param nstep int: run this steps for time integration.
        """
        # NVT on single molecule gives nan coords (guess due to translation)
        cmd = self.FIX_NVE + self.RUN_STEP % nstep + self.UNFIX
        self.cmd.append(cmd)

    def write(self):
        """
        Write the command to the file.
        """
        for idx, cmd in enumerate(self.cmd, 1):
            num = round(cmd.count('%s') / 2)
            ids = [f'{idx}{string.ascii_lowercase[x]}' for x in range(num)]
            ids += [x for x in reversed(ids)]
            cmd = cmd % tuple(ids) if ids else cmd
            self.fh.write(cmd + '\n')
        self.fh.write('quit 0\n')
