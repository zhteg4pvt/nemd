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
from nemd import lmpfix
from nemd import symbols

DUMP = 'dump'
DUMP_MODIFY = f'{DUMP}_modify'


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
    DUMP_ALL_CUSTOM = f"{DUMP} {DUMP_ID} all custom"
    DUMP_CUSTOM = f"{DUMP_ALL_CUSTOM} {DUMP_Q} {{file}} {{attrib}}"
    DUMP_MODIFY = f"{DUMP_MODIFY} {DUMP_ID} {{attrib}}"

    RUN_STEP = lmpfix.RUN_STEP

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
        attrib = []
        if xyz:
            attrib += ['xu', 'yu', 'zu']
        if force:
            attrib += ['fx', 'fy', 'fz']
        if not attrib:
            return
        attrib = ' '.join(['id'] + attrib)
        cmd = self.DUMP_CUSTOM.format(
            file=f"{self.options.JOBNAME}{self.CUSTOM_EXT}", attrib=attrib)
        self.append(cmd)
        # Dumpy modify
        attrib = []
        if sort:
            attrib += ['sort', 'id']
        if fmt:
            attrib += ['format', fmt]
        if not attrib:
            return
        self.append(self.DUMP_MODIFY.format(attrib=' '.join(attrib)))

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
    FIX_NVE = "fix %s all nve"
    BERENDSEN = 'berendsen'
    UNFIX = lmpfix.UNFIX

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
                lmpfix.TEMP_BERENDSEN.format(stemp=stemp,
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
            cmds.append(
                lmpfix.PRESS_BERENDSEN.format(spress=spress,
                                              press=press,
                                              pdamp=self.pdamp,
                                              modulus=modulus))
            cmds.append(
                lmpfix.TEMP_BERENDSEN.format(stemp=stemp,
                                             temp=temp,
                                             tdamp=self.tdamp))
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
        self.append(self.FIX_NVE + self.RUN_STEP % nstep + self.UNFIX)


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

    def average(self, modulus=10, xyz='xyz', record_num=10):
        """
        Average the boundary based on NPT ensemble.

        :param modulus float: the modules in berendsen barostat.
        :param xyz str: the filename to record boundary.
        :param record_num int: the number of records.
        """
        # Record the xyz span durning NPT
        spans = [f'{i}l' for i in xyz]
        for name, dim in zip(spans, xyz):
            self.variable(name, 'equal', f'"{dim}hi - {dim}lo"')
        num = int(self.relax_step / record_num)
        record = ' '.join(f'v_{i}' for i in spans)
        record = f"fix %s all ave/time 1 {num} {num} {record} file {xyz}"
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
            self.variable(name, 'python', func)
            self.append(f'python {func} input 1 {xyz} return v_{name} format '
                        f'sf here "from nemd.lmpfunc import {func}"')
        self.print(*aves, label='Averaged')
        # Calculate the ratio and change the box
        ratios = [f'ratio_{i}' for i in xyz]
        for ratio, ave, span in zip(ratios, aves, spans):
            self.variable(ratio, 'equal', f'"v_{ave} / v_{span}"')
        scales = ' '.join(f'{i} scale ${{{r}}}' for i, r in zip(xyz, ratios))
        self.append(f"change_box all {scales} remap")
        # Delete used variables
        for name in spans + aves + ratios:
            self.variable(name, 'delete')

    def variable(self, *args):
        """
        Define a lammps variable.
        """
        self.append(f'variable {" ".join(args)}')

    def print(self, *args, label=None):
        """
        Print variables with header.

        :param args tuple of str: the variables to be printed.
        :param label str: the label of the variables.
        """
        to_print = ' '.join(f'{i}=${{{i}}}' for i in args)
        if label:
            to_print = f'{label}: {to_print}'
        self.append(f'print "{to_print}"')


class Script(Ave):
    """
    Customized with relaxation cycles.
    """
    MODULUS_VAR = f'${{{lmpfix.MODULUS}}}'
    DUMP_EVERY = f"{DUMP_MODIFY} {{id}} every {{arg}}\n"

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
        self.npt(nstep=self.relax_step / 1E1,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 spress=f'${{{lmpfix.PRESS}}}',
                 press=self.options.press,
                 modulus=self.MODULUS_VAR)

    def cycle(self,
              max_loop=1000,
              num=3,
              record_num=100,
              defm_id='defm_id',
              defm_start='defm_start',
              defm_break='defm_break'):
        """
        Deform the box by cycles to get close to the target pressure.
        One cycle: sinusoidal wave, printing, deformation, and relaxation.

        :param max_loop int: the maximum number of big cycle loops.
        :param num int: the number of sinusoidal waves in each cycle.
        :param record_num int: each sinusoidal wave records this number of data.
        :param defm_id str: deformation id loop from 0 to max_loop - 1
        :param defm_start str: label to start the deformation loop
        :param defm_break str: terminated deformation goes to this label
        """
        nstep = int(self.relax_step * 10 / max_loop / (num + 1))
        # One sinusoidal cycle that yields at least 10 records
        nstep = max([int(nstep / record_num), 10]) * record_num
        # Maximum Total Cycle Steps (cyc_nstep): self.relax_steps * 10
        # Each cycle dumps one trajectory frame
        self.append(
            self.DUMP_EVERY.format(id=self.DUMP_ID, arg=nstep * (num + 1)))
        # Set variables used in the loop
        self.append(f"variable {lmpfix.VOL} equal {lmpfix.VOL}")
        self.append(f"variable {lmpfix.AMP} equal 0.01*v_{lmpfix.VOL}^(1/3)")
        self.append(lmpfix.SET_IMMED_PRESS)
        self.append(lmpfix.SET_IMMED_MODULUS.format(record_num=record_num))
        self.append(lmpfix.SET_FACTOR.format(press=self.options.press))
        self.append(lmpfix.SET_LOOP.format(id=defm_id, end=max_loop - 1))
        self.append(lmpfix.SET_LABEL.format(label=defm_start))
        self.append(lmpfix.PRINT.format(var=defm_id))
        # Run in a subdirectory as some output files are of the same names
        dirname = f"defm_${{{defm_id}}}"
        self.append(lmpfix.MKDIR.format(dir=dirname))
        self.append(lmpfix.CD.format(dir=dirname))
        self.append("")
        pre = self.getCyclePre(nstep, record_num=record_num)
        self.nvt(nstep=nstep * num,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 pre=pre)
        self.append(lmpfix.PRINT.format(var=lmpfix.IMMED_PRESS))
        self.append(lmpfix.PRINT.format(var=lmpfix.IMMED_MODULUS))
        self.append(lmpfix.PRINT.format(var=lmpfix.FACTOR))
        cond = f"${{{defm_id}}} == {max_loop - 1} || ${{{lmpfix.FACTOR}}} == 1"
        self.append(lmpfix.IF_JUMP.format(cond=cond, label=defm_break))
        self.append("")
        self.nvt(nstep=nstep / 2,
                 stemp=self.options.temp,
                 temp=self.options.temp,
                 pre=[lmpfix.FIX_DEFORM])
        self.nvt(nstep=nstep / 2,
                 stemp=self.options.temp,
                 temp=self.options.temp)
        self.append(lmpfix.CD.format(dir=os.pardir))
        self.append(lmpfix.NEXT.format(id=defm_id))
        self.append(lmpfix.JUMP.format(label=defm_start))
        self.append("")
        self.append(lmpfix.SET_LABEL.format(label=defm_break))
        # Record press and modulus as immediate variable evaluation uses files
        self.append(lmpfix.SET_MODULUS)
        self.append(lmpfix.SET_PRESS)
        self.append(lmpfix.CD.format(dir=os.pardir))
        # Delete variables used in the loop
        self.append(lmpfix.DEL_VAR.format(var=lmpfix.VOL))
        self.append(lmpfix.DEL_VAR.format(var=lmpfix.AMP))
        self.append(lmpfix.DEL_VAR.format(var=lmpfix.IMMED_PRESS))
        self.append(lmpfix.DEL_VAR.format(var=lmpfix.IMMED_MODULUS))
        self.append(lmpfix.DEL_VAR.format(var=lmpfix.FACTOR))
        self.append(lmpfix.DEL_VAR.format(var=defm_id))
        # Restore dump defaults
        cmd = self.DUMP_EVERY.format(id=self.DUMP_ID, arg=self.DUMP_Q)
        self.append('\n' + cmd)

    def getCyclePre(self, nstep, record_num=100):
        """
        Get the pre-stage str for the cycle simulation.

        :param nstep int: the simulation steps of the one cycles
        :param record_num int: each cycle records this number of data
        :return list: the prefix string of the cycle stage.
        """
        wiggle = lmpfix.WIGGLE_VOL.format(period=nstep * self.options.timestep)
        record_period = int(nstep / record_num)
        record_press = lmpfix.RECORD_PRESS_VOL.format(period=record_period)
        return [record_press, wiggle]

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
