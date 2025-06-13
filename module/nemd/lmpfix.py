# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
lammps commands: fix, unfix, run, variable, dump, etc.
"""
RUN_STEP = "run %i"
UNFIX = "unfix %s"

TEMP_BERENDSEN = f"fix %s all temp/berendsen {{stemp}} {{temp}} {{tdamp}}"

MODULUS = 'modulus'
PRESS_BERENDSEN = f"fix %s all press/berendsen iso {{spress}} {{press}} {{pdamp}} {MODULUS} {{modulus}}"

VOL = 'vol'
PRESS_VOL_FILE = 'press_vol.data'
RECORD_PRESS_VOL = f"fix %s all ave/time 1 {{period}} {{period}} c_thermo_press v_vol file {PRESS_VOL_FILE}"

IMMED_MODULUS = 'immed_modulus'
SET_IMMED_MODULUS = f"""variable {IMMED_MODULUS} python getModulus
python getModulus input 2 {PRESS_VOL_FILE} {{record_num}} return v_{IMMED_MODULUS} format sif here "from nemd.lmpfunc import getModulus"
"""
SET_MODULUS = f"variable {MODULUS} equal ${{{IMMED_MODULUS}}}"

IMMED_PRESS = 'immed_press'
SET_IMMED_PRESS = f"""variable {IMMED_PRESS} python getPress
python getPress input 1 {PRESS_VOL_FILE} return v_{IMMED_PRESS} format sf here "from nemd.lmpfunc import getPress"
"""
PRESS = 'press'
SET_PRESS = f"variable {PRESS} equal ${{{IMMED_PRESS}}}"

FACTOR = 'factor'
SET_FACTOR = f"""variable {FACTOR} python getBdryFactor
python getBdryFactor input 2 {{press}} press_vol.data return v_{FACTOR} format fsf here "from nemd.lmpfunc import getBdryFactor"
"""

FIX_DEFORM = f"fix %s all 'deform' 100 x scale ${{{FACTOR}}} y scale ${{{FACTOR}}} z scale ${{{FACTOR}}} remap v"

AMP = 'amp'
WIGGLE_DIM = "%s wiggle ${{amp}} {period}"
PARAM = ' '.join([WIGGLE_DIM % dim for dim in ['x', 'y', 'z']])
WIGGLE_VOL = f"fix %s all 'deform' 100 {PARAM}"

DEL_VAR = "variable {var} delete"

SET_LABEL = "label {label}"
MKDIR = "shell mkdir {dir}"
CD = "shell cd {dir}"
JUMP = "jump SELF {label}"
IF_JUMP = f'if "{{cond}}" then "{JUMP}"'
PRINT = 'print "{var} = ${{{var}}}"'
NEXT = "next {id}"
