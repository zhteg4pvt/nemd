# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
Shared numerical constants.
"""
import numpy as np
from scipy import constants

EYE4 = np.eye(4)
ANG_TO_CM = constants.angstrom / constants.centi
NANO_TO_FEMTO = constants.nano / constants.femto
PICO_TO_FEMTO = constants.pico / constants.femto
CM_INV_THZ = constants.physical_constants['inverse meter-hertz relationship'][
    0] / constants.tera / constants.centi
ANG_TO_BOHR = constants.angstrom / constants.physical_constants['Bohr radius'][
    0]
