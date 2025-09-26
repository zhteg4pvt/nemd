# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
Shared numerical constants.
"""
from scipy import constants

ANG_TO_CM = constants.angstrom / constants.centi
NANO_TO_FEMTO = constants.nano / constants.femto
FEMTO_TO_PICO = constants.femto / constants.pico
CM_INV_THZ = constants.physical_constants['inverse meter-hertz relationship'][
    0] / constants.tera / constants.centi
ANG_TO_BOHR = constants.angstrom / constants.physical_constants['Bohr radius'][
    0]
MB_TO_GB = constants.mebi / constants.gibi