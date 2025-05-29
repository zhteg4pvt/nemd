# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module provides non-numerical symbols and hardcoded numerical constants.
"""
STAR = '*'
CARBON = 'C'
UNKNOWN = 'X'
HYDROGEN = 'H'
NITROGEN = 'N'
OXYGEN = 'O'
POUND = '#'
POUND_SEP = f'_{POUND}_'
BACKSLASH = '\\'
FORWARDSLASH = '/'
PIPE = '|'
RETURN = '\n'
RC_BRACKET = '}'
DOUBLE_QUOTATION = '"'
COLON = ':'
SEMICOLON = ';'
COMMA = ','
COMMA_SEP = f'{COMMA} '
PERIOD = '.'
AND = '&'
OFF = "off"
PLUS_MIN = '\u00B1'
ELEMENT_OF = '\u2208'
ANGSTROM = '\u212B'
XYZ = 'xyz'
XYZU = [f'{d}u' for d in XYZ]
SPC = 'SPC'
SPCE = 'SPCE'
TIP3P = 'TIP3P'
WMODELS = [SPC, SPCE, TIP3P]
WATER_DSC = 'Water ({model})'
WATER_SPC = WATER_DSC.format(model=SPC)
WATER_SPCE = WATER_DSC.format(model=SPCE)
WATER_TIP3P = WATER_DSC.format(model=TIP3P)
OPLSUA = 'OPLSUA'
OPLSUA_TIP3P = [OPLSUA, TIP3P]
SW = 'SW'
FF_NAMES = [OPLSUA, SW]
TYPE_ID = 'type_id'
RES_NUM = 'res_num'
IMPLICIT_H = 'implicit_h'
SPACE = ' '
TIME = 'Time'
TIME_LB = f'{TIME} (ps)'
ID = 'id'
NONE = 'none'
NAME = 'name'
FILE = 'file'
FILENAME = f'{FILE}{NAME}'
MSG = 'msg'
LMP = 'lmp'
OPTIMIZE = 'optimize'
SUGGEST = 'suggest'
PHONONS = 'phonons'
XML_EXT = '.xml'
DFSET_EXT = '.dfset'
LOG_EXT = '.log'
LMP_LOG = f'lmp{LOG_EXT}'
DARWIN = 'darwin'
LINUX = 'linux'
TASK = 'task'
AGGREGATOR = 'aggregator'
ARGS = 'args'
ALL = 'all'
FLOAT_FMT = '%.4f'
JSON_EXT = '.json'
# Hardcoded Numerical Constants
# A 32-bit integer limit allows for 4,294,967,296 ( 2**32 ) pieces of data.
# For signed integers, this would range from -2,147,483,648 to 2,147,483,647.
MAX_INT32 = 2**31 - 1
DEFAULT_CUT = 11.
