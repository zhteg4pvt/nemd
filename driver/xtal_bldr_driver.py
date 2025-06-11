# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver builds a crystal.
"""
import sys

from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import stillinger
from nemd import xtal


class Crystal(logutils.Base):
    """
    This class builds a crystal.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver': Parsed command-line options
        """
        super().__init__(**kwargs)
        self.options = options
        self.struct = None

    def run(self):
        """
        Main method to run.
        """
        crystal = xtal.Crystal.fromDatabase(options=self.options)
        self.struct = stillinger.Struct.fromMols([crystal.mol],
                                                 options=self.options)
        self.struct.write()
        self.log(f"LAMMPS data file written as {self.struct.datafile}")
        jobutils.Job.reg(self.struct.datafile)
        script = stillinger.In(self.struct)
        script.write()
        self.log(f"LAMMPS input script written as {script.inscript}")
        jobutils.Job.reg(script.inscript, file=True)


def main(argv):
    parser = parserutils.XtalBldr(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options) as logger:
        crystal = Crystal(options, logger=logger)
        crystal.run()


if __name__ == "__main__":
    main(sys.argv[1:])
