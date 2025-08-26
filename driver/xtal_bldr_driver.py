# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver builds a crystal.
"""
from nemd import jobutils
from nemd import lmpatomic
from nemd import logutils
from nemd import parserutils
from nemd import xtal


class Crystal(logutils.Base):
    """
    This class builds a crystal.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver':  Parsed command-line options
        """
        super().__init__(options=options, **kwargs)
        self.struct = None

    def run(self):
        """
        Main method to run.
        """
        crystal = xtal.Crystal.fromDatabase(options=self.options)
        self.struct = lmpatomic.Struct.fromMols([crystal.mol],
                                                options=self.options)
        self.struct.write()
        self.log(f'Data file written into {self.struct.outfile}')
        jobutils.Job.reg(self.struct.outfile)
        self.struct.script.write()
        self.log(f'In script written into {self.struct.script.outfile}')
        jobutils.Job.reg(self.struct.script.outfile, file=True)


if __name__ == "__main__":
    logutils.Script.run(Crystal, parserutils.XtalBldr(descr=__doc__))
