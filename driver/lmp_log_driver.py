# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver analyzes the lammps log file.
"""
import sys

from nemd import analyzer
from nemd import lmpfull
from nemd import lmplog
from nemd import logutils
from nemd import parserutils
from nemd import symbols


class LmpLog(logutils.Base):
    """
    Main class to analyze a lammps log.
    """

    def __init__(self, options, **kwargs):
        """
        :param options 'argparse.Driver': Parsed command-line options
        """
        super().__init__(options=options, **kwargs)
        self.thermo = None
        self.rdr = None
        self.task = None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.setThermo()
        self.setTasks()
        self.analyze()

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return
        self.rdr = lmpfull.Reader.read(self.options.data_file)

    def setThermo(self):
        """
        Grep thermo output information.
        """
        self.thermo = lmplog.Log(self.options.log, options=self.options).thermo
        if self.thermo.empty:
            self.error(f"No thermo output found in {self.options.log}.")
        self.log(f"{self.thermo.shape[0]} steps of thermo data found.")
        self.log(f"Averages results from {self.thermo.range[0]:.3f} ps to "
                 f"{self.thermo.range[1]:.3f} ps")

    def setTasks(self, tasks=tuple(x.name for x in analyzer.THERMO)):
        """
        Set the analyzer tasks.

        :param tasks tuple: supported tasks.
        """
        parsed = [analyzer.Job.parse(x) for x in self.thermo.columns]
        avail = set([name.lower() for name, unit, _ in parsed])
        self.task = avail.intersection(
            tasks if symbols.ALL in self.options.task else self.options.task)
        if not self.task:
            self.error(f"No tasks found. Please select from {avail}.")
        missed = set(self.options.task).difference(self.task)
        missed.discard(symbols.ALL)
        if not missed:
            return
        self.warning(f"Tasks {missed} cannot be found out of {avail}.")

    def analyze(self):
        """
        Run analyzers.
        """
        for Anlz in analyzer.THERMO:
            if Anlz.name not in self.task:
                continue
            anl = Anlz(self.thermo,
                       options=self.options,
                       logger=self.logger,
                       rdr=self.rdr)
            anl.run()


def main(argv):
    parser = parserutils.LmpLog(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        log = LmpLog(options, logger=logger)
        log.run()


if __name__ == "__main__":
    main(sys.argv[1:])
