# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver analyzes the log file from previous molecular dynamics simulations.
"""
import sys

from nemd import analyzer
from nemd import lammpsdata
from nemd import lmplog
from nemd import logutils
from nemd import parserutils
from nemd import symbols


class Log(logutils.Base):
    """
    Main class to analyze a lammps log.
    """

    def __init__(self, options, logger=None):
        """
        :param options 'argparse.Driver': Parsed command-line options
        :param logger 'logging.Logger': the logger to log messages
        """
        super().__init__(logger=logger)
        self.options = options
        self.thermo = None
        self.df_reader = None

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
        self.df_reader = lammpsdata.read(self.options.data_file)

    def setThermo(self):
        """
        Grep thermo output information.
        """
        lmp_log = lmplog.Log(self.options.log, options=self.options)
        lmp_log.run()
        self.thermo = lmp_log.thermo
        if self.thermo.empty:
            self.error(f"No thermo output found in {self.options.log}.")
        self.log(f"{self.thermo.shape[0]} steps of thermo data found.")
        self.log(f"Averages results from {self.thermo.range[0]:.3f} ps to "
                 f"{self.thermo.range[1]:.3f} ps")

    def setTasks(self):
        """
        Set the tasks to be performed.
        """
        parsed = [analyzer.Base.parse(x) for x in self.thermo.columns]
        avail = [name.lower() for name, unit, _ in parsed]
        if symbols.ALL in self.options.task:
            self.options.task = [x for x in avail if x in analyzer.THERMO]
            return
        tasks = set(self.options.task)
        self.options.task = tasks.intersection(avail)
        missed = symbols.COMMA_SEP.join(tasks.difference(self.options.task))
        self.warning(f"{missed} tasks cannot be found out of "
                     f"{symbols.COMMA_SEP.join(avail)}.")

    def analyze(self):
        """
        Run analyzers.
        """
        for task in self.options.task:
            anl = analyzer.ANLZ[task](self.thermo,
                                      options=self.options,
                                      logger=self.logger,
                                      df_reader=self.df_reader)
            anl.run()


def main(argv):
    parser = parserutils.Log(__file__, descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        log = Log(options, logger=logger)
        log.run()


if __name__ == "__main__":
    main(sys.argv[1:])
