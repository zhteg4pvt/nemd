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
from nemd import symbols
from nemd import task


class Log(logutils.Base):
    """
    Main class to analyze a lammps log.
    """

    def __init__(self, options, logger=None):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        :param logger 'logging.Logger': the logger to log messages
        """
        super().__init__(logger=logger)
        self.options = options
        self.thermo = None
        self.df_reader = None
        self.tasks = [x for x in analyzer.THERMO if x in self.options.task]

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
            self.log_error(f"No thermo output found in {self.options.log}.")
        self.log(f"{self.thermo.shape[0]} steps of thermo data found.")
        self.log(f"{', '.join(self.options.task)} averages results from last "
                 f"{self.options.last_pct * 100}% frames {symbols.ELEMENT_OF} "
                 f"[{', '.join([f'{x:.3f}' for x in self.thermo.range])}] ps")

    def setTasks(self):
        """
        Set the tasks to be performed.
        """
        columns = self.thermo.columns
        available = [analyzer.Base.parse(x)[0].lower() for x in columns]
        selected = set(self.tasks).intersection(available)
        if len(selected) == len(self.tasks):
            return
        missed = symbols.COMMA_SEP.join(set(self.tasks).difference(selected))
        available = symbols.COMMA_SEP.join(available)
        self.log_warning(f"{missed} tasks cannot be found out of {available}.")
        self.tasks = list(selected)

    def analyze(self):
        """
        Run analyzers.
        """
        for task in self.tasks:
            anl = analyzer.ANLZ[task](self.thermo,
                                      options=self.options,
                                      logger=self.logger,
                                      df_reader=self.df_reader)
            anl.run()


def main(argv):
    parser = task.LogJob.get_parser()
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        log = Log(options, logger=logger)
        log.run()


if __name__ == "__main__":
    main(sys.argv[1:])
