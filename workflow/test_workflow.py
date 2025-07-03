# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
Run integration, scientific, or performance tests.

Each one-integer sub-folder contain one cmd file.
Each integration test contains a check file to verify the expected outputs.
Each performance test may contain a param file to parameterize the command.

Supported check commands: exist, glob, has, cmp, and collect
Supported tag commands: slow, label
"""
import functools
import glob
import os
import sys

from nemd import envutils
from nemd import flow
from nemd import jobcontrol
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import task
from nemd import test

FLAG_DIRNAME = jobutils.FLAG_DIRNAME


class Runner(jobcontrol.Runner):
    """
    The main class to run integration and performance tests.
    """

    def setJobs(self, pre=False):
        """
        Set operators to run cmds, check results, and tag tests.

        :param pre bool: the pre task.
        """
        if test.Cmd.name in self.options.task:
            pre = self.add(task.Cmd)
        if test.Check.name in self.options.task:
            self.add(task.Check, pre=pre)
        if test.Tag.name in self.options.task:
            self.add(task.Tag, pre=pre)

    def setState(self):
        """
        Set state with test dirs.
        """
        self.state = {FLAG_DIRNAME: self.dirs}

    @functools.cached_property
    def dirs(self):
        """
        Return the test directories.

        :return list: the selected test directories.
        """
        if self.options.id:
            ids = [f"{x:0>4}" for x in self.options.id]
            dirs = [os.path.join(self.options.dirname, x) for x in ids]
            dirs = [x for x in dirs if os.path.isdir(x)]
        else:
            dirs = glob.glob(os.path.join(self.options.dirname, '[0-9]' * 4))

        if not dirs:
            self.error(f'No valid tests found in {self.options.dirname}.')
        dirs = [x for x in dirs if test.Tag(x, options=self.options).selected]
        if not dirs:
            self.error('All tests are skipped according to the tag files.')

        return dirs

    def logStatus(self):
        """
        Log the number of the succeed check jobs.
        """
        super().logStatus()
        self.log(f"{sum([x is True for x in self.status.values()])} / "
                 f"{len(self.status)} succeed sub-jobs.")

    def setAggs(self):
        """
        Set the aggregator operators.
        """

        def select(job, dirs=self.dirs):
            """
            Select the jobs.

            :param job 'signac.job.Job': one found signac job.
            :param dirs list: the selected dirnames. (RuntimeError of cannot
                pickle '_thread.RLock' object without explicitly passing)
            """
            return job.sp[FLAG_DIRNAME] in dirs

        agg = flow.aggregator(select=select) if self.options.id else None
        self.add(task.TestAgg, aggregator=agg)

    def setAggProj(self, filter=None):
        """
        Customized to filter dirnames. (see parent)
        """
        if self.options.id:
            filter = {"$or": [{FLAG_DIRNAME: x} for x in self.dirs]}
        super().setAggProj(filter=filter)


class TestValid(parserutils.Valid):
    """
    Customized for dirname.
    """

    def run(self):
        """
        Main method to run the validation.

        :raises FileNotFoundError: if the input directory cannot be located.
        """
        if self.options.dirname is None:
            self.options.dirname = envutils.get_src('test', self.options.name)
        if not self.options.dirname:
            raise FileNotFoundError(
                f'Cannot locate the test dir ({FLAG_DIRNAME}).')


class Parser(parserutils.Workflow):
    """
    Customized for tests.
    """
    WFLAGS = parserutils.Workflow.WFLAGS[1:]
    INTEGRATION = 'integration'
    SCIENTIFIC = 'scientific'
    PERFORMANCE = 'performance'
    TESTS = [INTEGRATION, SCIENTIFIC, PERFORMANCE]

    def setUp(self):
        """
        See parent.
        """
        self.add_argument('id',
                          metavar='INT',
                          type=parserutils.type_positive_int,
                          nargs='*',
                          help='Select the tests according to these ids.')
        self.add_argument(parserutils.FLAG_NAME,
                          default=self.INTEGRATION,
                          choices=self.TESTS,
                          help=f'{self.INTEGRATION}: reproducible; '
                          f'{self.SCIENTIFIC}: physical meaningful; '
                          f'{self.PERFORMANCE}: resource efficient.')
        self.add_argument(FLAG_DIRNAME,
                          type=parserutils.type_dir,
                          help='Search test(s) under this directory.')
        self.add_argument(
            '-slow',
            type=parserutils.type_positive_float,
            metavar='SECOND',
            help='Skip (sub)tests marked with time longer than this criteria.')
        self.add_argument('-label',
                          nargs='+',
                          help='Select the tests marked with the given label.')
        self.add_argument(jobutils.FLAG_TASK,
                          nargs='+',
                          choices=[x.name for x in test.TASKS],
                          default=[test.Cmd.name, test.Check.name],
                          help='cmd: run the commands in cmd file; '
                          'check: check the results; tag: update the tag file')
        self.valids.add(TestValid)


def main(argv):
    parser = Parser(descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        obj = Runner(options, argv, logger=logger)
        obj.run()


if __name__ == "__main__":
    main(sys.argv[1:])
