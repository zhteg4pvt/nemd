# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This workflow runs integration and performance tests.

The sub-folder name must be one integer, and contain one cmd file.
Each integration test contains a check file to verify the expected outputs.
Each performance test may contain a param file to parameterize the command.

Supported check commands are: cmp, exist, not_exist, in ...
Supported tag commands are: slow, label
"""
import functools
import glob
import os
import sys

from nemd import envutils
from nemd import jobcontrol
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import task
from nemd import test

FLAG_DIR = jobutils.FLAG_DIR


class Test(jobcontrol.Runner):
    """
    The main class to run integration and performance tests.
    """
    AggClass = task.TestAgg

    def setJobs(self):
        """
        Set operators to run cmds, check results, and tag tests.
        """
        cmd = self.add(task.Cmd) if test.CMD in self.options.task else False
        if test.CHECK in self.options.task:
            self.add(task.Check, pre=cmd)
        if test.TAG in self.options.task:
            self.add(task.Tag, pre=cmd)

    def setState(self):
        """
        Set state with test dirs.
        """
        self.state = {FLAG_DIR: self.getDirs()}

    @functools.cache
    def getDirs(self):
        """
        Get the dirs of the tests.

        :return list: each dir contains one test.
        """
        if self.options.id:
            ids = [f"{x:0>4}" for x in self.options.id]
            dirs = [os.path.join(self.options.dir, x) for x in ids]
            dirs = [x for x in dirs if os.path.isdir(x)]
        else:
            dirs = glob.glob(os.path.join(self.options.dir, '[0-9]' * 4))

        if not dirs:
            self.error(f'No valid tests found in {self.options.dir}.')

        if any([self.options.slow, self.options.label]):
            dirs = [
                x for x in dirs
                if test.Tag(x, options=self.options).selected()
            ]
        if not dirs:
            self.error('All tests are skipped according to the tag file.')
        return dirs

    def logStatus(self):
        """
        Log the number of the succeed check jobs.
        """
        super().logStatus()
        msgs = [x for x in self.status.values() if x is not True]
        total = len(self.status)
        self.log(f"{total - len(msgs)} / {total} succeed jobs.")

    def setAggProj(self):
        """
        Report the task timing after filtering.
        """
        flag_dirs = [{FLAG_DIR: x} for x in self.getDirs()]
        super().setAggProj(filter={"$or": flag_dirs})


class TestValid(parserutils.Valid):
    """
    Class to validate the input options.
    """

    def run(self):
        """
        Main method to run the validation.

        :raises ValueError: if the input directory is None.
        """
        if self.options.dir is None:
            self.options.dir = envutils.get_nemd_src('test', self.options.name)
        if not self.options.dir:
            raise ValueError(f'Please define the test dir via {FLAG_DIR}')


class Parser(parserutils.Workflow):
    """
    A customized parser that supports cross argument validation options.
    """
    WFLAGS = parserutils.Workflow.WFLAGS[1:]
    FLAG_ID = 'id'
    FLAG_TASK = jobutils.FLAG_TASK
    FLAG_LABEL = '-label'
    CMD = 'cmd'
    CHECK = 'check'
    TAG = 'tag'

    INTEGRATION = 'integration'
    SCIENTIFIC = 'scientific'
    PERFORMANCE = 'performance'
    NAMES = [INTEGRATION, SCIENTIFIC, PERFORMANCE]
    TASKS = [CMD, CHECK, TAG]

    def setUp(self):
        """
        Add test related flags.
        """
        self.add_argument(self.FLAG_ID,
                          metavar=self.FLAG_ID.upper(),
                          type=parserutils.type_positive_int,
                          nargs='*',
                          help='Select the tests according to these ids.')
        self.add_argument(jobutils.FlAG_NAME,
                          default=self.INTEGRATION,
                          choices=self.NAMES,
                          help=f'{self.INTEGRATION}: reproducible; '
                          f'{self.SCIENTIFIC}: physical meaningful; '
                          f'{self.PERFORMANCE}: resource efficient.')
        self.add_argument(FLAG_DIR,
                          metavar=FLAG_DIR[1:].upper(),
                          type=parserutils.type_dir,
                          help='Search test(s) under this directory.')
        self.add_argument(
            jobutils.FLAG_SLOW,
            type=parserutils.type_positive_float,
            metavar='SECOND',
            help='Skip (sub)tests marked with time longer than this criteria.')
        self.add_argument(self.FLAG_LABEL,
                          nargs='+',
                          metavar='LABEL',
                          help='Select the tests marked with the given label.')
        self.add_argument(self.FLAG_TASK,
                          nargs='+',
                          choices=self.TASKS,
                          default=[self.CMD, self.CHECK],
                          help='cmd: run the commands in cmd file; '
                          'check: check the results; tag: update the tag file')
        self.valids.add(TestValid)
        self.suppress([parserutils.Workflow.FLAG_SCREEN])


def main(argv):
    parser = Parser(__file__, descr=__doc__)
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        obj = Test(options, argv, logger=logger)
        obj.run()


if __name__ == "__main__":
    main(sys.argv[1:])
