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
import os
import pathlib
import shutil

import flow

from nemd import envutils
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

    def openJobs(self):
        """
        See parent.
        """
        for name in self.names:
            dirname = self.options.dirname / name
            state = {FLAG_DIRNAME: name if self.options.copy else str(dirname)}
            job = self.proj.open_job(state).init()
            self.jobs.append(job)
            if self.options.copy:
                shutil.copytree(dirname, os.path.join(job.path, name))

    @functools.cached_property
    def names(self):
        """
        Return the selected test names.

        :return list: the selected test names.
        """
        dirs = [self.options.dirname / x for x in self.options.id]
        if not dirs:
            dirs = self.options.dirname.glob('[0-9]' * 4)
        return [
            x.name for x in dirs if test.Tag(x, options=self.options).selected
        ]

    def logStatus(self):
        """
        Log the number of the succeed check jobs.
        """
        super().logStatus()
        # cmd: True; check: True | failure message
        status = [y is True for x in self.status.values() for y in x.values()]
        self.log(f"{sum(status)} / {len(status)} succeed sub-jobs.")

    def setAggs(self):
        """
        Set the aggregator operators.
        """

        def select(job, names=self.names):
            """
            Select the jobs.

            :param job 'signac.job.Job': one found signac job.
            :param names list: the selected dirnames. (RuntimeError of cannot
                pickle '_thread.RLock' object without explicitly passing)
            """
            return any(job.sp[FLAG_DIRNAME].endswith(x) for x in names)

        agg = flow.aggregator(select=select) if self.options.id else None
        self.add(task.TestAgg, aggregator=agg)

    def findJobs(self, filter=None):
        """
        See parent.
        """
        if self.options.name:
            filter = {FLAG_DIRNAME: {"$regex": f".*({'|'.join(self.names)})$"}}
        super().findJobs(filter=filter)


class TestValid(parserutils.Valid):
    """
    Customized for dirname.
    """

    def run(self):
        """
        See parent.
        """
        self.dirname()
        self.id()

    def dirname(self):
        """
        Validate dirname.

        :raises FileNotFoundError: if the input directory cannot be located.
        """
        if self.options.dirname is None:
            dirname = envutils.get_src('test', self.options.name)
            if dirname:
                self.options.dirname = pathlib.Path(dirname)
        if self.options.dirname and self.options.dirname.is_dir():
            return
        raise FileNotFoundError(f'Cannot locate the tests ({FLAG_DIRNAME}).')

    def id(self):
        """
        Validate test ids.

        :raises FileNotFoundError: no tests found.
        """
        if not self.options.id:
            return
        dirs = [self.options.dirname / f"{x:0>4}" for x in self.options.id]
        found = [x for x in dirs if x.is_dir()]
        if not found:
            raise FileNotFoundError(f"{', '.join(map(str, dirs))} not found.")
        self.options.id = [x.name for x in found]


class Parser(parserutils.Workflow):
    """
    Customized for tests.
    """
    WFLAGS = parserutils.Workflow.WFLAGS[1:]
    INTEGRATION = 'integration'
    SCIENTIFIC = 'scientific'
    PERFORMANCE = task.Cmd.PERFORMANCE
    TESTS = [INTEGRATION, SCIENTIFIC, PERFORMANCE]

    def setUp(self):
        """
        See parent.
        """
        self.add_argument('id',
                          metavar='INT',
                          type=parserutils.type_positive_int,
                          nargs='*',
                          help='Select the tests by ids.')
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
        self.add_argument('-copy',
                          action='store_true',
                          help='Copy test data into the working directory.')
        self.valids.add(TestValid)


if __name__ == "__main__":
    logutils.Script.run(Runner, Parser(descr=__doc__))
