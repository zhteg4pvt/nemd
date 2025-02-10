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

from nemd import jobcontrol
from nemd import logutils
from nemd import parserutils
from nemd import task
from nemd import taskbase
from nemd import test


class AggJob(taskbase.AggJob):

    def run(self):
        """
        Filter jobs by ids before collecting time.
        """
        if self.options.id:
            dirs = [x.statepoint[parserutils.FLAG_DIR] for x in self.jobs]
            ids = map(int, [os.path.basename(x) for x in dirs])
            self.jobs = [
                x for x, y in zip(self.jobs, ids) if y in self.options.id
            ]
        super().run()


class Task(taskbase.Task):

    AggClass = AggJob


class Test(jobcontrol.Runner, logutils.Base):
    """
    The main class to run integration and performance tests.
    """

    def setJob(self):
        """
        Set operators to run cmds, check results, and tag tests.

        FIXME: tag and check cannot be paralleled due to the same job json file
          being touched by multiple operators.
        """
        pre_after = []
        if test.CMD in self.options.task:
            pre_after.append(self.setOpr(task.Cmd))
        if test.CHECK in self.options.task:
            pre_after.append(self.setOpr(task.Check))
        if test.TAG in self.options.task:
            pre_after.append(self.setOpr(task.Tag))
        for pre, after in zip(pre_after[:-1], pre_after[1:]):
            self.setPreAfter(pre, after)

    def setState(self):
        """
        Set state with test dirs.
        """
        self.state = {parserutils.FLAG_DIR: self.getDirs()}

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
            self.log_error(f'No valid tests found in {self.options.dir}.')

        if any([self.options.slow, self.options.label]):
            dirs = [
                x for x in dirs
                if test.Tag(x, options=self.options).selected()
            ]
        if not dirs:
            self.log_error('All tests are skipped according to the tag file.')
        return dirs

    def setAggJobs(self):
        """
        Register the aggregator to collect the time of the select jobs.
        """
        super().setAggJobs(TaskClass=Task)

    def cleanAggJobs(self):
        """
        Report the task timing after filtering.
        """
        flag_dirs = [{parserutils.FLAG_DIR: x} for x in self.getDirs()]
        super().cleanAggJobs(filter={"$or": flag_dirs})


class WorkflowParser(parserutils.WorkflowParser):
    """
    A customized parser that supports cross argument validation options.
    """

    WFLAGS = parserutils.WorkflowParser.WFLAGS[1:]


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'ArgumentParser': argparse figures out how to parse sys.argv.
    """
    parser = WorkflowParser(__file__, descr=__doc__)
    parser.add_test_arguments()
    parser.add_job_arguments()
    parser.add_workflow_arguments()
    return parser


def main(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    with logutils.Script(options, file=True) as logger:
        obj = Test(options, argv, logger=logger)
        obj.run()


if __name__ == "__main__":
    main(sys.argv[1:])
