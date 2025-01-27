# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This workflow runs integration and performance tests.

The sub-folder name must be one integer, and contain one cmd file.
Each integration test contains a check file to verify the expected outputs.
Each performance test may contain a param file to parameterize the command.

Supported check commands are: cmp, exist, not_exist, in ..
Supported tag commands are: slow, label
"""
import glob
import os
import sys

from nemd import jobcontrol
from nemd import logutils
from nemd import parserutils
from nemd import task
from nemd import test


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
            pre_after.append(self.setOpr(task.CmdTask))
        if test.CHECK in self.options.task:
            pre_after.append(self.setOpr(task.CheckTask))
        if test.TAG in self.options.task:
            pre_after.append(self.setOpr(task.TagTask))
        for pre, after in zip(pre_after[:-1], pre_after[1:]):
            self.setPreAfter(pre, after)

    def setState(self):
        """
        Set state with test dirs.
        """
        if self.options.id:
            ids = [f"{x:0>4}" for x in self.options.id]
            dirs = [os.path.join(self.options.dir, x) for x in ids]
            dirs = [x for x in dirs if os.path.isdir(x)]
        else:
            dirs = glob.glob(os.path.join(self.options.dir, '[0-9]' * 4))

        if not dirs:
            self.log_error(f'No valid tests found in {self.options.dir}.')

        dirs = [
            x for x in dirs if test.Tag(x, options=self.options).selected()
        ]
        if not dirs:
            self.log_error('All tests are skipped according to the tag file.')
        self.state = {parserutils.FLAG_DIR: dirs}

    def setAggJobs(self):
        """
        Report the task timing after filtering.
        """
        self.setAgg(task.TestTask)


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
