# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This class handles jobs and aggregators:
    1) register operators
    2) create or load project
    3) set parametrizes
    4) distribute resources
    5) initiate jobs
    6) clean previous results
    7) execute the operators
    8) log the status and message.
"""
import collections
import itertools
import math
import os

import flow
import networkx as nx
import numpy as np
import pandas as pd

from nemd import jobutils
from nemd import logutils
from nemd import plotutils
from nemd import symbols
from nemd import task
from nemd import taskbase


class Runner(logutils.Base):
    """
    The main class to set up a workflow.
    """
    AggClass = task.TimeAgg
    COMPLETED = 'completed'
    OPERATIONS = 'operations'
    JOB_ID = 'job_id'
    PREREQ = taskbase.Agg.PREREQ
    FLAG_SEED = jobutils.FLAG_SEED
    MESSAGE = taskbase.MESSAGE

    def __init__(self, options, original, logger=None):
        """
        :param options 'argparse.Namespace': parsed commandline options
        :param original list: list of commandline arguments
        :param logger 'logging.Logger': print to this logger if exists
        """
        super().__init__(logger=logger)
        self.options = options
        self.original = original
        self.args = self.original[:]
        self.state = {}
        self.jobs = []
        self.added = []
        self.logged = set()
        self.proj = None
        self.max_cpu = 1
        self.cpu = None
        self.prereq = collections.defaultdict(list)
        # flow/project.py gets logger from logging.getLogger(__name__)
        logutils.Logger.get('flow.project')
        if self.options.CPU:
            self.max_cpu = self.options.CPU[0]
        elif not self.options.DEBUG:
            # Production mode: 75% of cpu count as total to avoid overloading
            self.max_cpu = max([round(os.cpu_count() * 0.75), 1])
        # Debug mode: 1 cpu as total to avoid parallelism

    def run(self):
        """
        The main method to run the integration tests.

        The linear pipline handles three things on request:
        1) clean previous projects
        2) run a project with task jobs
        3) run a project with aggregator jobs
        """
        if symbols.TASK in self.options.jtype:
            self.setJobs()
            self.setProj()
            self.plotJobs()
            self.setState()
            self.openJobs()
            self.setCpu()
            self.clean()
            self.runProj()
            self.logStatus()
        if symbols.AGGREGATOR in self.options.jtype:
            self.setAggs()
            self.setAggProj()
            self.clean(agg=True)
            self.runProj(agg=True)

    def setJobs(self):
        """
        Set the job operators for one parameter set.
        """
        pass

    def add(self, TaskClass, pre=None, **kwargs):
        """
        Add one operator to the project.

        :param TaskClass 'task.Task' (sub)-class: the class to get the operator
        :param pre 'types.SimpleNamespace': the prerequisite operator container
        :return 'types.SimpleNamespace': the added operator container
        """
        if pre is None:
            pres = [x for x in self.added if x.cls.agg == TaskClass.agg]
            if pres:
                pre = pres[-1]
        opr = TaskClass.getOpr(**kwargs,
                               logged=self.logged,
                               logger=self.logger,
                               options=self.options)
        self.added.append(opr)
        if pre:
            self.setPreAfter(pre, opr)
        return opr

    def setProj(self):
        """
        Initiate the project.
        """
        self.proj = flow.project.FlowProject.init_project()
        self.proj.document.update({self.PREREQ: self.prereq})
        self.proj.document[symbols.ARGS] = self.args

    def plotJobs(self):
        """
        Plot the job-dependency graph.
        """
        if not self.options.DEBUG:
            return
        with plotutils.get_pyplot(inav=self.options.INTERAC,
                                  name='workflow') as plt:
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            opr_graph = self.proj.detect_operation_graph()
            graph = nx.DiGraph(np.asarray(opr_graph))
            pos = nx.spring_layout(graph)
            names = [x for x in self.proj.operations.keys()]
            labels = {
                key: name
                for (key, name) in zip(range(len(names)), names)
            }
            nx.draw_networkx(graph, pos, ax=ax, labels=labels)
        fig.savefig(self.options.JOBNAME + '_nx.png')

    def setState(self):
        """
        Set the parameter flags and values.
        """
        try:
            seed_incre = np.arange(self.options.state_num)
        except AttributeError:
            return
        # seed from 1 as EmbedMolecule assigns the same coordinates for 0 and 1
        # mol = AllChem.MolFromSmiles("CCCC")
        # for randomSeed in [0, 1]:
        #     AllChem.EmbedMolecule(mol, randomSeed=randomSeed)
        #     print(mol.GetConformer().GetPositions())
        jobutils.pop_arg(self.args, self.FLAG_SEED)
        seed = getattr(self.options, self.FLAG_SEED[1:], 1)
        seeds = (seed_incre + seed) % symbols.MAX_INT32
        self.state = {self.FLAG_SEED: list(map(str, seeds))}

    def setCpu(self):
        """
        Set cpu numbers for the project.
        """
        if self.options.CPU is None:
            # Single cpu per job ensures efficiency
            self.cpu = [self.max_cpu, 1]
            return

        try:
            num = self.options.CPU[1]
        except IndexError:
            # Only total cpu specified: evenly distribute among subjobs
            num = math.floor(self.options.CPU[0] / len(self.jobs))
            num = max([num, 1])
        self.cpu = [math.floor(self.options.CPU[0] / num), num]

        try:
            index = self.args.index(jobutils.FLAG_CPU)
        except ValueError:
            pass
        else:
            # _StatePointDict warns NumpyConversionWarning if statepoint dict
            # contains numerical data types.
            self.args[index + 1] = str(self.cpu[1])

    def openJobs(self):
        """
        Open one job for each parameter set.
        """
        for values in itertools.product(*self.state.values()):
            # e.g. arg = (['-seed', '0'], ['-scale_factor', '0.95'])
            state = {x: y for x, y in zip(self.state.keys(), values)}
            job = self.proj.open_job(state)
            job.init()
            self.jobs.append(job)
        if not self.jobs:
            self.error('No jobs to run.')

    def clean(self, agg=False):
        """
        Clean the previous jobs or aggregators so than they can operate again
        (the post functions return False after the clean).

        :param agg bool: clean aggregators instead of jobs if True
        """
        if not self.options.clean:
            return
        for opr in self.added:
            if opr.cls.agg ^ agg:
                continue
            for job in self.jobs:
                opr.cls(job, name=opr.name).clean()

    def runProj(self, agg=False, **kwargs):
        """
        Run all jobs or aggregators registered in the project.

        :param agg bool: run aggregators instead of jobs.
        """
        cpu = self.max_cpu if agg else self.cpu[0]
        prog = self.options.screen and jobutils.PROGRESS in self.options.screen
        jobs = None if agg else self.jobs
        self.proj.run(np=cpu, progress=prog, jobs=jobs, **kwargs)

    def logStatus(self):
        """
        Look into each parameter set and report the status.
        """
        status_file = f"{self.options.JOBNAME}_status{symbols.LOG_EXT}"
        with open(status_file, 'w') as fh:
            # Fetching status and Fetching labels are printed to err handler
            self.proj.print_status(detailed=True,
                                   jobs=self.jobs,
                                   file=fh,
                                   err=fh)
        # Log job status
        status = [self.proj.get_job_status(x) for x in self.jobs]
        ops = [x[self.OPERATIONS] for x in status]
        completed = [all([y[self.COMPLETED] for y in x.values()]) for x in ops]
        failed_num = len([x for x in completed if not x])
        self.log(
            f"{len(self.jobs) - failed_num} / {len(self.jobs)} completed jobs."
        )

        if not failed_num:
            return
        id_ops = []
        for completed, op, stat, in zip(completed, ops, status):
            if completed:
                continue
            failed_ops = [x for x, y in op.items() if not y[self.COMPLETED]]
            id_ops.append([stat[self.JOB_ID], ', '.join(reversed(failed_ops))])
        id_ops = pd.DataFrame(id_ops, columns=[self.JOB_ID, 'operations'])
        id_ops.set_index(self.JOB_ID, inplace=True)
        self.log(id_ops.to_markdown())

    def setPreAfter(self, pre, cur):
        """
        Set the prerequisite of a job.

        :param pre 'types.SimpleNamespace': the operation runs first
        :param cur 'types.SimpleNamespace': the operation runs after
        """
        if pre is None or cur is None:
            return
        flow.project.FlowProject.pre.after(pre.opr)(cur.opr)
        self.prereq[cur.name].append(pre.name)

    def setAggs(self):
        """
        Set the aggregator operators.
        """
        self.add(self.AggClass)

    def setAggProj(self, filter=None):
        """
        Initiate the aggregation project.

        :param filter dict: the parameter filter.
        """
        prj = self.proj.path if self.proj else self.options.prj_path
        try:
            self.proj = flow.project.FlowProject.get_project(prj)
        except LookupError as err:
            self.error(str(err))
        if not self.jobs:
            self.jobs = self.proj.find_jobs(filter=filter)
        if not self.jobs:
            self.error(f"No jobs to aggregate.")
