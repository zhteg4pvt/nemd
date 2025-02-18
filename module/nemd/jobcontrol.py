# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This class handles both jobs and aggregators under the jobcontrol:
    1) define the operators
    2) initiate or load project
    3) parametrize the operators
    4) distribute cpu resources among the operators
    5) register operators under the project
    6) clean previous results
    7) execute the registered operators
    8) log the operator status and message.
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
from nemd import symbols
from nemd import taskbase


class Runner(logutils.Base):
    """
    The main class to set up a workflow.
    """

    WORKSPACE = symbols.WORKSPACE
    ARGS = taskbase.Base.ARGS
    PREREQ = taskbase.Job.PREREQ
    COMPLETED = 'completed'
    OPERATIONS = 'operations'
    JOB_ID = 'job_id'
    FLAG_SEED = jobutils.FLAG_SEED
    MESSAGE = taskbase.Base.MESSAGE
    AGG_NAME_EXT = f"{symbols.POUND_SEP}agg"

    def __init__(self, options, argv, logger=None):
        """
        :param options: parsed commandline options
        :type options: 'argparse.Namespace'
        :param argv: list of commandline arguments
        :type argv: list
        :param logger: print to this logger if exists
        :type logger: 'logging.Logger'
        """
        super().__init__(logger=logger)
        self.options = options
        self.argv = argv
        self.state = {}
        self.jobs = []
        self.oprs = {}
        self.classes = {}
        self.cpu = None
        self.project = None
        self.agg_project = None
        self.prereq = collections.defaultdict(list)
        # flow/project.py gets logger from logging.getLogger(__name__)
        logutils.Logger.get('flow.project')

    def run(self):
        """
        The main method to run the integration tests.

        The linear pipline handles three things on request:
        1) clean previous projects
        2) run a project with task jobs
        3) run a project with aggregator jobs
        """
        if jobutils.TASK in self.options.jtype:
            self.setJob()
            self.setProject()
            self.setState()
            self.setCpu()
            self.addJobs()
            self.cleanJobs()
            self.plot()
            self.runJobs()
            self.logStatus()
            self.logMessage()
        if jobutils.AGGREGATOR in self.options.jtype:
            self.setAggJobs()
            self.setAggProject()
            self.cleanAggJobs()
            self.runAggJobs()

    def setJob(self):
        """
        Set the tasks for the job.
        """
        raise NotImplementedError('This method adds operators as job tasks.')

    def setOpr(self, TaskCLass, agg=False, **kwargs):
        """
        Set one operation for the job.

        :param TaskCLass 'task.Task' (sub)-class: the task class
        :param agg str: whether this is aggregator operation or not
        :return: the name of the operation
        """
        opr = TaskCLass.getAgg(**kwargs) if agg else TaskCLass.getOpr(**kwargs)
        name = opr.__name__
        self.oprs[name] = opr
        self.classes[name] = TaskCLass.AggClass if agg else TaskCLass.JobClass
        return name

    def setProject(self):
        """
        Initiate the project.
        """
        self.project = flow.project.FlowProject.init_project()

    def setState(self):
        """
        Set the state flags and values.
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
        jobutils.pop_arg(self.argv, self.FLAG_SEED)
        seed = getattr(self.options, self.FLAG_SEED[1:], 1)
        seeds = (seed_incre + seed) % symbols.MAX_INT32
        self.state = {self.FLAG_SEED: list(map(str, seeds))}

    def setCpu(self):
        """
        Set cpu numbers for the project.
        """
        if self.options.cpu is None:
            # No cpu specified
            # Debug mode: 1 cpu as total to avoid parallelism
            # Production mode: 75% of cpu count as total to avoid overloading
            total = 1 if self.options.debug else round(os.cpu_count() * 0.75)
            # Single cpu per job ensures efficiency
            self.cpu = [max([total, 1]), 1]
            return
        try:
            per_subjob = self.options.cpu[1]
        except IndexError:
            # Only total cpu specified: evenly distribute among subjobs
            subjob_num = np.prod([len(x) for x in self.state.values()])
            per_subjob = max([math.floor(self.options.cpu[0] / subjob_num), 1])
        self.cpu = [math.floor(self.options.cpu[0] / per_subjob), per_subjob]

    def addJobs(self):
        """
        Add jobs to the project.

        NOTE: _StatePointDict warns NumpyConversionWarning if statepoint dict
        contains numerical data types.
        """
        input_args = self.argv[:]
        try:
            index = self.argv.index(jobutils.FLAG_CPU)
        except ValueError:
            pass
        else:
            input_args[index + 1] = str(self.cpu[1])

        argvs = [[[x, z] for z in y] for x, y in self.state.items()]
        for argv in itertools.product(*argvs):
            # e.g. arg = (['-seed', '0'], ['-scale_factor', '0.95'])
            job = self.project.open_job(dict(tuple(x) for x in argv))
            job.document[self.ARGS] = input_args[:] + sum(argv, [])
            job.document.update({self.PREREQ: self.prereq})
            self.jobs.append(job)

        if not self.jobs:
            self.log_error('No jobs to run.')

    def cleanJobs(self):
        """
        The post functions of the pre-job return False after the clean so that
        the job can run again on request.
        """
        if not self.options.clean:
            return
        for name, JobClass in self.classes.items():
            if name.endswith(self.AGG_NAME_EXT):
                continue
            for job in self.jobs:
                JobClass(job, jobname=name).clean()

    def runJobs(self, **kwargs):
        """
        Run all jobs registered in the project.
        """
        prog = self.options.screen and jobutils.PROGRESS in self.options.screen
        self.project.run(np=self.cpu[0],
                         progress=prog,
                         jobs=self.jobs,
                         **kwargs)

    def plot(self):
        """
        Plot the task workflow graph.
        """
        if not self.options.debug:
            return
        import matplotlib
        obackend = matplotlib.get_backend()
        backend = obackend if self.options.interactive else 'Agg'
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        depn = np.asarray(self.project.detect_operation_graph())
        graph = nx.DiGraph(depn)
        pos = nx.spring_layout(graph)
        names = [x for x in self.project.operations.keys()]
        labels = {key: name for (key, name) in zip(range(len(names)), names)}
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        nx.draw_networkx(graph, pos, ax=ax, labels=labels)
        if self.options.interactive:
            print("Showing task workflow graph. Click X to close the figure "
                  "and continue..")
            plt.show(block=True)
        fig.savefig(self.options.jobname + '_nx.png')

    def logStatus(self):
        """
        Look into each job and report the status.
        """
        status_file = f"{self.options.jobname}_status{symbols.LOG_EXT}"
        with open(status_file, 'w') as fh:
            # Fetching status and Fetching labels are printed to err handler
            self.project.print_status(detailed=True,
                                      jobs=self.jobs,
                                      file=fh,
                                      err=fh)
        # Log job status
        status = [self.project.get_job_status(x) for x in self.jobs]
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

    def logMessage(self):
        """
        Log the messages of the jobs.
        """
        ops = [
            self.project.get_job_status(x)[self.OPERATIONS] for x in self.jobs
        ]
        completed = [all(y[self.COMPLETED] for y in x.values()) for x in ops]
        if not any(completed):
            return
        jobs = [x for x, y in zip(self.jobs, completed) if y]
        fjobs = [x for x in jobs if any(x.doc.get(self.MESSAGE, {}).values())]
        self.log(f"{len(jobs) - len(fjobs)} / {len(jobs)} succeeded jobs.")
        if not fjobs:
            return
        func = lambda x: '\n'.join(f"{k}: {v}" for k, v in x.items() if v)
        data = {self.MESSAGE: [func(x.doc[self.MESSAGE]) for x in fjobs]}
        fcn = lambda x: '\n'.join(f"{k.strip('-')}: {v}" for k, v in x.items())
        data['parameters'] = [fcn(x.statepoint) for x in fjobs]
        ids = pd.Index([x.id for x in fjobs], name=self.JOB_ID)
        info = pd.DataFrame(data, index=ids)
        self.log(info.to_markdown())

    def setAggJobs(self, TaskClass=taskbase.Task):
        """
        Register aggregators with prerequisites.

        :param 'taskbase.Task': the agg job of this task is registered.
        """
        pnames = [x for x in self.oprs.keys() if x.endswith(self.AGG_NAME_EXT)]
        # taskbase.AggJob reports the task timing
        name = self.setAgg(TaskClass)
        for pre_name in pnames:
            self.setPreAfter(pre_name, name)

    def setAgg(self, *args, **kwargs):
        """
        Set one aggregator that analyzes jobs for statics, chemical space, and
        states.
        """
        return self.setOpr(*args,
                           **kwargs,
                           agg=True,
                           logger=self.logger,
                           options=self.options)

    def setAggProject(self):
        """
        Initiate the aggregation project.
        """
        prj = self.project.path if self.project else self.options.prj_path
        try:
            self.agg_project = flow.project.FlowProject.get_project(prj)
        except LookupError as err:
            self.log_error(str(err))

    def cleanAggJobs(self, filter=None):
        """
        Run aggregation project.

        :param dict: the filter to select jobs.
        """
        if not self.options.clean:
            return

        jobs = self.jobs if self.jobs else self.agg_project.find_jobs(
            filter=filter)
        if not jobs:
            return
        for jobname, JobClass in self.classes.items():
            if not jobname.endswith(self.AGG_NAME_EXT):
                continue
            JobClass(*jobs, jobname=jobname).clean()

    def runAggJobs(self):
        """
        Run aggregation project.
        """
        prog = self.options.screen and jobutils.PROGRESS in self.options.screen
        # FIXME: no parallelism as multiple aggregation touch the same file
        try:
            self.agg_project.run(np=1, progress=prog)
        except IndexError:
            # workspace dir removed
            self.log_error(f"no jobs to aggregate.")

    def setPreAfter(self, pre, cur):
        """
        Set the prerequisite of a job.

        :param pre str: the operation name runs first
        :param cur str: the operation name who runs after the prerequisite job
        """
        if pre is None or cur is None:
            return
        flow.project.FlowProject.pre.after(self.oprs[pre])(self.oprs[cur])
        self.prereq[cur].append(pre)
