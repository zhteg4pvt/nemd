# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# Part 4: Clustering
# https://www.udemy.com/course/machinelearning
"""
Machine learning clustering.
"""
from nemd import logutils
from nemd import ml
from nemd import parserutils

if __name__ == "__main__":
    logutils.Script.run(ml.Cluster, parserutils.Clus(descr=__doc__), file=True)
