# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# Part 3: Classification
# https://www.udemy.com/course/machinelearning
"""
Machine learning regression.
"""
from nemd import logutils
from nemd import ml
from nemd import parserutils

if __name__ == "__main__":
    logutils.Script.run(ml.Classification,
                        parserutils.Clf(descr=__doc__),
                        file=True)
