# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# Machine Learning A-Z: AI, Python & R + ChatGPT Prize 2025 by Kirill Eremenko,
# Hadelin de Ponteves, SuperDataScience Team, Ligency
# https://www.udemy.com/course/machinelearning
"""
Machine learning regression.
"""
from nemd import logutils
from nemd import ml
from nemd import parserutils

if __name__ == "__main__":
    logutils.Script.run(ml.Ml, parserutils.Reg(descr=__doc__), file=True)
