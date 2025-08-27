# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.

# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module provides plot utilities.
"""
import contextlib
import platform


@contextlib.contextmanager
def pyplot(inav=False, name='the plot'):
    """
    Get the pyplot with requested backend and restoration after usage.

    :param inav bool: use Agg on False (no show)
    :param name str: the name of the plot
    :return 'matplotlib.pyplot': the pyplot with requested backend
    """
    import matplotlib
    obackend = matplotlib.get_backend()
    match platform.system():
        case 'Darwin':  # pragma: no linux
            backend = 'macosx'
        case 'Linux':  # pragma: no darwin
            backend = 'qtagg'
    matplotlib.use(backend)
    from matplotlib import pyplot as plt
    yield plt
    if inav:
        print(f"Showing {name}. Click X to close and continue..")
        plt.show(block=True)
    plt.close('all')
    matplotlib.use(obackend)
