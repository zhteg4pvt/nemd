# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
psutil utilities.
"""
import queue
import sys
import threading
import time

import psutil
import scipy

from nemd import symbols


class ProcessLinux(psutil.Process):  # pragma: no darwin
    """
    This`psutil.Process` subclass adds a method to get the used memory of the
    process.
    """

    def getUsed(self):
        """
        Get the used memory of the process.

        :return float: “Virtual Memory Size”, the total amount of virtual memory
            used by the process.
        """
        return self.memory_info().vms


class ProcessDarwin(psutil.Process):  # pragma: no linux
    """
    This psutil.Process subclass customizes the methods for the macOS.
    """

    def getUsed(self):
        """
        Get the used memory of the process. On macOS, the memory_info().vms
        returns the total amount of virtual memory available to the process
        (larger than the actual used memory).

        The per-process swap + non-swapped memory can be found in the Memory
        column of the Activity Monitor and MEM column of the `top` command. But,
        no lightweight python api is available.

        The summation of resident set size and swap memory can provide a good
        estimation of the actual usage. However, exiting the previous process
        may not completely free the used swap memory, which can accessed by the
        newly created process.

        Resident Set Size: the non-swapped physical memory used by a process.
        Swap Memory: the amount of the virtual memory swapped out to the disk.

        FIXME: swapped memory used by a process is not available on macOS.
        https://superuser.com/questions/97235/how-much-swap-is-a-given-mac-application-using

        uss (Linux, macOS, Windows): aka “Unique Set Size”, this is the memory
        which is unique to a process and which would be freed if the process was
        terminated right now.
        https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info

        :return float: the best estimated usage of the memory.
        """
        return self.memory_full_info().uss  # psutil.swap_memory().used


Process = ProcessDarwin if sys.platform == symbols.DARWIN else ProcessLinux


class Memory:
    """
    This class periodically profiles memory usage and reports the peak ussage.
    """

    def __init__(self, intvl=2):
        """
        :param intvl float: The periodical checking interval in seconds.
        """
        self.intvl = intvl
        self.peak = queue.Queue()
        self.stop = threading.Event()
        args = (self.peak, self.stop, self.intvl)
        self.thread = threading.Thread(target=self.setPeak, args=args)

    def start(self):
        """
        Start the memory profiling thread.
        """
        self.thread.start()

    @property
    def result(self):
        """
        Stop the profiling and return the peak memory usage.

        :return float: The peak memory usage in MB.
        """
        self.stop.set()
        self.thread.join()
        return self.peak.get(block=False)

    @staticmethod
    def setPeak(peak, stop, intvl):
        """
        Periodically check the memory usage and save the peak value.

        :param peak `queue.Queue`: Save the peak value to this reference.
        :param stop_event `threading.Event`: The stop event to stop checking
        :param intvl float: The periodical checking interval in seconds.
        """
        process = Process()
        original = process.getUsed()
        used = original
        while not stop.is_set():
            time.sleep(intvl)
            used = max([used, process.getUsed()])
        peak.put((used - original) / scipy.constants.mega)
