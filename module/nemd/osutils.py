# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
os utilities
"""
import os
import shutil


def symlink(src, dst, overwrite=True, **kwargs):
    """
    Create a symbolic link. Overwrite the existing link if overwrite is True.

    :param src str: source file
    :param dst str: destination file
    :param overwrite bool: whether to overwrite the existing link
    """
    try:
        os.symlink(src, dst, **kwargs)
    except FileExistsError:
        if not overwrite:
            raise
        os.remove(dst)
        os.symlink(src, dst, **kwargs)


class chdir:
    """
    Context manager for changing the working directory.
    """

    def __init__(self, dirname, rmtree=False):
        """
        :param dirname str: the target directory to be created and switched to.
        :param rmtree bool: remove the working directory on exiting if True.
        """
        self.dirname = dirname
        self.rmtree = rmtree
        self.original = os.getcwd()

    def __enter__(self):
        """
        Change to the working directory on entering. (create the working
        directory if it doesn't exist)

        :return str: the working directory.
        """
        if self.dirname is os.curdir:
            return self.dirname
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        os.chdir(self.dirname)
        self.working = os.getcwd()
        return self.dirname

    def __exit__(self, *args, **kwargs):
        """
        Switch back to the original working directory on exiting. (remove the
        working directory if rmtree is True)
        """
        if self.dirname is os.curdir:
            return
        os.chdir(self.original)
        if self.rmtree:
            shutil.rmtree(self.dirname)