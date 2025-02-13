import os
from pathlib import Path

import pytest

from nemd import osutils


class TestFunction:

    def testSymlink(self, tmp_dir):
        with open('original.txt', 'w') as fh:
            fh.write('original')
        osutils.symlink('original.txt', 'link.txt')
        with open('link.txt', 'r') as fh:
            line = fh.read()
        assert 'original' == line
        with open('second.txt', 'w') as fh:
            fh.write('second')
        osutils.symlink('second.txt', 'link.txt')
        with open('link.txt', 'r') as fh:
            line = fh.read()
        assert 'second' == line


class TestChdir:

    @pytest.fixture
    def chdir(self, name, rmtree):
        return osutils.chdir(name, rmtree=rmtree)

    @pytest.mark.parametrize('name,rmtree', [('mydir', False), ('mydir', True),
                                             (os.curdir, True)])
    def testChdir(self, chdir, name, rmtree, tmp_dir, filename='file.txt'):
        with chdir as dirname:
            Path(filename).touch()
        assert name == dirname
        is_file = Path(f'{chdir.original}/{name}/{filename}').is_file()
        if dirname == os.curdir or not rmtree:
            assert is_file
            return
        assert not is_file
