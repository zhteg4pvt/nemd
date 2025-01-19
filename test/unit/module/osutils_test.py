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
