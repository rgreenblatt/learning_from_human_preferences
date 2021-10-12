from pathlib import Path
import sys


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class WriteBroadcaster():
    def __init__(self, to_write, always_flush=False):
        self._to_write = to_write
        self._always_flush = always_flush

    def write(self, data):
        for writable in self._to_write:
            writable.write(data)
        if self._always_flush:
            self.flush()

    def flush(self):
        for writable in self._to_write:
            writable.flush()


class PrintAndLogStdoutStderr():
    def __init__(self, base_file_path, always_flush=False):
        super().__init__()

        self._combined_file = open(f'{base_file_path}/out.txt', 'w')
        self._stdout_file = open(f'{base_file_path}/stdout.txt', 'w')
        self._stderr_file = open(f'{base_file_path}/stderr.txt', 'w')

        self._orig_stderr = sys.stderr
        self._orig_stdout = sys.stdout

        sys.stdout = WriteBroadcaster(
            [self._combined_file, self._stdout_file, sys.stdout],
            always_flush=always_flush)
        sys.stderr = WriteBroadcaster(
            [self._combined_file, self._stderr_file, sys.stderr],
            always_flush=always_flush)

    def __del__(self):
        sys.stderr = self._orig_stderr
        sys.stdout = self._orig_stdout
        self._combined_file.close()
        self._stdout_file.close()
        self._stdout_file.close()
