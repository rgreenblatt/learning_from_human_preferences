from pathlib import Path
import sys

import os
import logging
from pprint import pformat
from functools import partial

from colorlog import ColoredFormatter

float2int = lambda f: int(float(f))

WIDTH = 100

pformat = partial(pformat, depth=3, width=WIDTH, compact=False, indent=1)


def is_readable(s):
    return len(str(s)) <= WIDTH


def format_with_newline(s):
    return pformat(s) if is_readable(s) else '\n' + pformat(s)


class PrettyFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.args, tuple):
            if len(record.args):
                record.args = tuple(
                    [format_with_newline(a) for a in record.args]
                )
            else:
                record.msg = format_with_newline(record.msg)
        else:
            record.args = format_with_newline(record.args)
        return record


def get_logger(log_level, name=None):
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white,bold',
            'INFOV': 'cyan,bold',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)
    log = logging.getLogger(name)
    log.setLevel(log_level)
    log.handlers = []  # No duplicated handlers
    log.propagate = False  # workaround for duplicated logs in ipython
    log.addHandler(ch)
    log.addFilter(PrettyFilter())

    return log


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
            always_flush=always_flush
        )
        sys.stderr = WriteBroadcaster(
            [self._combined_file, self._stderr_file, sys.stderr],
            always_flush=always_flush
        )

    def __del__(self):
        sys.stderr = self._orig_stderr
        sys.stdout = self._orig_stdout
        self._combined_file.close()
        self._stdout_file.close()
        self._stdout_file.close()
