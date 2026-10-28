"""Background durability checks for completed MOSA output writes (stdlib only)."""
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import sys
import threading
import time


def _sync_outputs(image_paths, csv_path, csv_fd, require_mount):
    directories = set()
    errors = []
    for path in (*image_paths, *((csv_path,) if csv_path is not None else ())):
        try:
            require_mount(path)
            # Include both the target and the directory containing a file symlink.
            directories.add(os.path.dirname(os.path.realpath(path)))
            directories.add(os.path.realpath(os.path.dirname(path) or '.'))
            if path == csv_path:
                # append() already flushed Python's buffer. This duplicate keeps
                # the original open file alive across later log switches/close.
                os.fsync(csv_fd)
            else:
                with open(path, 'rb') as file:
                    os.fsync(file.fileno())
        except Exception as exc:
            errors.append(f'{path}: {exc}')

    for directory in sorted(directories):
        try:
            require_mount(directory)
            fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            errors.append(f'{directory}: {exc}')
    if errors:
        raise RuntimeError('Storage synchronization failed:\n' + '\n'.join(errors))


class BackgroundStorageSync:
    """Serialize fsync work without holding up the next GUI measurement."""

    def __init__(self, require_mount):
        self._require_mount = require_mount
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mosa-storage')
        self._errors = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._pending = 0

    @property
    def pending(self):
        with self._lock:
            return self._pending

    def submit(self, image_paths, *, csv_path=None, csv_fd=None):
        """Take ownership of csv_fd, including when scheduling fails."""
        submitted = time.perf_counter()
        with self._lock:
            self._pending += 1
        try:
            paths = tuple(dict.fromkeys(os.fspath(path) for path in image_paths))
            return self._executor.submit(self._run, paths, csv_path, csv_fd, submitted)
        except Exception:
            if csv_fd is not None:
                os.close(csv_fd)
            with self._lock:
                self._pending -= 1
            raise

    def _run(self, paths, csv_path, csv_fd, submitted):
        started = time.perf_counter()
        status = 'ok'
        try:
            _sync_outputs(paths, csv_path, csv_fd, self._require_mount)
            return {'queue_seconds': started - submitted,
                    'sync_seconds': time.perf_counter() - started}
        except Exception as exc:
            status = 'failed'
            self._errors.put(str(exc))
            print(f'[Storage sync] {exc}', file=sys.stderr, flush=True)
            raise
        finally:
            try:
                if csv_fd is not None:
                    os.close(csv_fd)
            finally:
                print(f'[Storage sync] status={status} queue={started - submitted:.3f}s '
                      f'sync={time.perf_counter() - started:.3f}s '
                      f'images={len(paths)} csv={int(csv_path is not None)}', flush=True)
                with self._lock:
                    self._pending -= 1

    def take_errors(self):
        errors = []
        while True:
            try:
                errors.append(self._errors.get_nowait())
            except queue.Empty:
                return errors

    def close(self):
        # Called after Tk closes; do not abandon queued syncs at process exit.
        self._executor.shutdown(wait=True)
