"""Measure synthetic output writes, GUI queue submission, and background fsync."""
import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from mosa_storage import BackgroundStorageSync


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True, help='Existing output directory')
    parser.add_argument('--mount-point', type=Path, required=True, help='Required mounted filesystem')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--image-mib', type=int, default=15)
    args = parser.parse_args()
    if args.iterations < 1 or args.image_mib < 1:
        parser.error('iterations and image-mib must be positive')
    directory = args.directory.resolve()
    mount_point = args.mount_point.resolve()

    def require_mount(path):
        if not os.path.ismount(mount_point):
            raise RuntimeError(f'Benchmark mount is unavailable: {mount_point}')
        if os.path.commonpath((os.path.realpath(path), str(mount_point))) != str(mount_point):
            raise RuntimeError(f'Benchmark path is outside {mount_point}: {path}')

    require_mount(directory)
    if not directory.is_dir():
        parser.error('directory must already exist')
    raw_bytes = os.urandom(args.image_mib * 1024 * 1024)
    result_bytes = os.urandom(512 * 1024)
    samples = []
    storage = BackgroundStorageSync(require_mount)
    try:
        with tempfile.TemporaryDirectory(prefix='mosa-sync-bench-', dir=directory) as folder:
            root = Path(folder)
            csv_path = root / 'log.csv'
            with csv_path.open('a', encoding='utf-8') as csv_file:
                for i in range(args.iterations):
                    require_mount(root)
                    started = time.perf_counter()
                    raw, result = root / f'{i}-raw.bin', root / f'{i}-result.bin'
                    raw.write_bytes(raw_bytes)
                    result.write_bytes(result_bytes)
                    csv_file.write(f'{i},synthetic-output\n')
                    csv_file.flush()
                    written = time.perf_counter()
                    future = storage.submit([raw, result], csv_path=str(csv_path),
                                            csv_fd=os.dup(csv_file.fileno()))
                    submitted = time.perf_counter()
                    timing = future.result()
                    samples.append(dict(write_seconds=written - started,
                                        submit_seconds=submitted - written, **timing))
    finally:
        storage.close()
    summary = {key: {'median_ms': statistics.median(row[key] for row in samples) * 1000,
                     'max_ms': max(row[key] for row in samples) * 1000}
               for key in samples[0]}
    print(json.dumps(dict(directory=str(directory), iterations=args.iterations,
                          synthetic_raw_mib=args.image_mib, synthetic_result_kib=512,
                          timings=summary), indent=2))


if __name__ == '__main__':
    main()
