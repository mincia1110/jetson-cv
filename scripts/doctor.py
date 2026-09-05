"""Read-only inventory; a report is not a GPU execution test."""
import argparse
import importlib
import json
import platform
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version()}
    for name in ('cv2', 'numpy', 'tensorrt', 'polygraphy', 'PyQt5.QtCore', 'PyQt6.QtCore'):
        try:
            mod = importlib.import_module(name)
            report[name] = {'version': getattr(mod, '__version__', getattr(mod, 'PYQT_VERSION_STR', 'unknown'))}
        except Exception as exc:
            report[name] = {'error': str(exc)}
    for command in (['dpkg-query', '-W', 'nvidia-l4t-core', 'nvidia-jetpack'], ['lsusb'], ['v4l2-ctl', '--list-devices']):
        if shutil.which(command[0]):
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            report[' '.join(command)] = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    report['trtexec'] = shutil.which('trtexec') or (str(Path('/usr/src/tensorrt/bin/trtexec')) if Path('/usr/src/tensorrt/bin/trtexec').exists() else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
