"""Launch MOSA with native stdout/stderr logging and optional NvMap display filtering."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

NVMAP_LINES = {
    'NvMapMemAllocInternalTagged: 1075072515 error 12',
    'NvMapMemHandleAlloc: error 0',
}
JPEG_WARNING = 'Corrupt JPEG data: premature end of data segment'


def load_settings(path):
    text = path.read_text(encoding='utf-8-sig')
    text = re.sub(r'"(?:\\.|[^"\\])*"|//[^\r\n]*',
                  lambda m: '' if m.group().startswith('//') else m.group(), text)
    data = json.loads(text)
    hide = data.get('hide_nvmap_messages', False)
    if type(hide) is not bool:
        raise ValueError('hide_nvmap_messages must be true or false')
    return hide


def relay(lines, logfile, terminal, hide_nvmap):
    jpeg_count = hidden_count = 0
    for line in lines:
        stamp = datetime.datetime.now().astimezone().isoformat(timespec='milliseconds')
        logfile.write(f'{stamp} {line}')
        logfile.flush()
        if hide_nvmap and line.strip() in NVMAP_LINES:
            hidden_count += 1
            continue
        if JPEG_WARNING in line:
            jpeg_count += 1
            terminal.write(f'[{stamp} JPEG warning #{jpeg_count}] ')
        terminal.write(line)
        terminal.flush()
    return jpeg_count, hidden_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path, default=Path('MOSA_visualAD_comb_jetson.py'))
    parser.add_argument('--log-dir', type=Path, default=Path('artifacts/runtime-logs'))
    args = parser.parse_args()
    # Same working-directory data.json as the GUI. No second config path.
    hide = load_settings(Path('data.json'))
    script = args.script.resolve(strict=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    path = args.log_dir / (datetime.datetime.now().strftime('mosa-%Y%m%d-%H%M%S-%f') + '.log')
    print(f'[MOSA logging] raw log={path.resolve()} hide_nvmap_messages={hide}', flush=True)
    with path.open('x', encoding='utf-8') as logfile:
        process = subprocess.Popen([sys.executable, '-u', str(script)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            jpeg, hidden = relay(process.stdout, logfile, sys.stdout, hide)
            code = process.wait()
            summary = f'[MOSA logging] exit_code={code} jpeg_warnings={jpeg} hidden_nvmap_lines={hidden}\n'
            logfile.write(summary)
            print(summary, end='', flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
