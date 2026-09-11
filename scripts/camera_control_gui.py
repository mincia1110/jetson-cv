"""Camera-only Tkinter verification UI for the Jetson port of test.py."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from dinolite_camera import DinoLiteCamera
from camera_conditions import EXPOSURE_COMMANDS, validate_conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--brightness', type=int, help='Validated V4L2 brightness value')
    parser.add_argument('--config', type=Path, default=Path('camera_check.example.json'),
                        help='Camera condition JSON; reloaded on each check')
    args = parser.parse_args()
    args.config = args.config.resolve()
    print(f'Camera config: {args.config}', flush=True)
    initial_config = json.loads(args.config.read_text(encoding='utf-8'))
    print(f"ExposureTime: {initial_config.get('ExposureTime')!r}", flush=True)
    if args.brightness is not None:
        initial_config['Brightness'] = args.brightness
    configured = (initial_config.get('Brightness') is not None and
                  initial_config.get('ExposureTime') is not None)
    config_error = None
    if configured:
        try:
            initial_config = validate_conditions(initial_config)
        except (ValueError, KeyError, TypeError) as exc:
            configured = False
            config_error = str(exc)
    root = tk.Tk()
    root.title('Dino-Lite — Jetson camera check')
    ttk.Label(root, text=f'설정 파일: {args.config}', wraplength=900).pack()
    if config_error:
        ttk.Label(root, text=f'초기값 미적용: {config_error}', wraplength=900,
                  foreground='red').pack()
    controls = {} if args.brightness is None else {'brightness': args.brightness}
    camera = DinoLiteCamera(device=args.device, fps=args.fps, controls=controls,
                            initial_config=initial_config if configured else None)
    status = tk.StringVar(value='카메라 연결 중…')
    preview = ttk.Label(root)
    preview.pack(padx=10, pady=10)
    ttk.Label(root, textvariable=status).pack()
    bar = ttk.Frame(root, padding=10)
    bar.pack()
    report_box = tk.Text(root, height=10, width=110, state='disabled')
    report_box.pack(padx=10, pady=5)
    ttk.Label(bar, text='평균 프레임 수').pack(side='left')
    count = tk.StringVar(value=str(initial_config.get('capture_no', 5)))
    ttk.Spinbox(bar, from_=1, to=100, textvariable=count, width=5).pack(side='left')
    pending = [camera.ready, '연결 완료', None]
    buttons = []
    closing = False
    settings = ttk.Frame(root, padding=10)
    settings.pack()
    if initial_config.get('ExposureValue') is not None:
        ttk.Label(root, text=f"Windows ExposureValue={initial_config['ExposureValue']} "
                  '— Linux 노출 시간 대응 미확인. 임의 자동 변환하지 않습니다.').pack()
    brightness = tk.StringVar(value='' if initial_config.get('Brightness') is None
                              else str(initial_config['Brightness']))
    exposure = tk.StringVar(value=initial_config.get('ExposureTime') or '')
    ttk.Label(settings, text='초기 밝기 (V4L2)').pack(side='left')
    ttk.Entry(settings, textvariable=brightness, width=7).pack(side='left')
    ttk.Label(settings, text='고정 노출 시간').pack(side='left')
    ttk.Combobox(settings, textvariable=exposure, values=list(EXPOSURE_COMMANDS),
                 state='readonly', width=12).pack(side='left')

    def read_config():
        config = json.loads(args.config.read_text(encoding='utf-8'))
        config.update(Brightness=int(brightness.get()), ExposureTime=exposure.get(),
                      capture_no=int(count.get()))
        return validate_conditions(config)

    def apply_initial():
        try:
            start(camera.apply_initial(read_config()), '초기 고정값 적용 완료')
        except Exception as exc:
            messagebox.showerror('설정 오류', str(exc))

    apply_button = ttk.Button(settings, text='초기 고정값 적용', command=apply_initial)
    apply_button.pack(side='left', padx=5)
    buttons.append(apply_button)

    def start(future, text, path=None):
        pending[:] = [future, text, path]
        for button in buttons:
            button.configure(state='disabled')
        status.set('처리 중…')

    def capture():
        try:
            config = read_config()
        except Exception as exc:
            messagebox.showerror('입력 오류', str(exc))
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.bmp', filetypes=[('Bitmap', '*.bmp')],
            initialfile=datetime.now().strftime('capture_%Y%m%d_%H%M%S.bmp'))
        if path:
            start(camera.prepare_inference(config), '촬영 완료', Path(path))

    def check_conditions():
        try:
            config = read_config()
            start(camera.check_conditions(config), '조건 점검 완료')
        except Exception as exc:
            messagebox.showerror('조건 설정 오류', str(exc))

    for label, command in [
        ('상태 확인 후 촬영', capture),
        ('LED OFF', lambda: start(camera.set_led(False), 'LED OFF 명령 완료')),
        ('LED ON', lambda: start(camera.set_led(True), 'LED ON 명령 완료')),
        ('AE ON', lambda: start(camera.set_auto_exposure(True), 'AE ON 명령 완료')),
        ('AE OFF·노출 고정', lambda: start(camera.set_auto_exposure(False), 'AE OFF 명령 완료')),
        ('재연결·설정 적용', lambda: start(camera.reconnect(), '재연결 완료')),
        ('조건 검사·자동 복구', check_conditions),
    ]:
        button = ttk.Button(bar, text=label, command=command)
        button.pack(side='left', padx=4)
        buttons.append(button)

    def tick():
        if closing:
            if not camera.is_alive():
                root.destroy()
                return
            root.after(50, tick)
            return
        future, text, path = pending
        if future is not None and future.done():
            try:
                result = future.result()
                if isinstance(result, dict) and 'report' in result:
                    if result['frame'] is not None and path is not None:
                        if not cv2.imwrite(str(path), result['frame']):
                            raise RuntimeError(f'저장 실패: {path}')
                    elif path is not None:
                        text = '조건 불일치: 촬영 저장·추론 차단'
                    result = result['report']
                    path = None
                if path is not None:
                    if not cv2.imwrite(str(path), result):
                        raise RuntimeError(f'저장 실패: {path}')
                    text = f'{text}: {path}'
                elif isinstance(result, dict):
                    if 'before' in result:
                        detail = json.dumps(result, ensure_ascii=False, indent=2)
                        report_box.configure(state='normal')
                        report_box.delete('1.0', 'end')
                        report_box.insert('end', detail)
                        report_box.configure(state='disabled')
                        print(detail, flush=True)
                        text = (f"{result['status']} / 리셋 {result['reset_performed']} / "
                                f"추론 허용 {result['allow_inference']} / 노출 readback 없음")
                    else:
                        text += f' — {result}'
                status.set(text)
            except Exception as exc:
                status.set(str(exc))
                messagebox.showerror('카메라 오류', str(exc))
            pending[:] = [None, '', None]
        for button in buttons:
            button.configure(state='disabled' if pending[0] is not None else 'normal')
        frame, error = camera.latest()
        if frame is not None:
            rgb = cv2.cvtColor(cv2.resize(frame, (800, 600)), cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            preview.configure(image=photo)
            preview.image = photo
        elif error:
            preview.configure(image='')
            status.set(error)
        root.after(50, tick)

    def close():
        nonlocal closing
        closing = True
        for button in buttons:
            button.configure(state='disabled')
        status.set('캡처 스레드 종료 대기 중…')
        camera.close()

    root.protocol('WM_DELETE_WINDOW', close)
    tick()
    root.mainloop()


if __name__ == '__main__':
    main()
