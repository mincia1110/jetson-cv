"""Camera-only Tkinter verification UI for the Jetson port of test.py."""
import argparse
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from dinolite_camera import DinoLiteCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--brightness', type=int, help='Validated V4L2 brightness value')
    args = parser.parse_args()
    root = tk.Tk()
    root.title('Dino-Lite — Jetson camera check')
    controls = {} if args.brightness is None else {'brightness': args.brightness}
    camera = DinoLiteCamera(device=args.device, fps=args.fps, controls=controls)
    status = tk.StringVar(value='카메라 연결 중…')
    preview = ttk.Label(root)
    preview.pack(padx=10, pady=10)
    ttk.Label(root, textvariable=status).pack()
    bar = ttk.Frame(root, padding=10)
    bar.pack()
    ttk.Label(bar, text='평균 프레임 수').pack(side='left')
    count = tk.StringVar(value='5')
    ttk.Spinbox(bar, from_=1, to=100, textvariable=count, width=5).pack(side='left')
    pending = [camera.ready, '연결 완료', None]
    buttons = []
    closing = False

    def start(future, text, path=None):
        pending[:] = [future, text, path]
        for button in buttons:
            button.configure(state='disabled')
        status.set('처리 중…')

    def capture():
        try:
            n = int(count.get())
            if not 1 <= n <= 100:
                raise ValueError()
        except ValueError:
            messagebox.showerror('입력 오류', '프레임 수는 1~100 정수입니다.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.bmp', filetypes=[('Bitmap', '*.bmp')],
            initialfile=datetime.now().strftime('capture_%Y%m%d_%H%M%S.bmp'))
        if path:
            start(camera.capture(n), '촬영 완료', Path(path))

    for label, command in [
        ('촬영', capture),
        ('LED OFF', lambda: start(camera.set_led(False), 'LED OFF 명령 완료')),
        ('LED ON', lambda: start(camera.set_led(True), 'LED ON 명령 완료')),
        ('AE ON', lambda: start(camera.set_auto_exposure(True), 'AE ON 명령 완료')),
        ('AE OFF·노출 고정', lambda: start(camera.set_auto_exposure(False), 'AE OFF 명령 완료')),
        ('재연결·설정 적용', lambda: start(camera.reconnect(), '재연결 완료')),
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
                if path is not None:
                    if not cv2.imwrite(str(path), result):
                        raise RuntimeError(f'저장 실패: {path}')
                    text = f'{text}: {path}'
                elif isinstance(result, dict):
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
