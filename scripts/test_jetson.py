"""test.py measurement flow with V4L2 camera and shared ONNX/TensorRT backend."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from dinolite_camera import DinoLiteCamera
from measurement_pipeline import Pipeline, save_measurement


class CameraGUI:
    def __init__(self, root, config, file_mode=False):
        self.root, self.config = root, config
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference')
        self.pipeline = None
        self.initialized = False
        self.camera = None
        self.pending = None
        self.closing = False
        self.folder = None
        self.file_mode = file_mode
        self.source = None
        root.title('MOSA Measurement (NDC Task) — Jetson')
        main = ttk.Frame(root, padding=10)
        main.pack()
        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0)
        results = ttk.LabelFrame(main, text='검사 결과')
        results.grid(row=1, column=0)
        self.result_images = []
        for i, title in enumerate(('측정 사진 전처리', 'AI 분석 결과(NG 영역)', 'AI 분석 결과(Heatmap)')):
            ttk.Label(results, text=title).grid(row=i*2, column=0, sticky='w')
            label = ttk.Label(results)
            label.grid(row=i*2+1, column=0)
            self.result_images.append(label)
            self.image(label, np.full((120, 800, 3), 200, np.uint8))
        panel = ttk.Frame(main, padding=15)
        panel.grid(row=0, column=1, rowspan=2, sticky='ns')
        ttk.Label(panel, text=f"Model: {config['name']}\n{config['backend']}", font=('Arial', 18)).pack()
        self.folder_label = ttk.Label(panel, text='저장 폴더: 미선택', wraplength=260)
        self.folder_label.pack(pady=10)
        ttk.Button(panel, text='저장 폴더 선택', command=self.select_folder).pack(fill='x')
        ttk.Label(panel, text='측정 이름').pack()
        self.name = ttk.Entry(panel)
        self.name.pack(fill='x')
        ttk.Label(panel, text='Threshold (-20~20; 내부 값은 /10)').pack()
        self.threshold = ttk.Entry(panel)
        self.threshold.insert(0, str(config.get('threshold', 0)))
        self.threshold.pack(fill='x')
        if file_mode:
            ttk.Button(panel, text='BMP 선택', command=self.select_image).pack(fill='x')
        self.measure_button = ttk.Button(panel, text='측 정', command=self.capture_image, state='disabled')
        self.measure_button.pack(fill='x', pady=20, ipady=15)
        self.result_label = tk.Label(panel, text='LOADING', bg='gray', fg='white', font=('Arial', 32))
        self.result_label.pack(fill='x', pady=20)
        self.status = ttk.Label(panel, text='모델 로드 중', wraplength=260)
        self.status.pack()
        self.last_save = ttk.Label(panel, text='', wraplength=260)
        self.last_save.pack(pady=10)
        self.pending = self.pool.submit(self.initialize)
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(50, self.update_gui)

    def initialize(self):
        # CUDA session is created, used and destroyed in this same worker.
        self.pipeline = Pipeline(self.config)
        self.pipeline.__enter__()
        if not self.file_mode:
            config = json.loads(Path(self.config['camera_config']).read_text(encoding='utf-8'))
            self.camera = DinoLiteCamera(self.config.get('device', '/dev/video0'), initial_config=config)
            self.camera.ready.result()
        self.initialized = True
        return None

    def select_folder(self):
        value = filedialog.askdirectory()
        if value:
            self.folder = Path(value)
            self.folder_label.configure(text=value)

    def select_image(self):
        value = filedialog.askopenfilename(filetypes=[('Images', '*.bmp *.png')])
        if value:
            self.source = Path(value)

    def image(self, label, bgr):
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        label.configure(image=photo)
        label.image = photo

    def capture_image(self):
        if self.pending is not None:
            return
        try:
            name = self.name.get().strip()
            if not name or name in ('.', '..') or '/' in name or '\\' in name:
                raise ValueError('유효한 측정 이름을 입력하세요')
            if self.folder is None:
                raise ValueError('저장 폴더를 선택하세요')
            threshold = float(self.threshold.get())
            if not np.isfinite(threshold) or not -20 <= threshold <= 20:
                raise ValueError('Threshold는 -20~20입니다')
            if self.file_mode and self.source is None:
                raise ValueError('BMP 파일을 선택하세요')
            folder = self.folder / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
            self.measure_button.configure(state='disabled')
            self.result_label.configure(text='WAIT', bg='gray')
            self.pending = self.pool.submit(self.measure, folder, threshold/10, self.source)
        except Exception as exc:
            messagebox.showerror('측정 오류', str(exc))

    def measure(self, folder, threshold, source):
        if self.file_mode:
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError('이미지를 읽을 수 없습니다')
            camera_report = {'mode': 'file', 'source': str(source), 'camera_checks': 'not performed'}
        else:
            config = json.loads(Path(self.config['camera_config']).read_text(encoding='utf-8'))
            capture = self.camera.prepare_inference(config).result()
            if capture['frame'] is None:
                raise RuntimeError('카메라 조건 FAIL: ' + json.dumps(capture['report'], ensure_ascii=False))
            frame, camera_report = capture['frame'], capture['report']
        result = self.pipeline.run(frame, threshold)
        save_measurement(folder, frame, result, self.config, threshold, camera_report)
        rgb = result['display_rgb']
        amap = result['anomaly_map']
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = base.copy()
        mask = (amap >= threshold).astype(np.uint8)*255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill = base.copy()
        cv2.drawContours(fill, contours, -1, (0, 0, 255), cv2.FILLED)
        overlay = cv2.addWeighted(fill, .3, base, .7, 0)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
        if result['decision'] == 'OK':
            overlay = base.copy()
        normalized = cv2.normalize(amap.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.addWeighted(base, .5, cv2.applyColorMap(normalized, cv2.COLORMAP_JET), .5, 0)
        views = []
        for view in (base, overlay, heat):
            h = view.shape[0]
            offset = int(h*float(self.config.get('crop_v_offset', 0)))
            start, end = h//2-h//8+offset, h//2+h//8+offset
            if not 0 <= start < end <= h:
                raise ValueError('crop_v_offset outside image')
            views.append(cv2.resize(view[start:end], (800, 160)))
        if not cv2.imwrite(str(folder/'result.jpg'), cv2.vconcat(views[1:])):
            raise RuntimeError('결과 이미지 저장 실패')
        return frame, result, views, folder

    def update_gui(self):
        if self.closing:
            if self.pending.done() and (self.camera is None or not self.camera.is_alive()):
                self.pool.shutdown(wait=False)
                self.root.destroy()
                return
            self.root.after(50, self.update_gui)
            return
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            try:
                value = future.result()
                if value is not None:
                    frame, result, views, folder = value
                    self.image(self.video_label, cv2.resize(frame, (800, 600))[240:360])
                    for label, view in zip(self.result_images, views):
                        self.image(label, view)
                    self.result_label.configure(text=result['decision'], bg='green' if result['decision']=='OK' else 'red')
                    self.status.configure(text=f"score={result['pred_score']:.6g}\n{result['timings']}")
                    self.last_save.configure(text=str(folder))
                else:
                    self.result_label.configure(text='READY', bg='gray')
                    self.status.configure(text='측정 준비 완료')
                self.measure_button.configure(state='normal')
            except Exception as exc:
                self.result_label.configure(text='FAIL', bg='blue')
                self.status.configure(text=str(exc))
                if self.initialized:
                    self.measure_button.configure(state='normal')
        if self.camera is not None:
            frame, error = self.camera.latest()
            if frame is not None:
                self.image(self.video_label, cv2.resize(frame, (800, 600))[240:360])
            elif error:
                self.video_label.configure(image='')
        self.root.after(50, self.update_gui)

    def on_close(self):
        if self.closing:
            return
        self.closing = True
        self.measure_button.configure(state='disabled')
        if self.camera:
            self.camera.close()
        self.pending = self.pool.submit(self.cleanup)

    def cleanup(self):
        if self.camera:
            self.camera.close()
        if self.pipeline is not None and hasattr(self.pipeline, 'backend'):
            self.pipeline.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--file-mode', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    root = tk.Tk()
    CameraGUI(root, config, args.file_mode)
    root.mainloop()


if __name__ == '__main__':
    main()
