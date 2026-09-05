"""Minimal V4L2 camera diagnostic, not an application replacement."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int)
    p.add_argument('--height', type=int)
    p.add_argument('--fourcc')
    p.add_argument('--headless', action='store_true')
    args = p.parse_args()
    if args.fourcc and len(args.fourcc) != 4:
        p.error('--fourcc must contain 4 characters')
    if args.output.exists():
        p.error('Output already exists; choose a new path.')
    import cv2
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera. Check device path, permissions and UVC support.')
        if args.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, args.width), (cv2.CAP_PROP_FRAME_HEIGHT, args.height)):
            if value:
                cap.set(prop, value)
        print({key: cap.get(prop) for key, prop in [('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT), ('fps', cv2.CAP_PROP_FPS), ('fourcc', cv2.CAP_PROP_FOURCC)]})
        print('Space: capture; Q/Esc: cancel')
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError('Frame acquisition failed.')
            count += 1
            if args.headless:
                key = 32 if count >= 10 else -1
            else:
                cv2.imshow('Camera diagnostic', frame)
                key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')):
                raise SystemExit('Cancelled without capture.')
            if key == 32:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.output), frame):
                    raise RuntimeError('Image write failed.')
                print(f'Saved {args.output}: {frame.shape}, {frame.dtype}, OpenCV BGR')
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
