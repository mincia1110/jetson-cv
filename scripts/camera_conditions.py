"""test.py-compatible image checks; no model dependencies."""
import math

# Command data from miked63017/dinolite_uvc_led_control, Notes.txt (5MP).
# AM7115MZT: AE/LED verified by user; fixed-time commands need hardware validation.
EXPOSURE_COMMANDS = {
    '1/1000s': '05000001357810', '1/500s': '05010001357810',
    '1/250s': '05020001357810', '1/125s': '05040001357810',
    '1/60s': '05080001357810', '1/30s': '05100001357810',
    '1/15s': '0500000d387810', '1/8s': '051f0001357810',
    '1/4s': '0508000c387810', '1/2s': '053e0001357810',
    '1s': '05510035307810', '2s': '050d000c387810',
    '4s': '05610035307810', '8s': '050f000c387810', '16s': '0510000c387810',
}


def validate_conditions(config):
    result = dict(config)
    for key in ('BRIGHT_min', 'BRIGHT_max', 'RG_gab'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key}: finite number required')
    if not 0 <= result['BRIGHT_min'] <= result['BRIGHT_max'] <= 255:
        raise ValueError('Require 0 <= BRIGHT_min <= BRIGHT_max <= 255')
    for key, default, low, high in (('capture_no', 5, 1, 100),
                                   ('settle_frames', 5, 1, 300)):
        value = result.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{key}: integer {low}..{high} required')
        result[key] = value
    if type(result.get('reset_flag_en', False)) is not bool:
        raise ValueError('reset_flag_en: true or false required')
    result.setdefault('reset_flag_en', False)
    if type(result.get('Brightness')) is not int:
        raise ValueError('초기 Brightness를 Linux V4L2 정수로 입력하세요.')
    if result.get('ExposureTime') not in EXPOSURE_COMMANDS:
        raise ValueError(f"고정 ExposureTime을 선택하세요. 읽은 값={result.get('ExposureTime')!r}. "
                         '예: "ExposureTime": "1/60s". DLL ExposureValue의 숫자는 자동 변환하지 않습니다.')
    if result.get('exposure_reset_mode', 'fixed') != 'fixed':
        raise ValueError('AE 재적응 설정을 제거하세요. 현재 모드는 fixed입니다.')
    return result


def assess_frame(frame, config):
    import cv2
    if frame.shape[:2] != (1944, 2592):
        raise ValueError('test.py checks require a 2592x1944 frame')
    # test.py reloads BMP as RGB, blurs full image, then samples its bottom 10 rows.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.GaussianBlur(rgb, (7, 7), 0)
    r, g, b, _ = cv2.mean(rgb[1934:1944, 0:2590])
    return assess_rgb_means(r, g, b, config)


def assess_rgb_means(r, g, b, config):
    bright = int(r)  # Original variable bright_b actually holds RED.
    rg = int(r - g)
    reasons = []
    if not config['BRIGHT_min'] <= bright <= config['BRIGHT_max']:
        reasons.append('BRIGHT error')
    if rg > config['RG_gab']:
        reasons.append('RG_gab error')
    return {'R': r, 'G': g, 'B': b, 'bright_b_actual_R': bright,
            'RG_diff': rg, 'reasons': reasons}
