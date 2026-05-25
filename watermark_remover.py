"""
水印检测与消除 - 支持手动框选和自动检测
兼容 Windows/Linux 云端部署
"""

import cv2
import numpy as np
from typing import List, Tuple
import os
import subprocess

# Tesseract OCR 配置（兼容 Windows/Linux）
_tesseract_available = False
try:
    import pytesseract

    # 自动检测Tesseract路径
    if os.name == 'nt':  # Windows
        _win_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for p in _win_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    # Linux/Mac: tesseract 通常在 PATH 中，无需额外配置

    # 验证Tesseract是否可用
    try:
        pytesseract.get_tesseract_version()
        _tesseract_available = True
    except:
        _tesseract_available = False
except ImportError:
    _tesseract_available = False


def _ocr(image_gray, lang='chi_sim+eng'):
    """OCR识别文字"""
    if not _tesseract_available:
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': [], 'line_num': []}
    try:
        return pytesseract.image_to_data(image_gray, lang=lang, output_type=pytesseract.Output.DICT)
    except:
        try:
            return pytesseract.image_to_data(image_gray, lang='eng', output_type=pytesseract.Output.DICT)
        except:
            return {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': [], 'line_num': []}


class AutoWatermarkDetector:
    """自动水印检测器 - 全图OCR + 边缘检测"""

    def detect(self, image):
        regions = []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 图像预处理
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 全图OCR检测文字
        try:
            data = _ocr(binary, 'chi_sim+eng')
            lines = {}
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if conf < 15 or not text:
                    continue
                ln = data['line_num'][i]
                if ln not in lines:
                    lines[ln] = []
                lines[ln].append({
                    'text': text, 'x': data['left'][i], 'y': data['top'][i],
                    'w': data['width'][i], 'h': data['height'][i],
                })

            for ln, words in lines.items():
                min_x = min(wd['x'] for wd in words)
                min_y = min(wd['y'] for wd in words)
                max_x = max(wd['x'] + wd['w'] for wd in words)
                max_y = max(wd['y'] + wd['h'] for wd in words)
                rw, rh = max_x - min_x, max_y - min_y
                if rw < 20 or rh < 8 or rw > w * 0.5 or rh > h * 0.3:
                    continue
                regions.append((min_x, min_y, rw, rh))
        except:
            pass

        # 边缘检测找Logo（角落区域）
        corners = [
            (0, 0, int(w*0.20), int(h*0.20)),
            (int(w*0.80), 0, w, int(h*0.20)),
            (0, int(h*0.80), int(w*0.20), h),
            (int(w*0.80), int(h*0.80), w, h),
        ]
        for cx1, cy1, cx2, cy2 in corners:
            roi = gray[cy1:cy2, cx1:cx2]
            if roi.size == 0:
                continue
            edges = cv2.Canny(roi, 40, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if area < 80 or area > 5000:
                    continue
                fill = area / (cw*ch) if cw*ch > 0 else 0
                if fill < 0.15:
                    continue
                regions.append((cx1+x, cy1+y, cw, ch))

        return self._merge(regions)[:10]

    def _merge(self, regions):
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: r[2]*r[3], reverse=True)
        merged = []
        for r in regions:
            x1, y1, w1, h1 = r
            m = False
            for i, (mx, my, mw, mh) in enumerate(merged):
                xi, yi = max(x1, mx), max(y1, my)
                wi, hi = min(x1+w1, mx+mw)-xi, min(y1+h1, my+mh)-yi
                if wi > 0 and hi > 0:
                    iou = wi*hi / (w1*h1 + mw*mh - wi*hi)
                    if iou > 0.3:
                        merged[i] = (min(x1,mx), min(y1,my), max(x1+w1,mx+mw)-min(x1,mx), max(y1+h1,my+mh)-min(y1,my))
                        m = True
                        break
            if not m:
                merged.append(r)
        return merged


class WatermarkInpainter:
    """水印消除器"""
    @staticmethod
    def remove(image, regions):
        if not regions:
            return image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for x, y, w, h in regions:
            mask[max(0,y):min(image.shape[0],y+h), max(0,x):min(image.shape[1],x+w)] = 255
        # 大幅膨胀确保消除完整
        mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=3)
        return cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)


class SmartWatermarkRemover:
    def __init__(self):
        self.auto_detector = AutoWatermarkDetector()
        self.inpainter = WatermarkInpainter()

    def process_auto(self, image):
        regions = self.auto_detector.detect(image)
        if not regions:
            return {'success': True, 'watermark_found': False, 'message': '未检测到水印'}
        result_image = self.inpainter.remove(image, regions)
        preview = image.copy()
        for x, y, w, h in regions:
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return {'success': True, 'watermark_found': True, 'regions': regions,
                'image': result_image, 'preview': preview,
                'message': f'检测到 {len(regions)} 个水印区域并已消除'}

    def process_manual(self, image, regions):
        if not regions:
            return {'success': False, 'watermark_found': False, 'message': '没有选择区域'}
        result_image = self.inpainter.remove(image, regions)
        preview = image.copy()
        for x, y, w, h in regions:
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return {'success': True, 'watermark_found': True, 'regions': regions,
                'image': result_image, 'preview': preview,
                'message': f'已消除 {len(regions)} 个选中区域'}
