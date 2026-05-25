"""
智能水印消除工具 - Web界面
兼容 Windows/Linux 云端部署
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import io
import base64
import json
from PIL import Image
from watermark_remover import SmartWatermarkRemover

# 兼容两种文件结构：templates/index.html 或根目录 index.html
template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
if os.path.exists(os.path.join(template_path, 'index.html')):
    app = Flask(__name__, template_folder='templates')
else:
    app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

remover = SmartWatermarkRemover()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image, format='PNG'):
    """将OpenCV图像转换为base64"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def read_image(file):
    """从上传文件读取图像"""
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/remove', methods=['POST'])
def remove_watermark():
    """消除水印API - 支持两种模式：手动框选和自动检测"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if not file or not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400

    try:
        image = read_image(file)
        if image is None:
            return jsonify({'error': '无法读取图像'}), 400

        # 获取模式
        mode = request.form.get('mode', 'manual')

        if mode == 'auto':
            result = remover.process_auto(image)

        elif mode == 'manual':
            regions_data = request.form.get('regions', '[]')
            regions = json.loads(regions_data)
            regions = [(r[0], r[1], r[2], r[3]) for r in regions]
            result = remover.process_manual(image, regions)
        else:
            return jsonify({'error': '未知模式'}), 400

        # 构建响应
        response = {
            'success': result['success'],
            'watermark_found': result.get('watermark_found', False),
            'message': result.get('message', ''),
        }

        if result.get('preview') is not None:
            response['preview'] = image_to_base64(result['preview'])

        if result.get('image') is not None:
            response['result'] = image_to_base64(result['image'])

        if result.get('regions'):
            response['regions'] = [(int(x), int(y), int(w), int(h)) for x, y, w, h in result['regions']]

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
