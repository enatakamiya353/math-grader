from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import fitz  # PyMuPDF
import io

app = Flask(__name__)

def process_pdf_to_image(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    # A3サイズなので解像度を高めに設定
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

@app.route('/grade', methods=['POST'])
def grade():
    data = request.json
    file_b64 = data.get('file')
    wrong_numbers = data.get('wrong_numbers', [])

    header, encoded = file_b64.split(",", 1)
    file_bytes = base64.b64decode(encoded)
    
    if "pdf" in header:
        img = process_pdf_to_image(file_bytes)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
# 【A3・3列レイアウトの「新・新」座標設定】
    start_y = 0.215  # 縦の高さは前回で完璧です！
    step_y = 0.0606

    def get_pos(q_num):
        if 1 <= q_num <= 10:       # 左の列
            cx = int(w * 0.305)    # ★修正: 0.205 から 0.305 (右へシフト)
            cy = int(h * (start_y + (q_num - 1) * step_y))
        elif 11 <= q_num <= 20:    # 真ん中の列
            cx = int(w * 0.595)    # ★修正: 0.525 から 0.595 (右へシフト)
            cy = int(h * (start_y + (q_num - 11) * step_y))
        elif 21 <= q_num <= 25:    # 右の列
            cx = int(w * 0.89)     # ★修正: 0.845 から 0.890 (右へシフト)
            cy = int(h * (start_y + (q_num - 21) * step_y))
        else:
            return 0, 0
        return cx, cy
    
    # 25問分の描画ループ
    for q in range(1, 26):
        cx, cy = get_pos(q)
        if q in wrong_numbers:
            size = int(w * 0.015) 
            thickness = 5
            pt1 = (cx - int(size * 0.8), cy)
            pt2 = (cx - int(size * 0.2), cy + size)
            pt3 = (cx + size, cy - size)
            cv2.line(img, pt1, pt2, red, thickness)
            cv2.line(img, pt2, pt3, red, thickness)
        else:
            radius = int(w * 0.015)
            cv2.circle(img, (cx, cy), radius, red, 4)

    # 1問4点で減点計算
    score = 100 - (len(wrong_numbers) * 4)
    cv2.putText(img, f"{score}", (int(w * 0.85), int(h * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, red, 6)

    _, buffer = cv2.imencode('.jpg', img)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'status': 'success', 
        'image': 'data:image/jpeg;base64,' + result_b64,
        'score': score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
