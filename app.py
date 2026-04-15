from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import fitz  # PyMuPDF

app = Flask(__name__)

def process_pdf_to_image(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

@app.route('/grade', methods=['POST'])
def grade():
    data = request.json
    file_b64 = data.get('file')
    wrong_numbers = data.get('wrong_numbers', [])
    mode = data.get('mode', 'kanji') # モードを取得

    header, encoded = file_b64.split(",", 1)
    file_bytes = base64.b64decode(encoded)
    
    if "pdf" in header:
        img = process_pdf_to_image(file_bytes)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    red = (0, 0, 255)

    # --- モード別の設定 ---
    if mode == 'kanji':
        # 漢字テスト (50問)
        start_x, end_x = 0.10, 0.89
        start_y, end_y = 0.14, 0.92
        score = 100 - (len(wrong_numbers) * 2)
        for q in range(1, 51):
            idx = q - 1
            row, col = idx // 10, idx % 10
            cx = int(w * (end_x - (col * (end_x - start_x) / 10.0) - ((end_x - start_x) / 20.0)))
            cy = int(h * (start_y + (row * (end_y - start_y) / 5.0) + ((end_y - start_y) / 25.0)))
            if q in wrong_numbers:
                draw_check(img, cx, cy, w, red)
            else:
                cv2.circle(img, (cx, cy), int(w * 0.018), red, 4)

    elif mode == 'calc_contest':
        # 計算コンテスト (25問)
        sy, step = 0.215, 0.0606
        score = 100 - (len(wrong_numbers) * 4)
        for q in range(1, 26):
            cx, cy = get_calc_pos(q, w, h, sy, step)
            if q in wrong_numbers:
                draw_check(img, cx, cy, w, red)
            else:
                cv2.circle(img, (cx, cy), int(w * 0.015), red, 4)

    elif mode == 'calc_test':
        # 計算テスト (5問版) - 1問20点
        sy, step = 0.215, 0.0606
        score = 100 - (len(wrong_numbers) * 20)
        for q in range(1, 6):
            # 5問版は座標としては(1)〜(5)だが、位置は25問版の(21)〜(25)と同じ右端
            cx = int(w * 0.89)
            cy = int(h * (sy + (q - 1) * step))
            if q in wrong_numbers:
                draw_check(img, cx, cy, w, red)
            else:
                cv2.circle(img, (cx, cy), int(w * 0.015), red, 4)

    # スコア描画
    cv2.putText(img, f"{score}", (int(w * 0.85), int(h * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, red, 6)

    _, buffer = cv2.imencode('.jpg', img)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'status': 'success', 'image': 'data:image/jpeg;base64,' + result_b64, 'score': score})

def get_calc_pos(q, w, h, sy, step):
    if 1 <= q <= 10: cx = w * 0.305; cy = h * (sy + (q - 1) * step)
    elif 11 <= q <= 20: cx = w * 0.595; cy = h * (sy + (q - 11) * step)
    else: cx = w * 0.89; cy = h * (sy + (q - 21) * step)
    return int(cx), int(cy)

def draw_check(img, cx, cy, w, color):
    size = int(w * 0.015)
    pt1 = (cx - int(size * 0.8), cy); pt2 = (cx - int(size * 0.2), cy + size); pt3 = (cx + size, cy - size)
    cv2.line(img, pt1, pt2, color, 5); cv2.line(img, pt2, pt3, color, 5)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
