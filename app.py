from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import fitz  # PyMuPDF

app = Flask(__name__)

# ★既存：PDFのプレビュー用エンドポイント（指定ページだけを画像で返す）
@app.route('/preview', methods=['POST'])
def preview():
    data = request.json
    file_b64 = data.get('file')
    page_num = data.get('page_num', 0)

    header, encoded = file_b64.split(",", 1)
    file_bytes = base64.b64decode(encoded)
    
    if "pdf" in header:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = doc.page_count
        if page_num >= total_pages:
            page_num = total_pages - 1
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        total_pages = 1
        page_num = 0

    _, buffer = cv2.imencode('.jpg', img)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'status': 'success', 
        'image': 'data:image/jpeg;base64,' + result_b64,
        'total_pages': total_pages,
        'page_num': page_num
    })

# ★新規追加：串刺し採点（指定した問題番号の解答欄だけを全ページ分切り出す）
@app.route('/skewer', methods=['POST'])
def skewer():
    data = request.json
    file_b64 = data.get('file')
    mode = data.get('mode', 'kanji')
    q_num = int(data.get('q_num', 1))

    header, encoded = file_b64.split(",", 1)
    file_bytes = base64.b64decode(encoded)
    
    cropped_images = []

    if "pdf" in header:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        # PDFの全ページをループして切り出す
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            h, w = img.shape[:2]
            x1, y1, x2, y2 = get_crop_box(mode, q_num, w, h)
            
            # 座標が画像サイズをはみ出さないように補正
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = img[y1:y2, x1:x2]
            
            _, buffer = cv2.imencode('.jpg', crop_img)
            cropped_images.append({
                'page': page_num,
                'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
            })
    else:
        # 画像ファイルが1枚だけ送られてきた場合の処理
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = get_crop_box(mode, q_num, w, h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_img = img[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.jpg', crop_img)
        cropped_images.append({
            'page': 0,
            'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
        })

    return jsonify({'status': 'success', 'crops': cropped_images})

# ★既存：採点・描画用エンドポイント
@app.route('/grade', methods=['POST'])
def grade():
    data = request.json
    file_b64 = data.get('file')
    wrong_numbers = data.get('wrong_numbers', [])
    mode = data.get('mode', 'kanji')
    page_num = data.get('page_num', 0)

    header, encoded = file_b64.split(",", 1)
    file_bytes = base64.b64decode(encoded)
    
    if "pdf" in header:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if page_num >= doc.page_count:
            page_num = doc.page_count - 1
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    red = (0, 0, 255)

    if mode == 'kanji':
        start_x, end_x = 0.10, 0.89
        start_y, end_y = 0.14, 0.92
        score = 100 - (len(wrong_numbers) * 2)
        for q in range(1, 51):
            idx = q - 1
            row, col = idx // 10, idx % 10
            cx = int(w * (end_x - (col * (end_x - start_x) / 10.0) - ((end_x - start_x) / 20.0)))
            cy = int(h * (start_y + (row * (end_y - start_y) / 5.0) + ((end_y - start_y) / 25.0)))
            if q in wrong_numbers: draw_check(img, cx, cy, w, red)
            else: cv2.circle(img, (cx, cy), int(w * 0.018), red, 4)

    elif mode == 'calc_contest':
        sy, step = 0.215, 0.0606
        score = 100 - (len(wrong_numbers) * 4)
        for q in range(1, 26):
            cx, cy = get_calc_pos(q, w, h, sy, step)
            if q in wrong_numbers: draw_check(img, cx, cy, w, red)
            else: cv2.circle(img, (cx, cy), int(w * 0.015), red, 4)

    elif mode == 'calc_test':
        sy, step = 0.3, 0.0606
        score = 100 - (len(wrong_numbers) * 20)
        for q in range(1, 6):
            cx = int(w * 0.85) 
            cy = int(h * (sy + (q - 1) * step))
            if q in wrong_numbers: draw_check(img, cx, cy, w, red)
            else: cv2.circle(img, (cx, cy), int(w * 0.015), red, 4)

    cv2.putText(img, f"{score}", (int(w * 0.9), int(h * 0.20)), cv2.FONT_HERSHEY_SIMPLEX, 3.5, red, 6)

    _, buffer = cv2.imencode('.jpg', img)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'status': 'success', 'image': 'data:image/jpeg;base64,' + result_b64, 'score': score})

# --- サポート関数群 ---

def get_calc_pos(q, w, h, sy, step):
    if 1 <= q <= 10: cx = w * 0.305; cy = h * (sy + (q - 1) * step)
    elif 11 <= q <= 20: cx = w * 0.595; cy = h * (sy + (q - 11) * step)
    else: cx = w * 0.89; cy = h * (sy + (q - 21) * step)
    return int(cx), int(cy)

def draw_check(img, cx, cy, w, color):
    size = int(w * 0.015)
    pt1 = (cx - int(size * 0.8), cy); pt2 = (cx - int(size * 0.2), cy + size); pt3 = (cx + size, cy - size)
    cv2.line(img, pt1, pt2, color, 5); cv2.line(img, pt2, pt3, color, 5)

# ★新規追加：モードと問題番号から、切り抜くべき領域(Bounding Box)を計算する
def get_crop_box(mode, q_num, w, h):
    if mode == 'kanji':
        start_x, end_x = 0.10, 0.89
        start_y, end_y = 0.14, 0.92
        idx = q_num - 1
        row, col = idx // 10, idx % 10
        cx = int(w * (end_x - (col * (end_x - start_x) / 10.0) - ((end_x - start_x) / 20.0)))
        cy = int(h * (start_y + (row * (end_y - start_y) / 5.0) + ((end_y - start_y) / 25.0)))
        # 漢字の解答欄（縦長・少し幅狭）に合わせて切り抜き領域を計算
        return cx - int(w*0.04), cy - int(h*0.06), cx + int(w*0.04), cy + int(h*0.06)

    elif mode == 'calc_contest':
        sy, step = 0.215, 0.0606
        cx, cy = get_calc_pos(q_num, w, h, sy, step)
        # 計算の解答欄（横長・左側に余白あり）に合わせて切り抜き領域を計算
        return cx - int(w*0.18), cy - int(h*0.035), cx + int(w*0.05), cy + int(h*0.035)

    elif mode == 'calc_test':
        sy, step = 0.3, 0.0606
        cx = int(w * 0.85) 
        cy = int(h * (sy + (q_num - 1) * step))
        # 計算の解答欄（横長・左側に余白あり）に合わせて切り抜き領域を計算
        return cx - int(w*0.22), cy - int(h*0.04), cx + int(w*0.08), cy + int(h*0.04)
    
    return 0, 0, w, h

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
