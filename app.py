from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import fitz  # PyMuPDF

app = Flask(__name__)

SCALE_MATRIX = fitz.Matrix(1.3, 1.3)
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 75]

# ==========================================
# 座標計算ユーティリティ関数
# ==========================================

def get_calc_pos(q, w, h, sy, step):
    if 1 <= q <= 10:
        cx = w * 0.305; cy = h * (sy + (q - 1) * step)
    elif 11 <= q <= 20:
        cx = w * 0.595; cy = h * (sy + (q - 11) * step)
    else:
        cx = w * 0.89; cy = h * (sy + (q - 21) * step)
    return int(cx), int(cy)

def draw_check(img, cx, cy, w, color, thickness):
    size = int(w * 0.015)
    pt1 = (cx - int(size * 0.8), cy)
    pt2 = (cx - int(size * 0.2), cy + size)
    pt3 = (cx + size, cy - size)
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt2, pt3, color, thickness)

def get_crop_box(mode, q_num, w, h):
    """
    串刺し採点用の切り抜き座標(x1, y1, x2, y2)を返す
    """
    if mode in ['kanji', 'yojijukugo']:
        start_x, end_x = 0.10, 0.89
        start_y, end_y = 0.14, 0.92
        idx = q_num - 1
        row, col = idx // 10, idx % 10
        cx = int(w * (end_x - (col * (end_x - start_x) / 10.0) - ((end_x - start_x) / 20.0)))
        cy = int(h * (start_y + (row * (end_y - start_y) / 5.0) + ((end_y - start_y) / 25.0)))
        return cx - int(w * 0.04), cy - int(h * 0.06), cx + int(w * 0.04), cy + int(h * 0.06)

    elif mode == 'calc_contest':
        sy, step = 0.215, 0.0606
        cx, cy = get_calc_pos(q_num, w, h, sy, step)
        return cx - int(w * 0.18), cy - int(h * 0.035), cx + int(w * 0.05), cy + int(h * 0.035)

    elif mode == 'calc_test':
        sy, step = 0.3, 0.0606
        cx = int(w * 0.85)
        cy = int(h * (sy + (q_num - 1) * step))
        return cx - int(w * 0.22), cy - int(h * 0.04), cx + int(w * 0.08), cy + int(h * 0.04)

    elif mode == 'pref':
        # 都道府県コンテスト (94問)
        if q_num < 1 or q_num > 94:
            return 0, 0, w, h

        is_right_col = False
        row_idx = 0
        
        if q_num <= 47:
            if q_num <= 24:
                row_idx = q_num - 1
            else:
                is_right_col = True
                row_idx = q_num - 25
        else:
            base_q = q_num - 47
            if base_q <= 24:
                row_idx = base_q - 1
            else:
                is_right_col = True
                row_idx = base_q - 25

        # ★ 描写位置（マルバツの中心）と完全に同じY座標の計算式
        cy = int(h * (0.245 + row_idx * 0.02725))
        
        # ★ 描写位置（マルバツの中心）と完全に同じX座標の計算式
        if not is_right_col:
            cx = int(w * 0.22) if q_num <= 47 else int(w * 0.395)
        else:
            cx = int(w * 0.66) if q_num <= 47 else int(w * 0.835)

        # 縦幅の計算（1行分の高さ 約0.0271 の 1.2倍 = 約0.03252 → 上下 0.01626）
        half_h = int(h * 0.01626)
        y1 = cy - half_h
        y2 = cy + half_h

        # 横幅の計算（元の幅の 1.2倍）
        # 県名: 元幅 0.20 * 1.2 = 0.24 → 左右 0.12
        # 県庁: 元幅 0.15 * 1.2 = 0.18 → 左右 0.09
        if q_num <= 47:
            half_w = int(w * 0.12)
        else:
            half_w = int(w * 0.09)
        
        x1 = cx - half_w
        x2 = cx + half_w

        return x1, y1, x2, y2

    return 0, 0, w, h


# ==========================================
# API エンドポイント
# ==========================================

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
        if page_num >= total_pages: page_num = total_pages - 1
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=SCALE_MATRIX, alpha=False, colorspace=fitz.csRGB)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        total_pages = 1
        page_num = 0

    _, buffer = cv2.imencode('.jpg', img, JPEG_QUALITY)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'status': 'success', 'image': 'data:image/jpeg;base64,' + result_b64, 'total_pages': total_pages, 'page_num': page_num})


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
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=SCALE_MATRIX, alpha=False, colorspace=fitz.csRGB)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            
            x1, y1, x2, y2 = get_crop_box(mode, q_num, w, h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            crop_img = img[y1:y2, x1:x2]
            _, buffer = cv2.imencode('.jpg', crop_img, JPEG_QUALITY)
            cropped_images.append({'page': page_num, 'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')})
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        
        x1, y1, x2, y2 = get_crop_box(mode, q_num, w, h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        crop_img = img[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.jpg', crop_img, JPEG_QUALITY)
        cropped_images.append({'page': 0, 'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')})

    return jsonify({'status': 'success', 'crops': cropped_images})


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
        if page_num >= doc.page_count: page_num = doc.page_count - 1
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=SCALE_MATRIX, alpha=False, colorspace=fitz.csRGB)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    red = (0, 0, 255)
    thickness = max(3, int(w * 0.003))

    score = 0
    score_pos = (int(w * 0.8), int(h * 0.1))
    font_scale = max(1.8, w * 0.002)

    # ==========================================
    # 画像へのマルバツ・得点描画処理
    # ==========================================
    if mode in ['kanji', 'yojijukugo']:
        start_x, end_x = 0.10, 0.89
        start_y, end_y = 0.14, 0.92
        score = 100 - (len(wrong_numbers) * 2)
        score_pos = (int(w * 0.76), int(h * 0.085))
        font_scale = max(1.8, w * 0.0018)
        
        for q in range(1, 51):
            idx = q - 1
            row, col = idx // 10, idx % 10
            cx = int(w * (end_x - (col * (end_x - start_x) / 10.0) - ((end_x - start_x) / 20.0)))
            cy = int(h * (start_y + (row * (end_y - start_y) / 5.0) + ((end_y - start_y) / 25.0)))
            if q in wrong_numbers:
                draw_check(img, cx, cy, w, red, thickness)
            else:
                cv2.circle(img, (cx, cy), int(w * 0.018), red, thickness)

    elif mode == 'calc_contest':
        sy, step = 0.215, 0.0606
        score = 100 - (len(wrong_numbers) * 4)
        score_pos = (int(w * 0.87), int(h * 0.17))
        font_scale = max(2, w * 0.0025)
        
        for q in range(1, 26):
            cx, cy = get_calc_pos(q, w, h, sy, step)
            if q in wrong_numbers:
                draw_check(img, cx, cy, w, red, thickness)
            else:
                cv2.circle(img, (cx, cy), int(w * 0.015), red, thickness)

    elif mode == 'calc_test':
        sy, step = 0.275, 0.0606
        score = 100 - (len(wrong_numbers) * 20)
        score_pos = (int(w * 0.87), int(h * 0.17))
        font_scale = max(1.6, w * 0.0022)
        
        for q in range(1, 6):
            cx = int(w * 0.80) 
            cy = int(h * (sy + (q - 1) * step))
            if q in wrong_numbers:
                draw_check(img, cx, cy, w, red, thickness)
            else:
                cv2.circle(img, (cx, cy), int(w * 0.015), red, thickness)

    elif mode == 'pref':
        # 都道府県コンテスト（94点満点）
        score = 94 - len(wrong_numbers)
        score_pos = (int(w * 0.83), int(h * 0.16))
        font_scale = max(1.6, w * 0.00175)
        
        for q in range(1, 95):
            is_right_col = False
            row_idx = 0
            
            if q <= 47:
                if q <= 24:
                    row_idx = q - 1
                else:
                    is_right_col = True
                    row_idx = q - 25
            else:
                base_q = q - 47
                if base_q <= 24:
                    row_idx = base_q - 1
                else:
                    is_right_col = True
                    row_idx = base_q - 25
            
            # ★ 描画のY座標を1行分(約0.027)下に修正
            cy = int(h * (0.249 + row_idx * 0.0271))
            
            if not is_right_col:
                cx = int(w * 0.25) if q <= 47 else int(w * 0.425)
            else:
                cx = int(w * 0.69) if q <= 47 else int(w * 0.865)
            
            if q in wrong_numbers:
                draw_check(img, cx, cy, w, red, thickness)
            else:
                cv2.circle(img, (cx, cy), int(w * 0.012), red, max(2, thickness - 1))

    # ==========================================
    # 最終的な得点を描画してエンコード・返却
    # ==========================================
    cv2.putText(img, f"{score}", score_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, red, thickness + 2)

    _, buffer = cv2.imencode('.jpg', img, JPEG_QUALITY)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'status': 'success', 
        'image': 'data:image/jpeg;base64,' + result_b64, 
        'score': score
    })


if __name__ == '__main__':
    # サーバーの起動
    app.run(host='0.0.0.0', port=8080)
