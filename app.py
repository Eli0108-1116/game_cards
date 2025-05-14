import os
import cv2
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template_string
from flask import Response
import tempfile
import re
import html

app = Flask(__name__, static_url_path='/', static_folder='.')

# --- 設定區 ---
BASE_IMG      = r""
BASE_DIR      = r"../ygodb/class"
MIN_MATCH_CNT = 10
MAX_WORKERS   = min(8, os.cpu_count() or 4)

CATEGORIES = ["同步", "超量", "融合", "連結", "魔法", "陷阱", "效果", "通常", "儀式", "靈擺"]

# --- 2. 快取管理 ---
def extract(path,sift):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None
    kp, des = sift.detectAndCompute(img, None)
    if des is None or len(des) == 0: return None
    attrs = np.array([[p.pt[0], p.pt[1], p.size, p.angle] for p in kp], dtype=np.float32)
    return (path, attrs, des)

def load_or_build_cache(category):
    GALLERY_DIR = os.path.join(BASE_DIR, category)
    CACHE_FILE  = os.path.join(BASE_DIR, f"{category}.npz")
    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        npz = np.load(CACHE_FILE, allow_pickle=True)
        paths    = npz['paths'].tolist()
        names    = npz['names'].tolist()
        kp_attrs = npz['kp_attrs']
        descs    = [npz[f'des{i}'] for i in range(len(names))]
        print(f"載入快取：{category} 類別共 {len(names)} 張影像")
        return paths, names, kp_attrs, descs

    files = [os.path.join(GALLERY_DIR, f) for f in os.listdir(GALLERY_DIR)
             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    paths, names, kp_attrs, descs = [], [], [], []
    sift = cv2.SIFT_create()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(extract, p, sift): p for p in files}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                path, attrs, des = res
                paths.append(path)
                names.append(os.path.basename(path))
                kp_attrs.append(attrs)
                descs.append(des)

    savez = {'paths': np.array(paths), 'names': np.array(names), 'kp_attrs': np.array(kp_attrs, dtype=object)}
    for i, des in enumerate(descs):
        savez[f'des{i}'] = des
    np.savez_compressed(CACHE_FILE, **savez)
    print(f"重建快取：{category} 類別共 {len(names)} 張影像")
    return paths, names, kp_attrs, descs

# paths, names, kp_attrs_list, descriptors_list = load_or_build_cache()

# --- 5. 顯示結果 ---
INFO_DIR = r"../ygodb/cards_info_selenium_firefox"


def process_image(category, img_data):
    # --- 1. 載入基準影像 & SIFT 特徵 ---
    # base_img_path = BASE_IMG
    # img1 = cv2.imread(base_img_path, cv2.IMREAD_COLOR)
    img1 = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    if img1 is None:
        raise FileNotFoundError(f"無法載入基準影像")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    d = des1.shape[1]

    paths, names, kp_attrs_list, descriptors_list = load_or_build_cache(category)

    # --- 3. 建立 FAISS 索引 ---
    quantizer = faiss.IndexFlatL2(d)
    nlist, m_pq = 100, 8
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m_pq, 8)
    all_desc = np.vstack(descriptors_list).astype('float32')
    index.train(all_desc)
    index.add(all_desc)

    # --- 4. 查詢與比對 ---
    D, I = index.search(des1.astype('float32'), 2)
    good_per_img = [[] for _ in descriptors_list]
    boundaries = np.cumsum([len(des) for des in descriptors_list])
    for qi in range(len(des1)):
        d0, d1 = D[qi]
        if d0 < 0.75 * d1:
            tr = int(I[qi, 0])
            idx = np.searchsorted(boundaries, tr, side='right')
            start = boundaries[idx - 1] if idx > 0 else 0
            local = tr - start
            good_per_img[idx].append(cv2.DMatch(qi, local, d0))

    # --- 5. 顯示結果 ---
    INFO_DIR = r"../ygodb/cards_info_selenium_firefox"
    result_text = ""

    for idx, matches in enumerate(good_per_img):

        if len(matches) < MIN_MATCH_CNT:
            continue

        # 取得卡片 ID (前8碼)
        basename = names[idx]
        card_id = basename[:8]

        # 搜尋符合 ID 的 .txt 檔
        info_files = [f for f in os.listdir(INFO_DIR)
                      if f.startswith(card_id) and f.lower().endswith(".txt")]

        if info_files:
            info_path = os.path.join(INFO_DIR, info_files[0])
        with open(info_path, "r", encoding="utf-8") as f:
            info = f.read()
        info = info.replace('圖片 URL:', '')
        # 確保 URL 格式正確，並轉義 HTML 標籤
        url_pattern = r'(https?://[^\s]+)'
        
        # 先將除了換行符號以外的 HTML 標籤轉義
        escaped_info = html.escape(info, quote=False)
        
        # 再將換行符號轉換成 <br> 標籤
        br_info = escaped_info.replace('\n', ' <br>')
        # 再將 URL 轉換成圖片
        img_info = re.sub(
            url_pattern,
            r'<img src="\1" alt="圖片" />',
            br_info
        )

        result_text += f"{img_info}"
    
    return result_text

@app.route('/match', methods=['POST'])
def match():
    category = request.form.get("category")
    file = request.files.get("baseImg")
    if not file or not category:
        return Response("<p>缺少資料</p>", status=400, mimetype='text/html; charset=utf-8')

    # 讀取上傳檔案的內容
    img_data = file.read()
    # 執行比對
    try:
        result_html = process_image(category, img_data)
    except Exception as e:
        # 捕捉例外並回傳錯誤訊息
        err_msg = f"<p>處理錯誤：{str(e)}</p>"
        return Response(err_msg, status=500, mimetype='text/html; charset=utf-8')

    # 確保回傳的是 HTML
    return Response(result_html, mimetype='text/html; charset=utf-8')

@app.route('/')
def index():
    return render_template_string(open('index.html', encoding='utf-8').read())

if __name__ == '__main__':
    app.run(debug=True)
