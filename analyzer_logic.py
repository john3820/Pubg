import os, sys, base64, json, math
import google.generativeai as genai
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

# 설정값
GEMINI_MODEL   = "gemini-2.5-flash"
FRAME_WIDTH    = 960
NUM_FRAMES     = 12
CURRENT_VMULT  = 1.0

def _pick_gemini_model_name():
    return os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

def calc_rotation(end_deg, start_deg=0):
    diff = abs(end_deg - start_deg)
    if diff > 180: diff = 360 - diff
    return diff

def detect_muzzle_flash_frames(video_path, max_shots=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1 = int(w*0.375), int(h*0.375)
    x2, y2 = int(w*0.625), int(h*0.625)
    shot_frames, last_shot_at = [], -100
    for idx in range(0, total, 4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) > 18 and (idx - last_shot_at) > 24:
            shot_frames.append(idx + 1)
            last_shot_at = idx
            if len(shot_frames) >= max_shots: break
    cap.release()
    return shot_frames

def extract_frames(video_path, max_shots=15):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_h = int(h * FRAME_WIDTH / w)
    shot_indices = detect_muzzle_flash_frames(video_path, max_shots)
    if len(shot_indices) < 3:
        shot_indices = [int(total * i / 11) for i in range(12)]
    
    out = []
    for idx in shot_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total-1))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_WIDTH, new_h))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()
    return out, shot_indices

def analyze_recoil(model, frames_b64, shot_indices=None):
    import PIL.Image, io
    images = [PIL.Image.open(io.BytesIO(base64.b64decode(b))) for b in frames_b64]
    prompt = f"분석 항목: 1.발사별 조준점 위치 2.수직/수평 드리프트 3.패턴 4.추천 수직배수(현재 {CURRENT_VMULT}). JSON으로만 응답."
    resp = model.generate_content([prompt] + images)
    try:
        raw = resp.text.replace("```json","").replace("```","").strip()
        result = json.loads(raw)
        if "shot_trajectory" in result:
            result["frame_trajectory"] = [{"frame": s["shot"]-1, "x": s["x"], "y": s["y"]} for s in result["shot_trajectory"]]
        return result
    except: return None

def generate_report(model, result):
    prompt = f"반동 데이터: {result}. 파츠 추천과 진단을 한국어 JSON으로 작성."
    resp = model.generate_content(prompt)
    try:
        return json.loads(resp.text.replace("```json","").replace("```","").strip())
    except: return {"diagnosis": "분석 실패", "parts": []}

def draw_combined_graph(p1, p2_result, p2_report, save_path):
    # (그래프 그리는 코드는 사용자님의 원본과 동일하게 유지하거나 
    # Streamlit에서 직접 그릴 수 있으므로 여기서는 파일 저장 로직만 포함)
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "Analysis Graph", ha='center')
    plt.savefig(save_path)
    plt.close()
