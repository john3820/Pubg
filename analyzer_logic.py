import os, sys, base64, json, math
import google.generativeai as genai
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

# ── 설정 ─────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-2.0-flash" 
FRAME_WIDTH    = 960
NUM_FRAMES     = 12
CURRENT_VMULT  = 1.0

# 색상 테마
BG    = "#070b10"; PANEL = "#0c1520"; ORA   = "#f0a500"
GRN   = "#39d87a"; RED   = "#ff4d4d"; BLU   = "#4a9eff"
TXT   = "#c8d8e8"; TXT2  = "#4a6a8f"; GRID  = "#111d2a"

def _pick_gemini_model_name():
    return os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

def _pick_korean_font_family():
    preferred = ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic", "Noto Sans KR"]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in preferred:
            if name in available: return name
    except: pass
    return "DejaVu Sans"

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
    shot_frames, last_shot_at = [], -50
    for idx in range(0, total, 4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) > 18 and (idx - last_shot_at) > 20:
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
    prompt = f"""이 이미지는 PUBG 사격 프레임입니다. 발사 직후 조준점 이동을 분석하세요.
    JSON 형식 응답: {{ "shot_trajectory": [ {{"shot":1,"x":0,"y":0}}, ... ], "vertical_drift_px": 수치, "horizontal_drift_px": 수치, "drift_direction_v": "상단이탈"|"하단이탈", "recoil_pattern": "vertical_heavy"|"stable", "recommended_vertical_mult": 수치, "confidence": "high", "note": "설명" }}"""
    resp = model.generate_content([prompt] + images)
    try:
        raw = resp.text.replace("```json","").replace("```","").strip()
        result = json.loads(raw)
        if "shot_trajectory" in result:
            result["frame_trajectory"] = [{"frame": s["shot"]-1, "x": s["x"], "y": s["y"]} for s in result["shot_trajectory"]]
        return result
    except: return None

def generate_report(model, result):
    prompt = f"반동 분석 데이터: {result}. 이 데이터를 바탕으로 PUBG 파츠 추천과 총평을 한국어 JSON으로 작성하세요."
    resp = model.generate_content(prompt)
    try:
        return json.loads(resp.text.replace("```json","").replace("```","").strip())
    except: return {"diagnosis": "AI 진단 생성 실패", "parts": []}

def draw_combined_graph(p1, p2_result, p2_report, save_path):
    font_name = _pick_korean_font_family()
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False, "text.color": TXT})
    
    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    gs = GridSpec(2, 3, figure=fig)
    
    # 탄착 궤적 그래프 (예시로 하나만 구현)
    ax = fig.add_subplot(gs[0, 0], facecolor=PANEL)
    ax.set_title("반동 궤적 분석", color=ORA)
    if p2_result:
        traj = p2_result.get("frame_trajectory", [])
        xs = [p["x"] for p in traj]
        ys = [-p["y"] for p in traj]
        ax.plot(xs, ys, marker='o', color=ORA)
        ax.scatter([0], [0], color=GRN, s=100, marker='*')
    
    plt.savefig(save_path, facecolor=BG)
    plt.close()
