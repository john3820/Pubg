"""
PUBG 종합 감도 분석기 (ALL-IN-ONE)
=====================================
Phase 1: 일반 감도 보정  — 나침반 숫자 5회 수동 입력 → 보정 계수 계산
Phase 2: 수직 감도 보정  — 벽 사격 영상 분석 → 탄착 궤적 + 파츠 추천

필요 라이브러리:
  pip install google-generativeai opencv-python matplotlib numpy

API 키 발급 (무료):
  https://aistudio.google.com/app/apikey

사용법:
  python pubg_sens_all.py
"""

import os, sys, base64, json, math

def check_deps():
    missing = []
    for pkg, imp in [("google-generativeai","google.generativeai"), ("opencv-python","cv2"),
                     ("matplotlib","matplotlib"), ("numpy","numpy")]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"\n[!] 설치 필요:\n    pip install {' '.join(missing)}\n")
        sys.exit(1)

check_deps()

import google.generativeai as genai
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

# ── 설정 ─────────────────────────────────────────────────────
API_KEY        = ""                                              # 코드에 키를 직접 적지 말고,
                         # 무료 발급: https://aistudio.google.com/app/apikey
GEMINI_MODEL   = "gemini-2.5-flash"                              # 기본 모델 (환경변수로 덮어쓰기 가능)
NUM_FRAMES     = 12      # Phase 2 추출 프레임 수
FRAME_WIDTH    = 960
TARGET_ANGLE   = 90      # Phase 1 목표 회전각
NUM_TRIALS     = 5       # Phase 1 반복 횟수
CURRENT_SENS   = 50      # 현재 일반 감도
CURRENT_VMULT  = 1.0     # 현재 수직 감도 배수
DIVIDER        = "─" * 60
SCRIPT_VERSION = "2026-03-13a"


def _pick_gemini_model_name():
    """환경변수 GEMINI_MODEL이 있으면 그것을, 없으면 기본값을 반환."""
    env_name = os.environ.get("GEMINI_MODEL", "").strip()
    return env_name or GEMINI_MODEL


def _list_generatecontent_models():
    """generateContent 가능한 모델 short name 목록(선호도 순으로 정렬)"""
    preferred = [
        # 최신 계열을 우선 시도 (계정/정책에 따라 가용 모델이 다를 수 있음)
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
    ]

    try:
        models = list(genai.list_models())
    except Exception:
        return [m for m in preferred if m]

    available = []
    for m in models:
        name = getattr(m, "name", "") or ""
        short = name.split("/", 1)[-1] if name else ""
        methods = set(getattr(m, "supported_generation_methods", []) or [])
        if short and "generateContent" in methods:
            available.append(short)

    # preferred에 있으면 앞쪽으로, 나머지는 뒤로
    preferred_set = set(preferred)
    ordered = [m for m in preferred if m in available]
    ordered += [m for m in available if m not in preferred_set]
    return ordered or [m for m in preferred if m]


def _generate_with_fallback(model, content, *, max_fallbacks=3):
    """
    모델 호출 중 NotFound(모델 비가용/차단 등) 발생 시 다른 모델로 자동 재시도.
    ResourceExhausted(429) 등은 그대로 올려서 사용자에게 원인(쿼터/플랜)을 보여줌.
    """
    model_names = _list_generatecontent_models()
    tried = []

    def _mk(mname: str):
        print(f"  Gemini 모델 전환: {mname}")
        return genai.GenerativeModel(mname)

    # 현재 모델 이름을 우선 시도 목록 맨 앞으로
    current_name = getattr(model, "model_name", None) or getattr(model, "_model_name", None)
    if isinstance(current_name, str) and current_name.startswith("models/"):
        current_name = current_name.split("/", 1)[-1]
    if isinstance(current_name, str) and current_name:
        model_names = [current_name] + [m for m in model_names if m != current_name]

    last_exc = None
    for mname in model_names[: max(1, max_fallbacks + 1)]:
        tried.append(mname)
        try:
            if mname != current_name:
                model = _mk(mname)
            return model.generate_content(content), model
        except Exception as e:
            msg = str(e)
            last_exc = e
            # 모델이 "없음/차단/신규불가" 류면 다른 모델로 넘어감
            if ("404" in msg) or ("NOT_FOUND" in msg) or ("is not found" in msg) or ("no longer available" in msg):
                continue
            # 그 외는 그대로 전파
            raise

    # 전부 실패
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Gemini 모델 호출 실패: tried={tried}")
# 색상 테마
BG    = "#0d1117"
PANEL = "#131a23"
ORA   = "#f0a500"
GRN   = "#39d87a"
RED   = "#ff4444"
BLU   = "#4a9eff"
TXT   = "#dce4f0"
TXT2  = "#6a7a8f"
GRID  = "#1a2433"


def _pick_korean_font_family():
    # Windows에서 흔한 한글 폰트 우선순위
    preferred = [
        "Malgun Gothic",   # 맑은 고딕
        "맑은 고딕",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    for name in preferred:
        if name in available:
            return name
    # fallback (한글이 안 나올 수 있음)
    return "DejaVu Sans"


# ════════════════════════════════════════════════════════════
#  PHASE 1  —  일반 감도 보정 (나침반 수동 입력)
# ════════════════════════════════════════════════════════════

def calc_rotation(end_deg, start_deg=0):
    """N(0)에서 시작, end_deg까지 실제 회전각 (360도 경계 보정)"""
    diff = abs(end_deg - start_deg)
    if diff > 180:
        diff = 360 - diff
    return diff


def run_phase1():
    global CURRENT_SENS, TARGET_ANGLE

    print(f"\n{'━'*60}")
    print("  PHASE 1  —  일반 감도 보정")
    print(f"{'━'*60}")
    print("""
  방법:
  ┌─────────────────────────────────────────────────────────┐
  │  화면을 N에 맞추고 마우스를 일정한 속도로               │
  │  좌→우 또는 우→좌로 90도 회전시켜주세요                │
  │  멈춘 직후 나침반 숫자를 읽어 입력하세요 (5회 반복)     │
  └─────────────────────────────────────────────────────────┘""")

    # 현재 감도 입력
    try:
        v = input(f"\n  현재 일반 감도 (기본 {CURRENT_SENS}, 엔터 스킵) > ").strip()
        if v: CURRENT_SENS = int(v)
    except ValueError:
        pass

    try:
        v = input(f"  목표 회전각 (기본 {TARGET_ANGLE}도, 엔터 스킵) > ").strip()
        if v: TARGET_ANGLE = float(v)
    except ValueError:
        pass

    # 5회 입력
    rotations = []
    print(f"\n  {NUM_TRIALS}회 나침반 종료 숫자 입력 (시작은 항상 N=0)\n")

    for i in range(NUM_TRIALS):
        while True:
            try:
                val = input(f"  #{i+1}  종료 나침반 숫자 (0~359) > ").strip()
                end = float(val)
                if 0 <= end <= 359:
                    rot = calc_rotation(end)
                    rotations.append({"trial": i+1, "end": end, "rot": rot})
                    over = rot > TARGET_ANGLE * 1.05
                    low  = rot < TARGET_ANGLE * 0.95
                    tag  = "▲ 고감도" if over else ("▼ 저감도" if low else "✓ 정상")
                    print(f"       → 회전각: {rot:.1f}°  {tag}")
                    break
                else:
                    print("  [!] 0~359 사이 숫자를 입력하세요.")
            except ValueError:
                print("  [!] 숫자를 입력하세요.")

    # 계산
    avg    = sum(r["rot"] for r in rotations) / len(rotations)
    factor = TARGET_ANGLE / max(0.1, avg)
    rec    = round(CURRENT_SENS * factor)
    delta  = rec - CURRENT_SENS

    print(f"\n{DIVIDER}")
    print("  📊  Phase 1 결과")
    print(DIVIDER)
    for r in rotations:
        over = r["rot"] > TARGET_ANGLE * 1.05
        low  = r["rot"] < TARGET_ANGLE * 0.95
        tag  = "고감도" if over else ("저감도" if low else "정상 ")
        print(f"  #{r['trial']}  종료: {r['end']:5.1f}°  →  회전각: {r['rot']:5.1f}°  [{tag}]")
    print(DIVIDER)
    print(f"  평균 실제 회전각  : {avg:.1f}°")
    print(f"  보정 계수         : ×{factor:.3f}")
    print(f"  현재 일반 감도    : {CURRENT_SENS}")
    sign = "+" if delta >= 0 else ""
    print(f"  추천 일반 감도    : {rec}  ({sign}{delta})")
    print(DIVIDER)

    return {
        "trials":    rotations,
        "avg_rot":   avg,
        "factor":    factor,
        "current":   CURRENT_SENS,
        "rec_sens":  rec,
        "delta":     delta,
    }


# ════════════════════════════════════════════════════════════
#  PHASE 2  —  수직 감도 보정 (영상 분석)
# ════════════════════════════════════════════════════════════

def detect_muzzle_flash_frames(video_path, max_shots=15):
    """
    총구 불꽃(Muzzle Flash) 감지로 발사 순간 프레임 인덱스를 추출.
    원리:
      1. 영상을 0.25배속으로 처리 (4프레임마다 1프레임 샘플링)
      2. 화면 중앙 영역의 밝기(평균 휘도)를 프레임별로 계산
      3. 이전 프레임 대비 밝기가 임계값 이상 급등하면 발사로 판정
      4. 같은 발사 이벤트 내 중복 감지 방지(MIN_SHOT_GAP 프레임 이상 간격)
      5. 발사 직후(+1프레임)의 조준점 프레임을 기록
    반환: 발사 순간 프레임 인덱스 목록 (최대 max_shots개)
    """
    SLOWMO_FACTOR   = 4      # 0.25배속 = 4프레임마다 1샘플
    FLASH_THRESHOLD = 18     # 밝기 급등 임계값 (0~255 스케일)
    MIN_SHOT_GAP    = 6      # 같은 발사로 묶는 최소 프레임 간격
    CENTER_RATIO    = 0.25   # 중앙 영역 크기 비율 (화면의 25%)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cx, cy = w // 2, h // 2
    roi_w  = int(w * CENTER_RATIO)
    roi_h  = int(h * CENTER_RATIO)
    x1, y1 = cx - roi_w // 2, cy - roi_h // 2
    x2, y2 = cx + roi_w // 2, cy + roi_h // 2

    brightness_history = []
    for idx in range(0, total, SLOWMO_FACTOR):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        roi  = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness_history.append((idx, float(gray.mean())))
    cap.release()

    if len(brightness_history) < 2:
        return []

    shot_frames  = []
    last_shot_at = -MIN_SHOT_GAP * 10
    for i in range(1, len(brightness_history)):
        idx_cur,  b_cur  = brightness_history[i]
        idx_prev, b_prev = brightness_history[i - 1]
        delta = b_cur - b_prev
        if delta >= FLASH_THRESHOLD:
            if idx_cur - last_shot_at >= MIN_SHOT_GAP * SLOWMO_FACTOR:
                capture_idx = brightness_history[min(i + 1, len(brightness_history)-1)][0]
                shot_frames.append(capture_idx)
                last_shot_at = idx_cur
                if len(shot_frames) >= max_shots:
                    break
    return shot_frames


def extract_frames(video_path, max_shots=15):
    """
    총구 불꽃 감지로 발사 순간 프레임을 추출.
    감지 실패 시 균등 간격 추출로 자동 폴백.
    반환: (frames_b64_list, shot_frame_indices)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] 영상 열기 실패: {video_path}")
        sys.exit(1)

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    new_h    = int(h * FRAME_WIDTH / w)
    cap.release()

    print(f"  해상도: {w}x{h}  FPS: {fps:.1f}  길이: {duration:.1f}초  프레임: {total}")
    print("  총구 불꽃 감지 중 (0.25배속 처리)...")
    shot_indices = detect_muzzle_flash_frames(video_path, max_shots=max_shots)

    if len(shot_indices) >= 3:
        print(f"  발사 감지: {len(shot_indices)}발  프레임: {shot_indices[:5]}{'...' if len(shot_indices)>5 else ''}")
        indices = shot_indices
        detected = True
    else:
        print(f"  [!] 발사 감지 부족({len(shot_indices)}발) → 균등 간격으로 대체")
        n       = min(max_shots, NUM_FRAMES)
        indices = [int(total * i / max(n-1, 1)) for i in range(n)]
        indices = [min(i, total-1) for i in indices]
        detected = False

    cap2 = cv2.VideoCapture(video_path)
    out  = []
    for idx in indices:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap2.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, new_h))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap2.release()

    print(f"  프레임 {len(out)}장 추출 완료 ({'발사 감지' if detected else '균등 간격'})")
    return out, (shot_indices if detected else [])


def analyze_recoil(model, frames_b64, shot_indices=None):
    """
    발사 순간 프레임들을 AI에 전달해 탄착 궤적 분석.
    frames_b64: 발사 감지로 추출된 프레임 base64 목록
    shot_indices: 발사 프레임 번호 목록 (표시용)
    """
    import PIL.Image, io

    images = []
    for b64 in frames_b64:
        img_bytes = base64.b64decode(b64)
        images.append(PIL.Image.open(io.BytesIO(img_bytes)))

    n_shots = len(images)
    shot_info = f"{n_shots}발" if shot_indices else f"프레임 {n_shots}장"

    prompt = f"""이 이미지들은 PUBG 훈련장에서 총구 불꽃 감지로 추출한 실제 발사 순간 프레임 {n_shots}장입니다.
각 이미지는 1발 발사 직후의 화면이며, 순서대로 1발~{n_shots}발입니다.

분석 목표:
마우스를 전혀 움직이지 않은 상태에서 총기 반동으로 인해
각 발사 후 화면 중앙의 조준점(크로스헤어)이 이전 발사 대비 얼마나 이동했는지 추적하세요.

분석 항목:
1. 발사별 조준점 위치 (1발 기준 상대 좌표, px 단위 추정)
   Y양수=위쪽 이동(총구 상탄), Y음수=아래쪽 이동, X양수=우측, X음수=좌측
2. 전체 수직 드리프트 총량(px) 및 방향 ("상단이탈"/"하단이탈"/"안정")
3. 전체 수평 드리프트 총량(px) 및 방향 ("좌이탈"/"우이탈"/"안정")
4. 반동 패턴 유형:
   "vertical_heavy": 수직 지배적, "horizontal_heavy": 수평 지배적,
   "spread": 불규칙 퍼짐, "stable": 안정적
5. 현재 수직 배수 {CURRENT_VMULT} 기준 추천 수직 배수 (0.5~2.0)
6. 분석 신뢰도: "high"/"medium"/"low"

JSON만 응답 (다른 텍스트 없이):
{{
  "shot_trajectory": [
    {{"shot": 1, "x": 0, "y": 0}},
    {{"shot": 2, "x": <number>, "y": <number>}},
    ...
  ],
  "vertical_drift_px": <전체 수직 이동량, 양수>,
  "horizontal_drift_px": <전체 수평 이동량, 양수>,
  "drift_direction_v": <"상단이탈"|"하단이탈"|"안정">,
  "drift_direction_h": <"좌이탈"|"우이탈"|"안정">,
  "recoil_pattern": <"vertical_heavy"|"horizontal_heavy"|"spread"|"stable">,
  "recommended_vertical_mult": <number>,
  "confidence": <"high"|"medium"|"low">,
  "note": "<한국어로 발사 패턴 특징 메모>"
}}"""

    print("  Gemini API — 발사별 반동 궤적 분석 중...")
    resp, model = _generate_with_fallback(model, [prompt] + images)
    raw  = resp.text.replace("```json","").replace("```","").strip()
    try:
        result = json.loads(raw)
        # shot_trajectory → frame_trajectory 로 통일 (하위 호환)
        if "shot_trajectory" in result and "frame_trajectory" not in result:
            result["frame_trajectory"] = [
                {"frame": s["shot"]-1, "x": s["x"], "y": s["y"]}
                for s in result["shot_trajectory"]
            ]
        return result
    except json.JSONDecodeError:
        print(f"[!] 파싱 실패:\n{raw}")
        return None


def generate_report(model, result):
    v_drift = abs(result.get("vertical_drift_px",0))
    h_drift = abs(result.get("horizontal_drift_px",0))
    prompt  = f"""PUBG 반동 분석 결과로 파츠 추천 + 진단 보고서를 JSON으로만 작성:

수직 드리프트: {v_drift}px ({result.get('drift_direction_v','?')})
수평 드리프트: {h_drift}px ({result.get('drift_direction_h','?')})
반동 패턴: {result.get('recoil_pattern','?')}
현재 수직 배수: {CURRENT_VMULT} → 추천: {result.get('recommended_vertical_mult',CURRENT_VMULT)}
신뢰도: {result.get('confidence','?')}
메모: {result.get('note','')}

파츠 DB:
손잡이: 수직손잡이(수직-15%), 경량손잡이(수직-10%/회복+10%), 각도손잡이(수평-20%), 하프그립(수평-10%), 섬지손잡이(퍼짐-20%)
총구: 소음기(수직-5%/수평-5%), 머즐브레이크(수직-10%), 플래시하이더(퍼짐감소)
개머리판: 전술개머리판(회복+25%), 치크패드(퍼짐-20%)

JSON만 응답 (다른 텍스트 없이):
{{
  "diagnosis": "<4~6문장 한국어 종합 진단>",
  "parts": [{{"slot":"<손잡이|총구|개머리판>","name":"<파츠명>","reason":"<한 문장>","priority":<1~3>}}],
  "sensitivity_tip": "<수직 배수 팁 2문장>",
  "training_tip": "<훈련 방법 1문장>"
}}"""

    print("  Gemini API — 파츠 추천 + 보고서 작성 중...")
    resp, model = _generate_with_fallback(model, prompt)
    raw  = resp.text.replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"diagnosis":raw,"parts":[],"sensitivity_tip":"","training_tip":""}


def run_phase2(model):
    global CURRENT_VMULT

    print(f"\n{'━'*60}")
    print("  PHASE 2  —  수직 감도 보정 (영상 분석)")
    print(f"{'━'*60}")
    print("""
  방법:
  ┌─────────────────────────────────────────────────────────┐
  │  훈련장에서 벽을 향해 에임을 고정한 채로               │
  │  마우스를 움직이지 말고 1~2초 이상 연속 발사하세요      │
  └─────────────────────────────────────────────────────────┘""")

    print("\n  사격 영상 경로를 입력하세요.")
    print("  (예: C:\\Users\\me\\Videos\\recoil.mp4)\n")
    video_path = input("  경로 > ").strip().strip('"').strip("'")

    if not os.path.exists(video_path):
        print(f"\n[!] 파일 없음: {video_path}\n")
        return None, None, None

    try:
        v = input(f"\n  현재 수직 감도 배수 (기본 {CURRENT_VMULT}, 엔터 스킵) > ").strip()
        if v: CURRENT_VMULT = float(v)
    except ValueError:
        pass

    print(f"\n{DIVIDER}")
    print("  [1/3] 총구 불꽃 감지 + 프레임 추출 중...")
    print(DIVIDER)
    frames, shot_indices = extract_frames(video_path)

    print(f"\n{DIVIDER}")
    print("  [2/3] 발사별 반동 궤적 분석 중...")
    print(DIVIDER)
    result = analyze_recoil(model, frames, shot_indices)
    if not result:
        print("[!] 분석 실패.")
        return None, None, video_path

    print(f"\n{DIVIDER}")
    print("  [3/3] 파츠 추천 + 보고서 작성 중...")
    print(DIVIDER)
    report = generate_report(model, result)

    # 터미널 출력
    v_drift = result.get("vertical_drift_px", 0)
    h_drift = result.get("horizontal_drift_px", 0)
    vmult   = result.get("recommended_vertical_mult", CURRENT_VMULT)
    delta   = vmult - CURRENT_VMULT
    sign    = "+" if delta >= 0 else ""

    print(f"\n{DIVIDER}")
    print("  📊  Phase 2 결과")
    print(DIVIDER)
    print(f"  수직 드리프트     : {abs(v_drift):.0f} px  ({result.get('drift_direction_v','?')})")
    print(f"  수평 드리프트     : {abs(h_drift):.0f} px  ({result.get('drift_direction_h','?')})")
    print(f"  반동 패턴         : {result.get('recoil_pattern','?')}")
    print(f"  신뢰도            : {result.get('confidence','?').upper()}")
    print(DIVIDER)
    print(f"  현재 수직 배수    : {CURRENT_VMULT}")
    print(f"  추천 수직 배수    : {vmult:.2f}  ({sign}{delta:.2f})")
    print(DIVIDER)
    print("\n  🔧  추천 파츠\n")
    for p in sorted(report.get("parts",[]), key=lambda x: x.get("priority",9)):
        print(f"  [{p.get('priority')}순위] {p.get('slot')} → {p.get('name')}")
        print(f"         {p.get('reason','')}\n")

    return result, report, video_path


# ════════════════════════════════════════════════════════════
#  종합 그래프 (Phase 1 + Phase 2)
# ════════════════════════════════════════════════════════════

def draw_combined_graph(p1, p2_result, p2_report, save_path):
    font_family = _pick_korean_font_family()
    plt.rcParams.update({
        "font.family":      font_family,
        "axes.unicode_minus": False,  # 마이너스(-) 깨짐 방지
        "text.color":       TXT,
        "axes.facecolor":   PANEL,
        "axes.edgecolor":   TXT2,
        "axes.labelcolor":  TXT,
        "xtick.color":      TXT2,
        "ytick.color":      TXT2,
        "figure.facecolor": BG,
        "grid.color":       GRID,
        "grid.linewidth":   0.6,
    })

    has_p1 = p1 is not None
    has_p2 = p2_result is not None

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG)

    rows = 2
    cols = 4 if has_p2 else 2
    gs   = GridSpec(rows, cols, figure=fig,
                    hspace=0.5, wspace=0.38,
                    left=0.05, right=0.97, top=0.88, bottom=0.07)

    # ── 타이틀
    fig.text(0.5, 0.95, "PUBG  SENSITIVITY  ANALYSIS  REPORT",
             ha="center", fontsize=20, fontweight="bold", color=ORA)
    fig.text(0.5, 0.916,
             f"{'Phase 1: 일반 감도 보정' if has_p1 else 'Phase 1: (스킵됨)'}   {'|   Phase 2: 수직 감도 보정 + 파츠 추천' if has_p2 else ''}",
             ha="center", fontsize=10, color=TXT2)

    # ────────── PHASE 1 그래프 ──────────

    # P1-1: 회차별 회전각 막대
    ax_p1_bar = fig.add_subplot(gs[0, 0])
    ax_p1_bar.set_facecolor(PANEL)
    ax_p1_bar.set_title("Phase 1 — 회차별 회전각", color=ORA, fontsize=11, pad=8)

    if not has_p1:
        ax_p1_bar.axis("off")
        ax_p1_bar.text(
            0.5,
            0.5,
            "Phase 1 결과가 없어\n그래프를 건너뛰었습니다.",
            transform=ax_p1_bar.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=TXT2,
        )
    else:
        trials = p1["trials"]
        x      = [r["trial"] for r in trials]
        rots   = [r["rot"] for r in trials]
        colors = []
        for r in rots:
            if r > TARGET_ANGLE * 1.05:   colors.append(RED)
            elif r < TARGET_ANGLE * 0.95: colors.append(BLU)
            else:                          colors.append(GRN)

        bars = ax_p1_bar.bar(x, rots, color=colors, width=0.6, zorder=3)
        ax_p1_bar.axhline(TARGET_ANGLE, color=ORA, linewidth=1.5,
                          linestyle="--", label=f"목표 {TARGET_ANGLE}°", zorder=4)
        ax_p1_bar.axhline(p1["avg_rot"], color=TXT2, linewidth=1,
                          linestyle=":", label=f"평균 {p1['avg_rot']:.1f}°", zorder=4)

        for bar, rot in zip(bars, rots):
            ax_p1_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f"{rot:.1f}°", ha="center", va="bottom", fontsize=8, color=TXT)

        ax_p1_bar.set_xlabel("회차", fontsize=9)
        ax_p1_bar.set_ylabel("회전각 (°)", fontsize=9)
        ax_p1_bar.set_xticks(x)
        ax_p1_bar.legend(fontsize=8, facecolor=PANEL, edgecolor=TXT2, labelcolor=TXT)
        ax_p1_bar.grid(True, axis="y", alpha=0.4, zorder=0)

    # P1-2: 감도 보정 요약 카드
    ax_p1_card = fig.add_subplot(gs[1, 0])
    ax_p1_card.set_facecolor(PANEL)
    ax_p1_card.axis("off")
    ax_p1_card.set_title("Phase 1 — 보정 결과", color=ORA, fontsize=11, pad=8)

    if not has_p1:
        ax_p1_card.text(
            0.5,
            0.5,
            "Phase 1을 실행하지 않아\n요약 카드가 없습니다.",
            transform=ax_p1_card.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=TXT2,
        )
    else:
        delta = p1["delta"]
        sign  = "+" if delta >= 0 else ""
        metrics_p1 = [
            ("평균 회전각",   f"{p1['avg_rot']:.1f}°",       ORA),
            ("보정 계수",     f"×{p1['factor']:.3f}",         BLU),
            ("현재 일반 감도",f"{p1['current']}",              TXT2),
            ("추천 일반 감도",f"{p1['rec_sens']}",             GRN),
            ("변화량",        f"{sign}{delta}",                GRN if delta >= 0 else RED),
        ]
        for i, (label, val, color) in enumerate(metrics_p1):
            y = 0.87 - i * 0.18
            ax_p1_card.text(0.04, y, label, transform=ax_p1_card.transAxes,
                            fontsize=10, color=TXT2, va="center")
            ax_p1_card.text(0.62, y, val, transform=ax_p1_card.transAxes,
                            fontsize=14, fontweight="bold", color=color, va="center")
            # axhline은 transform 인자를 허용하지 않는 버전이 있어 plot으로 대체
            ax_p1_card.plot(
                [0.02, 0.98],
                [y - 0.10, y - 0.10],
                color=GRID,
                linewidth=0.6,
                transform=ax_p1_card.transAxes,
                clip_on=False,
            )

    if not has_p2:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.show()
        print(f"\n  그래프 저장: {save_path}")
        return

    # ────────── PHASE 2 그래프 ──────────
    # shot_trajectory 우선, 없으면 frame_trajectory 사용
    traj = p2_result.get("shot_trajectory", p2_result.get("frame_trajectory", []))
    xs   = [p["x"] for p in traj]
    ys   = [-p["y"] for p in traj]   # Y 반전: 위=화면상 아래
    fi   = list(range(1, len(traj)+1))   # 1발~N발

    # P2-1: 탄착 궤적
    ax_traj = fig.add_subplot(gs[0, 1])
    ax_traj.set_facecolor(PANEL)
    ax_traj.set_title("Phase 2 — 탄착 궤적 (총구 불꽃 감지)", color=ORA, fontsize=11, pad=8)

    if len(xs) >= 2:
        for i in range(len(xs)-1):
            alpha = 0.3 + 0.7 * (i / max(len(xs)-1, 1))
            ax_traj.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                         color=ORA, alpha=alpha, linewidth=1.5)
        sc = ax_traj.scatter(xs, ys, c=fi, cmap="YlOrRd",
                             s=55, zorder=5, edgecolors=TXT2, linewidths=0.4)
        cb = plt.colorbar(sc, ax=ax_traj, pad=0.02)
        cb.set_label("발수", color=TXT2, fontsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT2, fontsize=7)
        # 발수 번호 주석
        for shot_i, (sx, sy) in enumerate(zip(xs, ys)):
            ax_traj.annotate(str(shot_i+1), (sx, sy),
                             textcoords="offset points", xytext=(4, 4),
                             fontsize=6, color=TXT2, zorder=6)

    ax_traj.scatter([0], [0], color=GRN, s=120, zorder=10, marker="*", label="첫 발")
    ax_traj.axhline(0, color=TXT2, linewidth=0.5, linestyle="--", alpha=0.5)
    ax_traj.axvline(0, color=TXT2, linewidth=0.5, linestyle="--", alpha=0.5)
    ax_traj.set_xlabel("수평 이동 px  ← 좌/우 →", fontsize=8)
    ax_traj.set_ylabel("수직 이동 px  ↓상탄/하탄↑", fontsize=8)
    ax_traj.legend(fontsize=8, facecolor=PANEL, edgecolor=TXT2, labelcolor=TXT)
    ax_traj.grid(True, alpha=0.4)

    # P2-2: 수직 드리프트 시계열
    ax_v = fig.add_subplot(gs[0, 2])
    ax_v.set_facecolor(PANEL)
    ax_v.set_title("Phase 2 — 수직 드리프트 (발수별)", color=ORA, fontsize=11, pad=8)
    if ys:
        ax_v.plot(fi, ys, color=RED, linewidth=2, marker="o", markersize=4)
        ax_v.fill_between(fi, ys, 0, where=[y<0 for y in ys],
                          color=RED, alpha=0.15, label="상탄")
        ax_v.fill_between(fi, ys, 0, where=[y>=0 for y in ys],
                          color=BLU, alpha=0.15, label="하탄")
        ax_v.axhline(0, color=TXT2, linewidth=0.8, linestyle="--")
        ax_v.set_xticks(fi)
        ax_v.set_xticklabels([f"{i}발" for i in fi], fontsize=6, rotation=45)
        ax_v.legend(fontsize=8, facecolor=PANEL, edgecolor=TXT2, labelcolor=TXT)
    ax_v.set_xlabel("발수", fontsize=8)
    ax_v.set_ylabel("이동량 (px)", fontsize=8)
    ax_v.grid(True, alpha=0.4)

    # P2-3: 수평 드리프트 시계열
    ax_h = fig.add_subplot(gs[0, 3])
    ax_h.set_facecolor(PANEL)
    ax_h.set_title("Phase 2 — 수평 드리프트 (발수별)", color=ORA, fontsize=11, pad=8)
    if xs:
        ax_h.plot(fi, xs, color=BLU, linewidth=2, marker="o", markersize=4)
        ax_h.fill_between(fi, xs, 0, where=[x>0 for x in xs],
                          color=BLU, alpha=0.2, label="우이탈")
        ax_h.fill_between(fi, xs, 0, where=[x<=0 for x in xs],
                          color=GRN, alpha=0.15, label="좌이탈")
        ax_h.axhline(0, color=TXT2, linewidth=0.8, linestyle="--")
        ax_h.set_xticks(fi)
        ax_h.set_xticklabels([f"{i}발" for i in fi], fontsize=6, rotation=45)
        ax_h.legend(fontsize=8, facecolor=PANEL, edgecolor=TXT2, labelcolor=TXT)
    ax_h.set_xlabel("발수", fontsize=8)
    ax_h.set_ylabel("이동량 (px)", fontsize=8)
    ax_h.grid(True, alpha=0.4)

    # P2-4: 수치 요약
    ax_num = fig.add_subplot(gs[1, 1])
    ax_num.set_facecolor(PANEL)
    ax_num.axis("off")
    ax_num.set_title("Phase 2 — 분석 수치", color=ORA, fontsize=11, pad=8)

    vmult  = p2_result.get("recommended_vertical_mult", CURRENT_VMULT)
    vdelta = vmult - CURRENT_VMULT
    vsign  = "+" if vdelta >= 0 else ""
    metrics_p2 = [
        ("수직 드리프트",  f"{abs(p2_result.get('vertical_drift_px',0)):.0f} px",
         f"({p2_result.get('drift_direction_v','?')})", RED),
        ("수평 드리프트",  f"{abs(p2_result.get('horizontal_drift_px',0)):.0f} px",
         f"({p2_result.get('drift_direction_h','?')})", BLU),
        ("반동 패턴",      p2_result.get("recoil_pattern","?"), "", ORA),
        ("현재 수직 배수", f"{CURRENT_VMULT}", "", TXT2),
        ("추천 수직 배수", f"{vmult:.2f}", f"({vsign}{vdelta:.2f})", GRN),
    ]
    for i, (label, val, sub, color) in enumerate(metrics_p2):
        y = 0.87 - i * 0.18
        ax_num.text(0.04, y, label, transform=ax_num.transAxes,
                    fontsize=10, color=TXT2, va="center")
        ax_num.text(0.58, y, val, transform=ax_num.transAxes,
                    fontsize=13, fontweight="bold", color=color, va="center")
        if sub:
            ax_num.text(0.58, y-0.07, sub, transform=ax_num.transAxes,
                        fontsize=8, color=TXT2, va="center")
        # axhline()은 일부 matplotlib 버전에서 transform 인자를 허용하지 않음
        ax_num.plot([0.02, 0.98], [y - 0.11, y - 0.11],
                    color=GRID, linewidth=0.6, transform=ax_num.transAxes, clip_on=False)

    # P2-5: 파츠 추천
    ax_parts = fig.add_subplot(gs[1, 2])
    ax_parts.set_facecolor(PANEL)
    ax_parts.axis("off")
    ax_parts.set_title("추천 파츠", color=ORA, fontsize=11, pad=8)

    pcolors = {1: ORA, 2: BLU, 3: TXT2}
    sicons  = {"손잡이":"●", "총구":"▲", "개머리판":"◆"}
    for i, part in enumerate(sorted(p2_report.get("parts",[]), key=lambda p:p.get("priority",9))[:4]):
        y      = 0.87 - i * 0.22
        pri    = part.get("priority", 3)
        color  = pcolors.get(pri, TXT2)
        icon   = sicons.get(part.get("slot",""), "•")
        ax_parts.text(0.02, y, f"[P{pri}] {icon} {part.get('name','')}",
                      transform=ax_parts.transAxes,
                      fontsize=9, fontweight="bold", color=color, va="center")
        ax_parts.text(0.02, y-0.09, f"  {part.get('reason','')}",
                      transform=ax_parts.transAxes,
                      fontsize=7.5, color=TXT2, va="center")
        ax_parts.plot([0.01, 0.99], [y - 0.14, y - 0.14],
                      color=GRID, linewidth=0.6, transform=ax_parts.transAxes, clip_on=False)

    # P2-6: AI 진단
    ax_diag = fig.add_subplot(gs[1, 3])
    ax_diag.set_facecolor(PANEL)
    ax_diag.axis("off")
    ax_diag.set_title("AI 종합 진단", color=ORA, fontsize=11, pad=8)

    def wrap(text, w=36):
        words, lines, cur = text.split(), [], ""
        for word in words:
            if len(cur)+len(word)+1 <= w: cur = (cur+" "+word).strip()
            else:
                if cur: lines.append(cur)
                cur = word
        if cur: lines.append(cur)
        return lines

    y_cur = 0.93
    def write(header, body, hc=ORA):
        nonlocal y_cur
        ax_diag.text(0.02, y_cur, header, transform=ax_diag.transAxes,
                     fontsize=8.5, fontweight="bold", color=hc, va="top")
        y_cur -= 0.07
        for line in wrap(body):
            ax_diag.text(0.02, y_cur, line, transform=ax_diag.transAxes,
                         fontsize=7.5, color=TXT, va="top")
            y_cur -= 0.063
        y_cur -= 0.02

    write("◆ 종합 진단",   p2_report.get("diagnosis",""))
    write("◆ 감도 팁",     p2_report.get("sensitivity_tip",""), BLU)
    write("◆ 훈련 방법",   p2_report.get("training_tip",""),    GRN)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"\n  그래프 저장: {save_path}")


# ════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════

def main():
    print(f"[SCRIPT] {os.path.abspath(__file__)}  (ver {SCRIPT_VERSION})")
    print(f"\n{'━'*60}")
    print("  PUBG 종합 감도 분석기  (ALL-IN-ONE)")
    print(f"{'━'*60}")
    print("  Phase 1 : 일반 감도 보정  (나침반 수동 입력)")
    print("  Phase 2 : 수직 감도 보정  (영상 분석 + 파츠 추천)")

    # API 키
    api_key = API_KEY or os.environ.get("GEMINI_API_KEY","")
    if not api_key:
        print("\n[!] API 키 없음.")
        print("    1. https://aistudio.google.com/app/apikey 에서 무료 발급")
        print("    2. 스크립트 상단 API_KEY 에 붙여넣기")
        print("    또는: GEMINI_API_KEY=AIza... python pubg_sens_all.py\n")
        sys.exit(1)

    # Phase 선택
    print("\n  실행할 Phase를 선택하세요:")
    print("  [1] Phase 1만  (일반 감도)")
    print("  [2] Phase 2만  (수직 감도 + 파츠)")
    print("  [3] 둘 다      (추천)")
    choice = input("\n  선택 (1/2/3) > ").strip()

    genai.configure(api_key=api_key)
    model_name = _pick_gemini_model_name()
    print(f"\n  Gemini 모델: {model_name}")
    model = genai.GenerativeModel(model_name)
    p1_result = None
    p2_result = None
    p2_report = None
    video_path = None

    if choice in ("1", "3"):
        p1_result = run_phase1()

    if choice in ("2", "3"):
        p2_result, p2_report, video_path = run_phase2(model)

    # 그래프 저장 경로
    if video_path:
        graph_path = os.path.splitext(video_path)[0] + "_analysis.png"
    else:
        graph_path = os.path.join(os.path.expanduser("~"), "pubg_phase1_analysis.png")

    # 그래프 그리기
    if p1_result or p2_result:
        print(f"\n{DIVIDER}")
        print("  그래프 생성 중...")
        print(DIVIDER)
        draw_combined_graph(p1_result, p2_result, p2_report, graph_path)

    # 텍스트 저장
    save = input("\n  결과를 텍스트 파일로도 저장할까요? (y/n) > ").strip().lower()
    if save == "y":
        txt_path = graph_path.replace("_analysis.png","_result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("PUBG 종합 감도 분석 결과\n" + DIVIDER + "\n\n")
            if p1_result:
                f.write("[ Phase 1 — 일반 감도 보정 ]\n")
                for r in p1_result["trials"]:
                    f.write(f"  #{r['trial']}  종료: {r['end']:.1f}°  회전각: {r['rot']:.1f}°\n")
                sign = "+" if p1_result["delta"] >= 0 else ""
                f.write(f"  평균 회전각: {p1_result['avg_rot']:.1f}°\n")
                f.write(f"  추천 일반 감도: {p1_result['rec_sens']}  ({sign}{p1_result['delta']})\n\n")
            if p2_result and p2_report:
                f.write("[ Phase 2 — 수직 감도 보정 ]\n")
                f.write(f"  수직 드리프트: {abs(p2_result.get('vertical_drift_px',0)):.0f}px ({p2_result.get('drift_direction_v','?')})\n")
                f.write(f"  수평 드리프트: {abs(p2_result.get('horizontal_drift_px',0)):.0f}px ({p2_result.get('drift_direction_h','?')})\n")
                f.write(f"  반동 패턴: {p2_result.get('recoil_pattern','?')}\n")
                f.write(f"  추천 수직 배수: {p2_result.get('recommended_vertical_mult',CURRENT_VMULT):.2f}\n\n")
                f.write("  [ 추천 파츠 ]\n")
                for p in sorted(p2_report.get("parts",[]), key=lambda x:x.get("priority",9)):
                    f.write(f"  [{p.get('priority')}순위] {p.get('slot')} → {p.get('name')}: {p.get('reason')}\n")
                f.write(f"\n  [ AI 진단 ]\n  {p2_report.get('diagnosis','')}\n")
                f.write(f"\n  [ 감도 팁 ]\n  {p2_report.get('sensitivity_tip','')}\n")
                f.write(f"\n  [ 훈련 팁 ]\n  {p2_report.get('training_tip','')}\n")
        print(f"  저장: {txt_path}\n")

if __name__ == "__main__":
    main()
