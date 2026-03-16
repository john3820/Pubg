import os
import tempfile
import streamlit as st
import google.generativeai as genai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 중요: 새 파일 이름으로 임포트
import pubg_logic as logic

# ══════════════════════════════════════════════
#  ★ API 키 ★
# ══════════════════════════════════════════════
FIXED_API_KEY = "AIzaSyDAe7z_0hRu9ASlAfu5GYqvvL7VIcVydHM"

# 로직 연결
extract_frames      = logic.extract_frames
analyze_recoil      = logic.analyze_recoil
generate_report     = logic.generate_report
draw_combined_graph = logic.draw_combined_graph
_pick_model         = logic._pick_gemini_model_name
_calc_rotation      = logic.calc_rotation
CURRENT_VMULT       = logic.CURRENT_VMULT

def run_phase1_calc(current_sens, target_angle, readings):
    rotations = [{"trial": i+1, "end": v, "rot": _calc_rotation(v)} for i, v in enumerate(readings) if v > 0]
    if not rotations: return None
    avg    = sum(r["rot"] for r in rotations) / len(rotations)
    factor = target_angle / max(0.1, avg)
    rec    = round(current_sens * factor)
    return {"trials": rotations, "avg_rot": avg, "factor": factor, "current": current_sens, "rec_sens": rec, "delta": rec - current_sens}

# ══════════════════════════════════════════════
#  Streamlit UI (CSS 및 화면 구성)
# ══════════════════════════════════════════════
st.set_page_config(page_title="SENS.AI — PUBG Analyzer", page_icon="🎯", layout="wide")

# (사용자님의 기존 CSS 스타일 코드 생략 - 그대로 복사해서 넣으시면 됩니다)
st.markdown("<style>...</style>", unsafe_allow_html=True) 

st.markdown("""<div class="hero"><div class="hero-logo">SENS.AI</div><div class="hero-sub">PUBG ANALYSIS SYSTEM</div></div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["PHASE 1 - 일반 감도", "PHASE 2 - 수직/파츠"])

with tab1:
    st.title("일반 감도 보정")
    c1, c2 = st.columns(2)
    with c1: cur_s = st.number_input("현재 감도", 1, 100, 50)
    with c2: tar_a = st.number_input("목표 각도", 45, 180, 90)
    
    cols = st.columns(5)
    reads = [float(cols[i].number_input(f"#{i+1} 종료값", 0, 359, 0)) for i in range(5)]
    
    if any(v > 0 for v in reads):
        res = run_phase1_calc(cur_s, tar_a, reads)
        st.write(f"추천 감도: {res['rec_sens']}")

with tab2:
    st.title("수직 감도 & 파츠 분석")
    v_mult = st.number_input("현재 수직 배수", 0.5, 2.0, 1.0)
    uploaded = st.file_uploader("사격 영상 업로드", type=["mp4", "mov", "avi"])
    
    if st.button("AI 분석 시작") and uploaded:
        genai.configure(api_key=FIXED_API_KEY)
        model = genai.GenerativeModel(_pick_model())
        
        with tempfile.TemporaryDirectory() as tmp:
            vpath = os.path.join(tmp, "video.mp4")
            with open(vpath, "wb") as f: f.write(uploaded.read())
            
            with st.spinner("분석 중..."):
                frames, idxs = extract_frames(vpath)
                result = analyze_recoil(model, frames, idxs)
                report = generate_report(model, result)
                
                st.subheader("AI 분석 결과")
                st.json(result)
                st.subheader("추천 파츠")
                st.write(report.get("parts"))
