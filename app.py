import os
import tempfile
import importlib.util

import streamlit as st
import google.generativeai as genai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
#  ★ API 키 여기에 입력 ★
# ══════════════════════════════════════════════
FIXED_API_KEY = "AIzaSyDAe7z_0hRu9ASlAfu5GYqvvL7VIcVydHM"


# ── 분석 모듈 로드 ─────────────────────────────
def _load_pubg_module():
    path = os.path.join(os.path.dirname(__file__), "Pubg_sens_analyzer.py")
    spec = importlib.util.spec_from_file_location("pubg_orig", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"분석 스크립트를 찾을 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_pubg               = _load_pubg_module()
extract_frames      = _pubg.extract_frames
analyze_recoil      = _pubg.analyze_recoil
generate_report     = _pubg.generate_report
draw_combined_graph = _pubg.draw_combined_graph
_pick_model         = _pubg._pick_gemini_model_name
_calc_rotation      = _pubg.calc_rotation
CURRENT_VMULT       = _pubg.CURRENT_VMULT


def run_phase1_calc(current_sens, target_angle, readings):
    rotations = [
        {"trial": i+1, "end": v, "rot": _calc_rotation(v)}
        for i, v in enumerate(readings) if v > 0
    ]
    if not rotations:
        return None
    avg    = sum(r["rot"] for r in rotations) / len(rotations)
    factor = target_angle / max(0.1, avg)
    rec    = round(current_sens * factor)
    return {"trials": rotations, "avg_rot": avg, "factor": factor,
            "current": current_sens, "rec_sens": rec, "delta": rec - current_sens}


# ══════════════════════════════════════════════
#  페이지 설정
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="SENS.AI — PUBG Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
    background: #070b10 !important;
    color: #c8d8e8 !important;
    font-family: 'Noto Sans KR', 'Rajdhani', sans-serif !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #070b10; }
::-webkit-scrollbar-thumb { background: #f0a500; border-radius: 2px; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1400px !important; }

.hero {
    position: relative;
    background: linear-gradient(135deg, #070b10 0%, #0d1a28 50%, #070b10 100%);
    border: 1px solid rgba(240,165,0,0.2);
    border-radius: 2px; padding: 2.5rem 3rem; margin-bottom: 2rem; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background:
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(240,165,0,0.03) 39px, rgba(240,165,0,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(240,165,0,0.03) 39px, rgba(240,165,0,0.03) 40px);
    pointer-events: none;
}
.hero::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #f0a500, transparent);
}
.hero-logo {
    font-family: 'Rajdhani', sans-serif; font-weight: 700;
    font-size: 3.8rem; letter-spacing: 0.35em; color: #f0a500;
    text-shadow: 0 0 40px rgba(240,165,0,0.35), 0 0 80px rgba(240,165,0,0.1); line-height: 1;
}
.hero-sub { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; letter-spacing: 0.3em; color: #4a6a8a; margin-top: 0.4rem; }
.hero-badge {
    display: inline-block; background: rgba(240,165,0,0.1); border: 1px solid rgba(240,165,0,0.3);
    color: #f0a500; font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; padding: 3px 10px; border-radius: 1px; margin-top: 1rem;
}
.section-hd {
    display: flex; align-items: center; gap: 12px;
    font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 0.85rem;
    letter-spacing: 0.3em; color: #f0a500; text-transform: uppercase; margin: 2rem 0 1.2rem;
}
.section-hd::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(240,165,0,0.4), transparent); }
.section-num { background: #f0a500; color: #070b10; font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 1px; }
.card {
    background: #0c1520; border: 1px solid rgba(240,165,0,0.12); border-radius: 2px;
    padding: 1.4rem 1.6rem; position: relative; transition: border-color 0.2s; height: 100%;
}
.card:hover { border-color: rgba(240,165,0,0.28); }
.card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, #f0a500, transparent); border-radius: 2px 0 0 2px;
}
.card-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; letter-spacing: 0.25em; color: #4a6a8a; text-transform: uppercase; margin-bottom: 0.5rem; }
.card-val { font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 2.6rem; line-height: 1; color: #f0a500; }
.card-val.green { color: #39d87a; }
.card-val.blue  { color: #4a9eff; }
.card-val.white { color: #c8d8e8; }
.card-sub { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #4a6a8a; margin-top: 0.3rem; }
.card-delta { font-size: 0.8rem; font-weight: 600; margin-top: 0.2rem; }
.delta-up   { color: #39d87a; }
.delta-down { color: #ff4d4d; }
.delta-same { color: #4a6a8a; }
.infobox {
    background: rgba(240,165,0,0.04); border: 1px solid rgba(240,165,0,0.15);
    border-left: 3px solid #f0a500; border-radius: 0 2px 2px 0;
    padding: 1rem 1.4rem; font-size: 0.85rem; line-height: 1.9; color: #6a8aaa; margin-bottom: 1.5rem;
}
.infobox b { color: #f0a500; font-weight: 600; }
.infobox .cmd {
    display: block; margin-top: 0.6rem; background: #070b10;
    border: 1px solid rgba(240,165,0,0.2); border-radius: 2px; padding: 8px 14px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.78rem; color: #c8d8e8; letter-spacing: 0.05em;
}
.part-row {
    display: flex; align-items: flex-start; gap: 12px; background: #070b10;
    border: 1px solid rgba(240,165,0,0.1); border-radius: 2px;
    padding: 10px 14px; margin-bottom: 7px; transition: border-color 0.2s;
}
.part-row:hover { border-color: rgba(240,165,0,0.3); }
.part-pri {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; font-weight: 700;
    min-width: 26px; height: 26px; display: flex; align-items: center; justify-content: center; border-radius: 2px;
}
.pri-1 { background: #f0a500; color: #070b10; }
.pri-2 { background: rgba(74,158,255,0.2); color: #4a9eff; border: 1px solid #4a9eff; }
.pri-3 { background: rgba(100,120,140,0.2); color: #6a8aaa; border: 1px solid #4a6a8a; }
.part-body { flex: 1; }
.part-name { font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 0.9rem; color: #c8d8e8; }
.part-slot { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: #4a6a8a; letter-spacing: 0.15em; }
.part-reason { font-size: 0.78rem; color: #6a8aaa; margin-top: 3px; line-height: 1.5; }
.diag-box { background: #0c1520; border: 1px solid rgba(240,165,0,0.12); border-radius: 2px; padding: 1.4rem 1.6rem; }
.diag-section { margin-bottom: 1rem; }
.diag-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; letter-spacing: 0.25em;
    color: #4a6a8a; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 6px;
}
.diag-label::after { content:''; flex:1; height:1px; background:rgba(240,165,0,0.1); }
.diag-text { font-size: 0.85rem; color: #8aaac8; line-height: 1.8; }
.diag-text.tip-s { color: #4a9eff; }
.diag-text.tip-t { color: #39d87a; }
.hline { border: none; border-top: 1px solid rgba(240,165,0,0.1); margin: 2rem 0; }

/* Streamlit 오버라이드 */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #0c1520 !important; border: 1px solid rgba(240,165,0,0.25) !important;
    border-radius: 2px !important; color: #c8d8e8 !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.85rem !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #f0a500 !important; box-shadow: 0 0 0 1px rgba(240,165,0,0.2) !important; outline: none !important;
}
label { color: #6a8aaa !important; font-size: 0.78rem !important; letter-spacing: 0.05em !important; }
div[data-testid="stFileUploader"] {
    background: #0c1520 !important; border: 2px dashed rgba(240,165,0,0.25) !important; border-radius: 2px !important;
}
div[data-testid="stFileUploader"]:hover { border-color: rgba(240,165,0,0.5) !important; }
.stButton > button {
    width: 100% !important; background: #f0a500 !important; color: #070b10 !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; letter-spacing: 0.25em !important; border: none !important;
    border-radius: 2px !important; padding: 0.65rem 1.5rem !important; text-transform: uppercase !important;
}
.stButton > button:hover { background: #ffc130 !important; box-shadow: 0 0 24px rgba(240,165,0,0.35) !important; }
div[data-testid="stTabs"] button {
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.2em !important; font-size: 0.8rem !important;
    color: #4a6a8a !important; border-radius: 0 !important; text-transform: uppercase !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f0a500 !important; border-bottom: 2px solid #f0a500 !important; background: transparent !important;
}
.stSpinner > div > div { border-top-color: #f0a500 !important; }
.stImage img { border: 1px solid rgba(240,165,0,0.15); border-radius: 2px; }
details { background: #0c1520 !important; border: 1px solid rgba(240,165,0,0.12) !important; border-radius: 2px !important; }
summary { color: #6a8aaa !important; font-size: 0.82rem !important; }
div[data-testid="stDownloadButton"] button {
    background: transparent !important; border: 1px solid rgba(240,165,0,0.4) !important;
    color: #f0a500 !important; font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.15em !important; font-size: 0.85rem !important; border-radius: 2px !important;
}
div[data-testid="stDownloadButton"] button:hover { background: rgba(240,165,0,0.08) !important; }
div[data-testid="stRadio"] label { color: #8aaac8 !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] { background: #0a0f17 !important; border-right: 1px solid rgba(240,165,0,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-logo">SENS.AI</div>
  <div class="hero-sub">PUBG BATTLEGROUNDS · SENSITIVITY ANALYSIS SYSTEM · POWERED BY GEMINI</div>
  <div class="hero-badge">◈ MUZZLE FLASH DETECTION · AI TRAJECTORY ANALYSIS · PARTS RECOMMENDATION</div>
</div>
""", unsafe_allow_html=True)

# ── 색상 상수 (그래프용) ──────────────────────
BG="#070b10"; PANEL="#0c1520"; ORA="#f0a500"; GRN="#39d87a"
RED="#ff4d4d"; BLU="#4a9eff"; TXT="#c8d8e8"; TXT2="#4a6a8a"; GRID="#111d2a"

plt.rcParams.update({
    "font.family":"DejaVu Sans", "text.color":TXT, "axes.facecolor":PANEL,
    "axes.edgecolor":TXT2, "axes.labelcolor":TXT, "xtick.color":TXT2,
    "ytick.color":TXT2, "figure.facecolor":BG, "grid.color":GRID,
    "grid.linewidth":0.5, "axes.unicode_minus":False,
})

# ══════════════════════════════════════════════
#  탭
# ══════════════════════════════════════════════
tab1, tab2 = st.tabs(["  ◎  PHASE 1 — 일반 감도 보정  ", "  ◈  PHASE 2 — 수직 감도 + 파츠 추천  "])


# ════════════════════════════════════════════════════════════
#  PHASE 1
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-hd"><span class="section-num">01</span> 일반 감도 보정 — 나침반 수동 입력</div>
    <div class="infobox">
      <b>측정 방법</b><br>
      훈련장 또는 대기실에서 아래 순서를 따르세요.<br>
      <span class="cmd">화면을 N(0°)에 맞추고 → 마우스를 일정한 속도로 좌→우 또는 우→좌로 90° 회전 → 멈춘 직후 나침반 숫자 기록</span>
      위 동작을 <b>5회 반복</b>하고 각 회차의 종료 숫자를 아래에 입력하세요.
    </div>
    """, unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        current_sens = st.number_input("현재 일반 감도", 1, 100, 50, key="p1_sens")
    with cc2:
        target_angle = st.number_input("목표 회전각 (°)", 45, 180, 90, step=45, key="p1_angle")

    st.markdown('<div class="section-hd" style="margin-top:1.5rem"><span class="section-num">↓</span> 5회 나침반 종료값 입력</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    end_vals = [float(col.number_input(f"#{i+1} 종료°", 0, 359, 0, key=f"p1_e{i}")) for i, col in enumerate(cols)]
    filled = [(i+1, v, _calc_rotation(v)) for i, v in enumerate(end_vals) if v > 0]

    if filled:
        st.markdown('<hr class="hline">', unsafe_allow_html=True)
        avg    = sum(r[2] for r in filled) / len(filled)
        factor = target_angle / max(0.1, avg)
        rec    = round(current_sens * factor)
        delta  = rec - current_sens
        sign   = "+" if delta >= 0 else ""
        d_cls  = "delta-up" if delta > 0 else ("delta-down" if delta < 0 else "delta-same")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="card"><div class="card-label">측정 횟수</div><div class="card-val white">{len(filled)}<span style="font-size:1rem;color:#4a6a8a"> / 5</span></div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="card"><div class="card-label">평균 회전각</div><div class="card-val">{avg:.1f}°</div><div class="card-sub">목표 {target_angle}°</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="card"><div class="card-label">보정 계수</div><div class="card-val blue">×{factor:.3f}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="card"><div class="card-label">추천 일반 감도</div><div class="card-val green">{rec}</div><div class="card-delta {d_cls}">{sign}{delta} &nbsp;(현재 {current_sens})</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-hd" style="margin-top:2rem"><span class="section-num">↓</span> 회차별 회전각 분포</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        xs_   = [r[0] for r in filled]
        rots_ = [r[2] for r in filled]
        bar_c = [RED if r > target_angle*1.05 else (BLU if r < target_angle*0.95 else GRN) for r in rots_]
        bars  = ax.bar(xs_, rots_, color=bar_c, width=0.5, zorder=3, edgecolor="none")
        ax.axhline(target_angle, color=ORA, lw=1.5, ls="--", label=f"목표 {target_angle}°", zorder=4)
        ax.axhline(avg, color=TXT2, lw=1, ls=":", label=f"평균 {avg:.1f}°", zorder=4)
        for bar, rot in zip(bars, rots_):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6, f"{rot:.1f}°", ha="center", va="bottom", fontsize=9, color=TXT)
        ax.set_xticks(xs_)
        ax.set_xticklabels([f"#{x}회" for x in xs_])
        ax.set_ylabel("회전각 (°)")
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=TXT2, labelcolor=TXT)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.spines[:].set_color(GRID)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("회차별 상세"):
            for idx, end, rot in filled:
                tag = "🔴 고감도" if rot > target_angle*1.05 else ("🔵 저감도" if rot < target_angle*0.95 else "🟢 정상")
                st.write(f"**#{idx}**  종료: `{end:.0f}°`  →  회전각: `{rot:.1f}°`  {tag}")
    else:
        st.markdown('<div style="text-align:center;padding:3rem;color:#4a6a8a;font-family:\'Share Tech Mono\',monospace;font-size:0.8rem;letter-spacing:0.2em">↑ 나침반 종료 숫자를 입력하면 결과가 표시됩니다</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PHASE 2
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-hd"><span class="section-num">02</span> 수직 감도 보정 — 총구 불꽃 감지 분석</div>
    <div class="infobox">
      <b>영상 촬영 방법</b><br>
      훈련장에서 <b>벽을 향해 에임을 고정</b>한 상태로 마우스를 전혀 움직이지 말고 <b>1~2초 이상 연속 발사</b>하는 영상을 녹화 후 업로드하세요.<br>
      <span class="cmd">총구 불꽃(Muzzle Flash) 자동 감지 → 발사 순간 최대 15프레임 추출 → AI 반동 궤적 분석</span>
    </div>
    """, unsafe_allow_html=True)

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        v_mult = st.number_input("현재 수직 감도 배수", 0.5, 2.0, float(CURRENT_VMULT), step=0.05, key="p2_vmult")
    with cfg2:
        run_p1_also = st.radio("Phase 1 결과도 포함?", ["아니오", "예"], horizontal=True, key="p2_p1")

    p1_result_data = None
    if run_p1_also == "예":
        st.markdown('<div class="section-hd" style="margin-top:1rem"><span class="section-num">+</span> Phase 1 데이터 입력</div>', unsafe_allow_html=True)
        p1c1, p1c2 = st.columns(2)
        with p1c1:
            p1_sens_v = st.number_input("현재 일반 감도", 1, 100, 50, key="p2_sens")
        with p1c2:
            p1_ang_v  = st.number_input("목표 회전각 (°)", 45, 180, 90, step=45, key="p2_angle")
        p1_cols   = st.columns(5)
        p1_readings = [float(col.number_input(f"#{i+1}°", 0, 359, 0, key=f"p2_r{i}")) for i, col in enumerate(p1_cols)]
    else:
        p1_sens_v = 50; p1_ang_v = 90; p1_readings = [0]*5

    st.markdown('<hr class="hline">', unsafe_allow_html=True)
    st.markdown('<div class="section-hd"><span class="section-num">↓</span> 사격 영상 업로드</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("MP4 / MOV / AVI / MKV", type=["mp4","mov","avi","mkv"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("◈  AI 분석 시작", key="analyze")

    if analyze_btn:
        if uploaded is None:
            st.error("영상 파일을 먼저 업로드하세요.")
            st.stop()

        api_key = FIXED_API_KEY or os.environ.get("GEMINI_API_KEY","")
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            st.error("app.py 상단 FIXED_API_KEY 에 Gemini API 키를 입력하세요.")
            st.stop()

        genai.configure(api_key=api_key)
        model_name = _pick_model()
        model = genai.GenerativeModel(model_name)

        if run_p1_also == "예":
            p1_result_data = run_phase1_calc(int(p1_sens_v), float(p1_ang_v), p1_readings)

        with tempfile.TemporaryDirectory() as tmpdir:
            vpath = os.path.join(tmpdir, uploaded.name)
            with open(vpath, "wb") as f:
                f.write(uploaded.read())
            _pubg.CURRENT_VMULT = float(v_mult)

            prog = st.progress(0)
            stat = st.empty()
            def status(msg, pct):
                prog.progress(pct)
                stat.markdown(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;color:#4a9eff;letter-spacing:0.15em;padding:4px 0">▶ {msg}</div>', unsafe_allow_html=True)

            status("총구 불꽃 감지 + 프레임 추출 중...", 15)
            frames, shot_indices = extract_frames(vpath)

            status(f"Gemini AI — 발사별 반동 궤적 분석 중... ({len(frames)}발 감지)", 45)
            result = analyze_recoil(model, frames, shot_indices)
            if not result:
                st.error("분석 실패: Gemini 응답을 파싱할 수 없습니다.")
                st.stop()

            status("파츠 추천 + AI 진단 보고서 작성 중...", 75)
            report = generate_report(model, result)

            status("그래프 생성 중...", 90)
            graph_path = os.path.join(tmpdir, "analysis.png")
            draw_combined_graph(p1_result_data, result, report, graph_path)

            prog.empty(); stat.empty()

            # ── 결과 헤더
            st.markdown("""
            <div style="display:flex;align-items:center;gap:10px;margin:2rem 0 1.5rem">
              <div style="font-family:'Rajdhani',sans-serif;font-weight:700;font-size:1.1rem;letter-spacing:0.3em;color:#f0a500;text-transform:uppercase">Analysis Complete</div>
              <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(240,165,0,0.4),transparent)"></div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#39d87a;letter-spacing:0.2em">◉ ONLINE</div>
            </div>
            """, unsafe_allow_html=True)

            # ── 수치 카드
            v_drift = result.get("vertical_drift_px", 0)
            h_drift = result.get("horizontal_drift_px", 0)
            vmult   = result.get("recommended_vertical_mult", v_mult)
            vdelta  = vmult - v_mult
            vsign   = "+" if vdelta >= 0 else ""
            vc_cls  = "green" if abs(vdelta) < 0.05 else ("blue" if vdelta > 0 else "")
            vd_cls  = "delta-up" if vdelta > 0 else ("delta-down" if vdelta < 0 else "delta-same")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f'<div class="card"><div class="card-label">수직 드리프트</div><div class="card-val">{abs(v_drift):.0f}<span style="font-size:1rem;color:#4a6a8a"> px</span></div><div class="card-sub">{result.get("drift_direction_v","?")}</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="card"><div class="card-label">수평 드리프트</div><div class="card-val blue">{abs(h_drift):.0f}<span style="font-size:1rem;color:#4a6a8a"> px</span></div><div class="card-sub">{result.get("drift_direction_h","?")}</div></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="card"><div class="card-label">반동 패턴</div><div style="font-family:\'Rajdhani\',sans-serif;font-weight:700;font-size:1.1rem;color:#f0a500;margin:0.4rem 0">{result.get("recoil_pattern","?").replace("_"," ").upper()}</div><div class="card-sub">신뢰도 {result.get("confidence","?").upper()}</div></div>', unsafe_allow_html=True)
            with r4:
                st.markdown(f'<div class="card"><div class="card-label">추천 수직 배수</div><div class="card-val {vc_cls}">{vmult:.2f}</div><div class="card-delta {vd_cls}">{vsign}{vdelta:.2f} &nbsp;(현재 {v_mult:.2f})</div></div>', unsafe_allow_html=True)

            # ── 그래프
            st.markdown('<div class="section-hd" style="margin-top:2rem"><span class="section-num">↓</span> 탄착 궤적 + 드리프트 분석</div>', unsafe_allow_html=True)
            with open(graph_path, "rb") as gf:
                img_bytes = gf.read()
            st.image(img_bytes, use_column_width=True)

            st.markdown('<hr class="hline">', unsafe_allow_html=True)

            # ── 파츠 + 진단
            col_parts, col_diag = st.columns([1, 1], gap="large")
            with col_parts:
                st.markdown('<div class="section-hd"><span class="section-num">↓</span> 추천 파츠</div>', unsafe_allow_html=True)
                for part in sorted(report.get("parts",[]), key=lambda p: p.get("priority",9)):
                    pri = part.get("priority",3)
                    pc  = {1:"pri-1",2:"pri-2",3:"pri-3"}.get(pri,"pri-3")
                    st.markdown(f"""
                    <div class="part-row">
                      <div class="part-pri {pc}">P{pri}</div>
                      <div class="part-body">
                        <div class="part-slot">{part.get('slot','').upper()} ·</div>
                        <div class="part-name">{part.get('name','')}</div>
                        <div class="part-reason">{part.get('reason','')}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with col_diag:
                st.markdown('<div class="section-hd"><span class="section-num">↓</span> AI 종합 진단</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="diag-box">
                  <div class="diag-section">
                    <div class="diag-label">◆ 종합 진단</div>
                    <div class="diag-text">{report.get('diagnosis','')}</div>
                  </div>
                  <div class="diag-section">
                    <div class="diag-label">◆ 감도 설정 팁</div>
                    <div class="diag-text tip-s">{report.get('sensitivity_tip','')}</div>
                  </div>
                  <div class="diag-section">
                    <div class="diag-label">◆ 훈련 방법</div>
                    <div class="diag-text tip-t">{report.get('training_tip','')}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # ── 다운로드
            st.markdown("<br>", unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("↓  그래프 PNG 다운로드", img_bytes, "recoil_analysis.png", "image/png")
            with dl2:
                txt = (f"PUBG SENS.AI 분석 결과\n{'='*50}\n"
                       f"수직 드리프트 : {abs(v_drift):.0f}px ({result.get('drift_direction_v','?')})\n"
                       f"수평 드리프트 : {abs(h_drift):.0f}px ({result.get('drift_direction_h','?')})\n"
                       f"반동 패턴     : {result.get('recoil_pattern','?')}\n"
                       f"추천 수직 배수: {vmult:.2f} ({vsign}{vdelta:.2f})\n{'='*50}\n추천 파츠\n")
                for p in sorted(report.get("parts",[]), key=lambda x:x.get("priority",9)):
                    txt += f"  [P{p.get('priority')}] {p.get('slot')} — {p.get('name')}: {p.get('reason')}\n"
                txt += f"\n종합 진단\n{report.get('diagnosis','')}\n\n감도 팁\n{report.get('sensitivity_tip','')}\n\n훈련 팁\n{report.get('training_tip','')}\n"
                st.download_button("↓  텍스트 결과 다운로드", txt.encode("utf-8"), "recoil_result.txt", "text/plain")

# ── 푸터
st.markdown('<div style="text-align:center;padding:3rem 0 1rem;font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;letter-spacing:0.25em;color:#1e2d3d">SENS.AI · PUBG SENSITIVITY ANALYSIS SYSTEM · POWERED BY GOOGLE GEMINI</div>', unsafe_allow_html=True)
