import os
import tempfile

import streamlit as st
import google.generativeai as genai

from Pubg_sens_analyzer import (
    extract_frames,
    analyze_recoil,
    generate_report,
    draw_combined_graph,
    CURRENT_VMULT,
    DIVIDER,
    _pick_gemini_model_name,
)


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    model_name = _pick_gemini_model_name()
    model = genai.GenerativeModel(model_name)
    return model_name, model


def main():
    st.set_page_config(page_title="PUBG Sensitivity Analyzer", layout="wide")
    st.title("PUBG Sensitivity Analyzer (Phase 2)")

    st.markdown(
        "영상 업로드 + Gemini **gemini-2.5-flash** 로 반동을 분석해서 "
        "수직 감도 배수와 파츠를 추천합니다."
    )

    with st.sidebar:
        st.header("Gemini 설정")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="https://aistudio.google.com/app/apikey 에서 발급",
        )
        v_mult = st.number_input(
            "현재 수직 감도 배수",
            min_value=0.1,
            max_value=3.0,
            value=float(CURRENT_VMULT),
            step=0.05,
        )

    uploaded = st.file_uploader("훈련장 벽 사격 영상 업로드 (mp4 등)", type=["mp4", "mov", "mkv"])

    if st.button("분석 시작", disabled=uploaded is None):
        if not api_key:
            st.error("먼저 Gemini API Key를 입력하세요.")
            return

        if uploaded is None:
            st.error("먼저 영상을 업로드하세요.")
            return

        with st.spinner("Gemini 설정 중..."):
            try:
                model_name, model = configure_gemini(api_key)
            except Exception as e:
                st.error(f"Gemini 설정 실패: {e}")
                return

        st.write(f"사용 모델: `{model_name}`")

        # 업로드 파일을 임시 파일로 저장
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.read())

            # 환경의 CURRENT_VMULT 를 반영
            os.environ["CURRENT_VMULT_OVERRIDE"] = str(v_mult)

            with st.spinner("프레임 추출 중..."):
                frames = extract_frames(video_path)

            with st.spinner("반동 궤적 분석 중..."):
                result = analyze_recoil(model, frames)
            if not result:
                st.error("분석 실패 (Gemini 응답 파싱 오류). 터미널 로그를 확인하세요.")
                return

            with st.spinner("파츠 추천 + 보고서 작성 중..."):
                report = generate_report(model, result)

            # 그래프 생성 및 표시
            graph_path = os.path.join(tmpdir, "analysis.png")
            draw_combined_graph(None, result, report, graph_path)

            st.subheader("그래프")
            st.image(graph_path, use_column_width=True)

            st.subheader("추천 요약")
            st.json(
                {
                    "vertical_drift_px": result.get("vertical_drift_px"),
                    "horizontal_drift_px": result.get("horizontal_drift_px"),
                    "recoil_pattern": result.get("recoil_pattern"),
                    "recommended_vertical_mult": result.get("recommended_vertical_mult"),
                    "parts": report.get("parts", []),
                    "diagnosis": report.get("diagnosis", ""),
                    "sensitivity_tip": report.get("sensitivity_tip", ""),
                    "training_tip": report.get("training_tip", ""),
                }
            )


if __name__ == "__main__":
    main()

