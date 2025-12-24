import io
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ============================================================
# 0. Page config + Korean font (Streamlit + Plotly)
# ============================================================
st.set_page_config(
    page_title="🌱 극지식물 최적 EC 농도 연구",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"


# ============================================================
# 1. Metadata (given by task)
# ============================================================
EC_TARGETS = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(목표)
    "아라고": 4.0,
    "동산고": 8.0,
}

SCHOOL_COLORS = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728",
}


# ============================================================
# 2. Unicode-safe utilities (NFC/NFD)
# ============================================================
def norm_set(s: str) -> set[str]:
    return {unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s)}


def match_keyword(text: str, keyword: str) -> bool:
    """
    ✅ '송도고_환경데이터' 안에 '환경데이터'가 들어있는지
    ✅ NFC/NFD 양방향/교차 포함 비교로 안정적으로 탐지
    """
    t_nfc = unicodedata.normalize("NFC", str(text))
    t_nfd = unicodedata.normalize("NFD", str(text))
    k_nfc = unicodedata.normalize("NFC", str(keyword))
    k_nfd = unicodedata.normalize("NFD", str(keyword))

    return (k_nfc in t_nfc) or (k_nfd in t_nfd) or (k_nfc in t_nfd) or (k_nfd in t_nfc)


def parse_school_from_stem(stem: str) -> str:
    # ex) "송도고_환경데이터" -> "송도고"
    s = unicodedata.normalize("NFC", stem).strip()
    if "_" in s:
        return s.split("_", 1)[0].strip()
    return s


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# 3. File discovery (✅ Path.iterdir + ✅ NFC/NFD + no glob-only)
# ============================================================
def discover_data_files(data_dir: Path) -> tuple[dict[str, Path], Path | None]:
    env_csv: dict[str, Path] = {}
    growth_xlsx: Path | None = None

    for p in data_dir.iterdir():  # ✅ required
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        stem = p.stem

        if suffix == ".csv" and match_keyword(stem, "환경데이터"):
            school = parse_school_from_stem(stem)
            env_csv[school] = p

        if suffix in [".xlsx", ".xlsm"] and match_keyword(stem, "생육결과데이터"):
            growth_xlsx = p

    return env_csv, growth_xlsx


# ============================================================
# 4. Data loading (✅ @st.cache_data + spinner outside)
# ============================================================
@st.cache_data(show_spinner=False)
def load_env_data(env_files: dict[str, str]) -> pd.DataFrame:
    frames = []
    for school, path_str in env_files.items():
        path = Path(path_str)
        df = pd.read_csv(path).copy()
        df["school"] = school

        # time parsing
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # numeric
        df = ensure_numeric(df, ["temperature", "humidity", "ph", "ec"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    preferred = ["school", "time", "temperature", "humidity", "ph", "ec"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def map_sheets_to_schoolframes(sheets: dict) -> dict[str, pd.DataFrame]:
    """
    ✅ 시트 이름 하드코딩 금지:
    - sheet_name=None 로 전부 로드
    - 시트명을 NFC/NFD 정규화하여 학교명과 매칭
    """
    known_schools = list(EC_TARGETS.keys())
    known_norms = {s: norm_set(s) for s in known_schools}

    mapped: dict[str, pd.DataFrame] = {}

    for sheet_name, df in sheets.items():
        sn = str(sheet_name)
        sn_norms = norm_set(sn)

        matched = None
        for s, s_norms in known_norms.items():
            if len(sn_norms.intersection(s_norms)) > 0:
                matched = s
                break

        school_key = matched if matched else unicodedata.normalize("NFC", sn).strip()
        df2 = df.copy()
        df2["school"] = school_key
        mapped[school_key] = df2

    return mapped


@st.cache_data(show_spinner=False)
def load_growth_data(xlsx_path_str: str) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path_str)
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")  # ✅ no hardcoded sheets
    mapped = map_sheets_to_schoolframes(sheets)

    frames = []
    for school, df in mapped.items():
        df2 = df.copy()
        df2.columns = [unicodedata.normalize("NFC", str(c)).strip() for c in df2.columns]
        df2 = ensure_numeric(df2, ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"])
        frames.append(df2)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    preferred = ["school", "개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


# ============================================================
# 5. Small helpers (stats / filters)
# ============================================================
def filter_school(df: pd.DataFrame, school: str) -> pd.DataFrame:
    if df.empty or "school" not in df.columns:
        return df
    if school == "전체":
        return df
    return df[df["school"] == school].copy()


def safe_mean(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def fmt(x: float | None, nd: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def env_means(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("school", as_index=False)
        .agg(
            avg_temp=("temperature", "mean"),
            avg_humidity=("humidity", "mean"),
            avg_ph=("ph", "mean"),
            avg_ec=("ec", "mean"),
            n_rows=("ec", "size"),
        )
        .copy()
    )


def growth_means(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("school", as_index=False)
        .agg(
            n=("school", "size"),
            mean_leaf=("잎 수(장)", "mean"),
            mean_shoot=("지상부 길이(mm)", "mean"),
            mean_weight=("생중량(g)", "mean"),
        )
        .copy()
    )


def best_ec_by_weight(gm: pd.DataFrame) -> tuple[str | None, float | None, float | None]:
    if gm.empty or "mean_weight" not in gm.columns:
        return None, None, None
    tmp = gm.dropna(subset=["mean_weight"])
    if tmp.empty:
        return None, None, None
    r = tmp.loc[tmp["mean_weight"].idxmax()]
    school = str(r["school"])
    ec = EC_TARGETS.get(school)
    w = float(r["mean_weight"])
    return school, ec, w


def apply_plotly_font(fig):
    fig.update_layout(font=dict(family=PLOTLY_FONT))
    return fig


# ============================================================
# 6. App start
# ============================================================
st.title("🌱 극지식물 최적 EC 농도 연구")

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

if not data_dir.exists():
    st.error("❌ data/ 폴더를 찾지 못했습니다. 저장소에 data 폴더가 있는지 확인해주세요.")
    st.stop()

env_files, growth_xlsx = discover_data_files(data_dir)

if not env_files:
    st.error("❌ 환경 데이터 CSV를 찾지 못했습니다. data/ 폴더에 '*환경데이터*.csv' 파일이 있는지 확인해주세요.")
    st.stop()

if growth_xlsx is None:
    st.error("❌ 생육 결과 XLSX를 찾지 못했습니다. data/ 폴더에 '*생육결과데이터*.xlsx' 파일이 있는지 확인해주세요.")
    st.stop()

# Sidebar
st.sidebar.header("🔎 필터")
# 옵션은 발견된 학교 + EC_TARGETS 기반 학교를 합쳐서 제공
schools_found = sorted(set(env_files.keys()) | set(EC_TARGETS.keys()))
ordered = [s for s in EC_TARGETS.keys() if s in schools_found] + [s for s in schools_found if s not in EC_TARGETS.keys()]
school_options = ["전체"] + ordered
selected_school = st.sidebar.selectbox("학교 선택", school_options, index=0)

with st.spinner("데이터를 불러오는 중..."):
    env_df = load_env_data({k: str(v) for k, v in env_files.items()})
    growth_df = load_growth_data(str(growth_xlsx))

if env_df.empty:
    st.error("❌ 환경 데이터가 비어 있거나 로딩에 실패했습니다. 컬럼: time, temperature, humidity, ph, ec 를 확인해주세요.")
    st.stop()

if growth_df.empty:
    st.error("❌ 생육 데이터가 비어 있거나 로딩에 실패했습니다. XLSX 시트/컬럼을 확인해주세요.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ============================================================
# Tab 1: Overview
# ============================================================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
- 극지 식물(나도수영)은 저온·짧은 생육 기간 등 극한 환경에 적응한 종입니다.
- 본 대시보드는 4개교 데이터를 기반으로 학교별 환경(온도/습도/pH/EC)과 생육(생중량/잎수/길이)을 비교하여 최적 EC 농도를 도출합니다.
        """.strip()
    )

    st.subheader("학교별 EC 조건 표")
    gm_all = growth_means(growth_df)
    n_map = {str(r["school"]): int(r["n"]) for _, r in gm_all.iterrows()} if not gm_all.empty else {}

    rows = []
    for s in ordered:
        rows.append(
            {
                "학교명": s,
                "EC 목표": EC_TARGETS.get(s),
                "개체수": n_map.get(s),
                "색상": SCHOOL_COLORS.get(s, "#888888"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("주요 지표")
    env_filtered = filter_school(env_df, selected_school)
    growth_filtered = filter_school(growth_df, selected_school)

    total_n = int(growth_filtered.shape[0])
    avg_t = safe_mean(env_filtered["temperature"]) if "temperature" in env_filtered.columns else None
    avg_h = safe_mean(env_filtered["humidity"]) if "humidity" in env_filtered.columns else None

    best_school, best_ec, best_w = best_ec_by_weight(gm_all)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("총 개체수", f"{total_n:,}")
    with c2:
        st.metric("평균 온도(℃)", fmt(avg_t, 2))
    with c3:
        st.metric("평균 습도(%)", fmt(avg_h, 2))
    with c4:
        if best_ec is None or best_school is None:
            st.metric("최적 EC(생중량 기준)", "분석 불가")
        else:
            st.metric("최적 EC(생중량 기준)", f"{best_ec:.1f} dS/m", help=f"최댓값 학교: {best_school}, 평균 생중량: {best_w:.3f} g")

    st.info("‘하늘고(EC 2.0)’는 목표 최적값으로 표시되며, 실제 최적 EC는 **생중량 평균 최댓값** 기준으로 자동 도출됩니다.")


# ============================================================
# Tab 2: Environment
# ============================================================
with tab2:
    st.subheader("학교별 환경 평균 비교 (2x2 서브플롯)")

    em = env_means(env_df)
    if em.empty:
        st.error("환경 평균 계산 실패: 컬럼(time, temperature, humidity, ph, ec)을 확인해주세요.")
    else:
        em["target_ec"] = em["school"].map(EC_TARGETS)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도(℃)", "평균 습도(%)", "평균 pH", "목표 EC vs 실측 EC(평균)"),
        )

        fig.add_trace(go.Bar(x=em["school"], y=em["avg_temp"], name="평균 온도"), row=1, col=1)
        fig.add_trace(go.Bar(x=em["school"], y=em["avg_humidity"], name="평균 습도"), row=1, col=2)
        fig.add_trace(go.Bar(x=em["school"], y=em["avg_ph"], name="평균 pH"), row=2, col=1)

        fig.add_trace(go.Bar(x=em["school"], y=em["target_ec"], name="목표 EC"), row=2, col=2)
        fig.add_trace(go.Bar(x=em["school"], y=em["avg_ec"], name="실측 EC(평균)"), row=2, col=2)

        fig.update_layout(
            height=720,
            barmode="group",
            title="학교별 환경 평균 비교",
            font=dict(family=PLOTLY_FONT),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열 (온도/습도/EC)")

    if selected_school == "전체":
        st.warning("학교별 측정 주기가 다르므로, 시계열은 ‘전체’에서 표시하지 않습니다. 사이드바에서 학교를 선택해주세요.")
    else:
        df_ts = filter_school(env_df, selected_school)
        if df_ts.empty:
            st.error("선택한 학교의 환경 데이터가 없습니다.")
        else:
            df_ts = df_ts.dropna(subset=["time"]).sort_values("time")
            target_ec = EC_TARGETS.get(selected_school)

            fig_t = px.line(df_ts, x="time", y="temperature", title="온도 변화(℃)")
            st.plotly_chart(apply_plotly_font(fig_t), use_container_width=True)

            fig_h = px.line(df_ts, x="time", y="humidity", title="습도 변화(%)")
            st.plotly_chart(apply_plotly_font(fig_h), use_container_width=True)

            fig_ec = px.line(df_ts, x="time", y="ec", title="EC 변화(dS/m) (목표 EC 기준선 포함)")
            if target_ec is not None:
                fig_ec.add_hline(
                    y=target_ec,
                    line_dash="dash",
                    annotation_text=f"목표 EC {target_ec}",
                    annotation_position="top left",
                )
            st.plotly_chart(apply_plotly_font(fig_ec), use_container_width=True)

    with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
        env_show = filter_school(env_df, selected_school)
        if env_show.empty:
            st.error("표시할 환경 데이터가 없습니다.")
        else:
            st.dataframe(env_show, use_container_width=True)
            csv_bytes = env_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ 환경 데이터 CSV 다운로드",
                data=csv_bytes,
                file_name="환경데이터_필터결과.csv",
                mime="text/csv",
            )


# ============================================================
# Tab 3: Growth
# ============================================================
with tab3:
    st.subheader("🥇 핵심 결과 카드: EC별 평균 생중량 (최댓값 강조)")

    gm = growth_means(growth_df)
    if gm.empty:
        st.error("생육 평균 계산 실패: XLSX 컬럼을 확인해주세요.")
    else:
        best_school, best_ec, best_w = best_ec_by_weight(gm)

        left, right = st.columns([1, 2])
        with left:
            if best_school is None:
                st.metric("EC별 평균 생중량(최댓값)", "분석 불가")
            else:
                st.metric(
                    "EC별 평균 생중량(최댓값)",
                    f"{best_w:.3f} g",
                    help=f"최댓값 학교: {best_school} (EC {best_ec} dS/m)" if best_ec is not None else f"최댓값 학교: {best_school}",
                )

        with right:
            gm2 = gm.copy()
            gm2["EC 목표"] = gm2["school"].map(EC_TARGETS)
            gm2["label"] = gm2.apply(
                lambda r: f"{r['school']} (EC {r['EC 목표']})" if pd.notna(r["EC 목표"]) else str(r["school"]),
                axis=1,
            )

            fig_w = px.bar(gm2, x="label", y="mean_weight", title="학교(=EC 조건)별 평균 생중량(g)")
            fig_w = apply_plotly_font(fig_w)

            if best_school is not None:
                best_label = gm2.loc[gm2["school"] == best_school, "label"].iloc[0]
                fig_w.add_annotation(x=best_label, y=best_w, text="최댓값", showarrow=True, arrowhead=2)

            if "하늘고" in gm2["school"].values:
                h_label = gm2.loc[gm2["school"] == "하늘고", "label"].iloc[0]
                h_w = float(gm2.loc[gm2["school"] == "하늘고", "mean_weight"].iloc[0])
                fig_w.add_annotation(
                    x=h_label,
                    y=h_w,
                    text="목표 최적(EC 2.0)",
                    showarrow=True,
                    arrowhead=2,
                    yshift=20,
                )

            st.plotly_chart(fig_w, use_container_width=True)

    st.divider()
    st.subheader("EC별 생육 결과 비교 (2x2 막대 그래프)")

    if not gm.empty:
        gm2 = gm.copy()
        gm2["EC 목표"] = gm2["school"].map(EC_TARGETS)
        gm2["label"] = gm2.apply(
            lambda r: f"{r['school']} (EC {r['EC 목표']})" if pd.notna(r["EC 목표"]) else str(r["school"]),
            axis=1,
        )

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수(장)", "평균 지상부 길이(mm)", "개체수"),
        )

        fig.add_trace(go.Bar(x=gm2["label"], y=gm2["mean_weight"], name="평균 생중량"), row=1, col=1)
        fig.add_trace(go.Bar(x=gm2["label"], y=gm2["mean_leaf"], name="평균 잎 수"), row=1, col=2)
        fig.add_trace(go.Bar(x=gm2["label"], y=gm2["mean_shoot"], name="평균 지상부 길이"), row=2, col=1)
        fig.add_trace(go.Bar(x=gm2["label"], y=gm2["n"], name="개체수"), row=2, col=2)

        fig.update_layout(
            height=720,
            barmode="group",
            title="EC(학교)별 생육 지표 비교",
            font=dict(family=PLOTLY_FONT),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포 (박스플롯)")

    growth_show = filter_school(growth_df, selected_school)
    if growth_show.empty:
        st.error("표시할 생육 데이터가 없습니다.")
    else:
        if "생중량(g)" in growth_show.columns:
            fig_box = px.box(
                growth_show,
                x="school",
                y="생중량(g)",
                title="학교별 생중량 분포",
                points="all",
            )
            st.plotly_chart(apply_plotly_font(fig_box), use_container_width=True)
        else:
            st.error("생육 데이터에 '생중량(g)' 컬럼이 없습니다.")

    st.divider()
    st.subheader("상관관계 분석 (산점도 2개)")

    if not growth_show.empty and "생중량(g)" in growth_show.columns:
        c1, c2 = st.columns(2)

        with c1:
            if "잎 수(장)" in growth_show.columns:
                fig1 = px.scatter(
                    growth_show,
                    x="잎 수(장)",
                    y="생중량(g)",
                    color="school" if selected_school == "전체" else None,
                    title="잎 수 vs 생중량",
                )
                st.plotly_chart(apply_plotly_font(fig1), use_container_width=True)
            else:
                st.error("'잎 수(장)' 컬럼이 없습니다.")

        with c2:
            if "지상부 길이(mm)" in growth_show.columns:
                fig2 = px.scatter(
                    growth_show,
                    x="지상부 길이(mm)",
                    y="생중량(g)",
                    color="school" if selected_school == "전체" else None,
                    title="지상부 길이 vs 생중량",
                )
                st.plotly_chart(apply_plotly_font(fig2), use_container_width=True)
            else:
                st.error("'지상부 길이(mm)' 컬럼이 없습니다.")
    else:
        st.error("상관 분석에 필요한 컬럼이 부족합니다. (생중량(g) 필수)")

    with st.expander("📄 학교별 생육 데이터 원본 + XLSX 다운로드"):
        if growth_show.empty:
            st.error("표시할 생육 데이터가 없습니다.")
        else:
            st.dataframe(growth_show, use_container_width=True)

            # ✅ BytesIO Excel download (no path)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                if selected_school == "전체":
                    for s in sorted(growth_df["school"].dropna().unique().tolist()):
                        df_s = growth_df[growth_df["school"] == s].copy()
                        sheet = str(s)[:31]
                        df_s.to_excel(writer, index=False, sheet_name=sheet)
                else:
                    sheet = str(selected_school)[:31]
                    growth_show.to_excel(writer, index=False, sheet_name=sheet)

            buffer.seek(0)
            st.download_button(
                label="⬇️ 생육 데이터 XLSX 다운로드",
                data=buffer,
                file_name="생육데이터_필터결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
