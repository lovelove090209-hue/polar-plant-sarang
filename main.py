import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Streamlit Test",
    page_icon="✅"
)

st.title("✅ Streamlit 연결 테스트")

st.write("이 화면이 보이면 GitHub와 Streamlit이 정상적으로 연결되었습니다.")

st.divider()

st.write("⏰ 현재 시간:")
st.write(datetime.now())

st.caption("페이지를 새로고침하면 시간이 바뀌면 정상입니다.")

st.success("연결 성공!")

# main.py
import io
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ----------------------------
# Page & Font (Korean safe)
# ----------------------------
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


PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"


# ----------------------------
# Constants (analysis metadata)
# ----------------------------
EC_TARGETS = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(가정/목표)
    "아라고": 4.0,
    "동산고": 8.0,
}

SCHOOL_COLORS = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728",
}


# ----------------------------
# Unicode-safe helpers
# ----------------------------
def norm_variants(s: str) -> set[str]:
    return {unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s)}


def contains_keyword(name: str, keyword: str) -> bool:
    name_vars = norm_variants(name)
    key_vars = norm_variants(keyword)
    return len(name_vars.intersection(key_vars)) > 0


def safe_school_name_from_stem(stem: str) -> str:
    # "송도고_환경데이터" -> "송도고"
    # normalize then split
    stem_nfc = unicodedata.normalize("NFC", stem)
    if "_" in stem_nfc:
        return stem_nfc.split("_", 1)[0].strip()
    return stem_nfc.strip()


def discover_files(data_dir: Path) -> tuple[dict[str, Path], Path | None]:
    """
    ✅ Constraints:
    - pathlib.Path.iterdir() 사용
    - NFC/NFD 양방향 비교
    - f-string 파일명 조합 금지
    - glob 패턴만 사용 금지
    """
    env_files: dict[str, Path] = {}
    growth_xlsx: Path | None = None

    for p in data_dir.iterdir():
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        stem = p.stem

        if suffix == ".csv" and contains_keyword(stem, "환경데이터"):
            school = safe_school_name_from_stem(stem)
            env_files[school] = p

        if suffix in [".xlsx", ".xlsm"] and contains_keyword(stem, "생육결과데이터"):
            growth_xlsx = p

    return env_files, growth_xlsx


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Data loaders (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_environment_data(env_files: dict[str, str]) -> pd.DataFrame:
    frames = []
    for school, path_str in env_files.items():
        path = Path(path_str)
        df = pd.read_csv(path)
        df = df.copy()
        df["school"] = school

        # time parsing
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # numeric coercion
        df = coerce_numeric(df, ["temperature", "humidity", "ph", "ec"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # keep canonical column order if present
    preferred = ["school", "time", "temperature", "humidity", "ph", "ec"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def map_growth_sheets_to_schools(sheets: dict) -> dict[str, pd.DataFrame]:
    """
    ✅ 시트 이름 하드코딩 금지:
    - sheet_name=None로 전부 읽고
    - 시트명/학교명 정규화 포함 비교로 매핑
    """
    mapped: dict[str, pd.DataFrame] = {}

    known_schools = list(EC_TARGETS.keys())
    known_school_norms = {s: norm_variants(s) for s in known_schools}

    for sheet_name, df in sheets.items():
        sheet_norms = norm_variants(str(sheet_name))

        matched_school = None
        for s, s_norms in known_school_norms.items():
            # exact or contains match (via normalized variants)
            if len(sheet_norms.intersection(s_norms)) > 0:
                matched_school = s
                break
            # containment check
            if any(sn in unicodedata.normalize("NFC", str(sheet_name)) for sn in [unicodedata.normalize("NFC", s)]):
                matched_school = s
                break

        school_key = matched_school if matched_school else unicodedata.normalize("NFC", str(sheet_name)).strip()
        df2 = df.copy()
        df2["school"] = school_key
        mapped[school_key] = df2

    return mapped


@st.cache_data(show_spinner=False)
def load_growth_data(xlsx_path_str: str) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path_str)
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    mapped = map_growth_sheets_to_schools(sheets)

    frames = []
    for school, df in mapped.items():
        df = df.copy()

        # Normalize column names lightly (strip)
        df.columns = [unicodedata.normalize("NFC", str(c)).strip() for c in df.columns]

        # Coerce expected numeric columns if present
        df = coerce_numeric(
            df,
            ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"],
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # prefer column order
    preferred = ["school", "개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


# ----------------------------
# UI helpers
# ----------------------------
def filter_by_school(df: pd.DataFrame, school: str) -> pd.DataFrame:
    if df.empty:
        return df
    if school == "전체":
        return df
    if "school" not in df.columns:
        return df
    return df[df["school"] == school].copy()


def mean_safe(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def format_float(x: float | None, ndigits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{ndigits}f}"


def make_kpi_card(label: str, value: str, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def env_school_means(env_df: pd.DataFrame) -> pd.DataFrame:
    if env_df.empty:
        return pd.DataFrame()
    g = env_df.groupby("school", as_index=False).agg(
        avg_temp=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
        avg_ph=("ph", "mean"),
        avg_ec=("ec", "mean"),
        n_rows=("ec", "size"),
    )
    return g


def growth_school_means(growth_df: pd.DataFrame) -> pd.DataFrame:
    if growth_df.empty:
        return pd.DataFrame()
    g = growth_df.groupby("school", as_index=False).agg(
        n=("school", "size"),
        mean_leaf=("잎 수(장)", "mean"),
        mean_shoot=("지상부 길이(mm)", "mean"),
        mean_weight=("생중량(g)", "mean"),
    )
    return g


def pick_optimal_ec_by_weight(growth_means: pd.DataFrame) -> tuple[str | None, float | None]:
    if growth_means.empty or "mean_weight" not in growth_means.columns:
        return None, None
    tmp = growth_means.dropna(subset=["mean_weight"])
    if tmp.empty:
        return None, None
    best_row = tmp.loc[tmp["mean_weight"].idxmax()]
    best_school = str(best_row["school"])
    best_ec = EC_TARGETS.get(best_school)
    return best_school, best_ec


def fig_font(fig):
    fig.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
    return fig


# ----------------------------
# Discover & Load
# ----------------------------
st.title("🌱 극지식물 최적 EC 농도 연구")

data_dir = Path(__file__).resolve().parent / "data"

if not data_dir.exists():
    st.error("data/ 폴더를 찾을 수 없습니다. 저장소에 data 폴더가 있는지 확인해주세요.")
    st.stop()

env_files, growth_xlsx = discover_files(data_dir)

if not env_files:
    st.error("환경 데이터(CSV)를 찾지 못했습니다. data/ 폴더에 '*환경데이터*.csv'가 있는지 확인해주세요.")
    st.stop()

if growth_xlsx is None:
    st.error("생육 결과 데이터(XLSX)를 찾지 못했습니다. data/ 폴더에 '*생육결과데이터*.xlsx'가 있는지 확인해주세요.")
    st.stop()

# Sidebar: School selector
# Dropdown options should be stable ordering by EC target list, but include discovered schools too.
discovered_schools = sorted(set(env_files.keys()) | set(EC_TARGETS.keys()))
ordered = [s for s in EC_TARGETS.keys() if s in discovered_schools] + [s for s in discovered_schools if s not in EC_TARGETS.keys()]
school_options = ["전체"] + ordered

st.sidebar.header("🔎 필터")
selected_school = st.sidebar.selectbox("학교 선택", school_options, index=0)

with st.spinner("데이터를 불러오는 중..."):
    env_df = load_environment_data({k: str(v) for k, v in env_files.items()})
    growth_df = load_growth_data(str(growth_xlsx))

if env_df.empty or growth_df.empty:
    st.error("데이터 로딩에 실패했거나 데이터가 비어 있습니다. 파일 형식/컬럼명을 확인해주세요.")
    st.stop()


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ============================
# Tab 1: Overview
# ============================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
- 극지 식물(나도수영)은 **저온·짧은 생육 기간** 등 극한 환경에 적응한 종으로, 온도/양분(EC) 변화에 민감할 수 있습니다.
- 본 대시보드는 4개교 스마트팜 데이터를 기반으로 **학교별 환경(온도/습도/pH/EC)**와 **생육 결과(생중량/잎수/길이)**를 비교하여
  **최적 EC 농도(특히 생중량 중심)**를 도출하는 것을 목표로 합니다.
        """.strip()
    )

    # School EC condition table
    st.subheader("학교별 EC 조건")
    growth_means_all = growth_school_means(growth_df)
    count_map = {row["school"]: int(row["n"]) for _, row in growth_means_all.iterrows()} if not growth_means_all.empty else {}

    table_rows = []
    for s in ordered:
        table_rows.append(
            {
                "학교명": s,
                "EC 목표": EC_TARGETS.get(s, None),
                "개체수": count_map.get(s, None),
                "색상": SCHOOL_COLORS.get(s, "#888888"),
            }
        )
    ec_table = pd.DataFrame(table_rows)
    st.dataframe(ec_table, use_container_width=True)

    # KPI cards
    env_means_all = env_school_means(env_df)
    env_filtered = filter_by_school(env_df, selected_school)
    growth_filtered = filter_by_school(growth_df, selected_school)

    total_n = int(growth_filtered.shape[0]) if not growth_filtered.empty else 0
    avg_temp = mean_safe(env_filtered["temperature"]) if "temperature" in env_filtered.columns else None
    avg_hum = mean_safe(env_filtered["humidity"]) if "humidity" in env_filtered.columns else None

    best_school, best_ec = pick_optimal_ec_by_weight(growth_means_all)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        make_kpi_card("총 개체수", f"{total_n:,}", "선택된 학교(또는 전체)의 생육 데이터 행 수 기준")
    with c2:
        make_kpi_card("평균 온도(℃)", format_float(avg_temp, 2), "환경 데이터의 temperature 평균")
    with c3:
        make_kpi_card("평균 습도(%)", format_float(avg_hum, 2), "환경 데이터의 humidity 평균")
    with c4:
        # highlight target optimal (Haneul 2.0) + data-driven best
        if best_ec is None:
            make_kpi_card("최적 EC(도출)", "분석 불가", "생중량 평균을 기준으로 최댓값을 갖는 EC")
        else:
            make_kpi_card(
                "최적 EC(도출)",
                f"{best_ec:.1f} dS/m",
                f"생중량 평균이 가장 큰 학교: {best_school}",
            )

    st.info("참고: ‘하늘고 EC 2.0’을 **목표 최적값**으로 표시하되, 그래프/카드는 **데이터 기반(생중량 평균 최댓값)** 도출 결과도 함께 제공합니다.")


# ============================
# Tab 2: Environment
# ============================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    env_means = env_school_means(env_df)
    if env_means.empty:
        st.error("환경 데이터에서 평균을 계산할 수 없습니다. 컬럼명을 확인해주세요(time, temperature, humidity, ph, ec).")
    else:
        # Add target EC column (where available)
        env_means["target_ec"] = env_means["school"].map(EC_TARGETS)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도(℃)", "평균 습도(%)", "평균 pH", "목표 EC vs 실측 EC"),
        )

        # Row 1 Col 1: Avg temp
        fig.add_trace(
            go.Bar(
                x=env_means["school"],
                y=env_means["avg_temp"],
                name="평균 온도",
            ),
            row=1,
            col=1,
        )

        # Row 1 Col 2: Avg humidity
        fig.add_trace(
            go.Bar(
                x=env_means["school"],
                y=env_means["avg_humidity"],
                name="평균 습도",
            ),
            row=1,
            col=2,
        )

        # Row 2 Col 1: Avg pH
        fig.add_trace(
            go.Bar(
                x=env_means["school"],
                y=env_means["avg_ph"],
                name="평균 pH",
            ),
            row=2,
            col=1,
        )

        # Row 2 Col 2: Target vs measured EC (dual bar)
        fig.add_trace(
            go.Bar(
                x=env_means["school"],
                y=env_means["target_ec"],
                name="목표 EC",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=env_means["school"],
                y=env_means["avg_ec"],
                name="실측 EC(평균)",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=720,
            barmode="group",
            title="학교별 환경 평균(온도/습도/pH/EC)",
            font=dict(family=PLOTLY_FONT_FAMILY),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("선택한 학교 시계열")
    if selected_school == "전체":
        st.warning("시계열은 학교별로 측정 주기가 다르므로, ‘전체’에서는 표시하지 않습니다. 왼쪽에서 학교를 선택해주세요.")
    else:
        df_ts = filter_by_school(env_df, selected_school)
        if df_ts.empty:
            st.error("선택한 학교의 환경 데이터가 없습니다.")
        else:
            df_ts = df_ts.dropna(subset=["time"]).sort_values("time")

            target_ec = EC_TARGETS.get(selected_school)

            # Temperature
            fig_t = px.line(df_ts, x="time", y="temperature", title="온도 변화(℃)")
            fig_t = fig_font(fig_t)
            st.plotly_chart(fig_t, use_container_width=True)

            # Humidity
            fig_h = px.line(df_ts, x="time", y="humidity", title="습도 변화(%)")
            fig_h = fig_font(fig_h)
            st.plotly_chart(fig_h, use_container_width=True)

            # EC with target line
            fig_ec = px.line(df_ts, x="time", y="ec", title="EC 변화(dS/m) (목표 EC 기준선 포함)")
            if target_ec is not None:
                fig_ec.add_hline(
                    y=target_ec,
                    line_dash="dash",
                    annotation_text=f"목표 EC {target_ec}",
                    annotation_position="top left",
                )
            fig_ec = fig_font(fig_ec)
            st.plotly_chart(fig_ec, use_container_width=True)

    with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
        env_show = filter_by_school(env_df, selected_school)
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


# ============================
# Tab 3: Growth
# ============================
with tab3:
    st.subheader("🥇 핵심 결과: EC별 평균 생중량")

    g_means = growth_school_means(growth_df)
    if g_means.empty or "mean_weight" not in g_means.columns:
        st.error("생육 결과 평균을 계산할 수 없습니다. XLSX 컬럼명을 확인해주세요.")
    else:
        best_school, best_ec = pick_optimal_ec_by_weight(g_means)
        haneul_weight = None
        if "하늘고" in g_means["school"].values:
            haneul_weight = float(g_means.loc[g_means["school"] == "하늘고", "mean_weight"].iloc[0])

        # Core KPI card
        left, right = st.columns([1, 2])
        with left:
            if best_school is None:
                st.metric("EC별 평균 생중량(최댓값)", "분석 불가")
            else:
                best_weight = float(g_means.loc[g_means["school"] == best_school, "mean_weight"].iloc[0])
                st.metric(
                    "EC별 평균 생중량(최댓값)",
                    f"{best_weight:.3f} g",
                    help=f"최댓값 학교: {best_school} (EC {best_ec} dS/m)" if best_ec is not None else f"최댓값 학교: {best_school}",
                )
                if haneul_weight is not None:
                    st.caption(f"목표 최적(하늘고, EC 2.0) 평균 생중량: **{haneul_weight:.3f} g**")

        with right:
            # Bar: mean weight by school
            g_means2 = g_means.copy()
            g_means2["EC 목표"] = g_means2["school"].map(EC_TARGETS)
            g_means2["label"] = g_means2.apply(
                lambda r: f"{r['school']} (EC {r['EC 목표']})" if pd.notna(r["EC 목표"]) else str(r["school"]),
                axis=1,
            )

            fig_w = px.bar(
                g_means2,
                x="label",
                y="mean_weight",
                title="학교(=EC 조건)별 평균 생중량(g)",
            )
            # highlight best + haneul marker as annotation
            fig_w = fig_font(fig_w)
            if best_school is not None:
                best_label = g_means2.loc[g_means2["school"] == best_school, "label"].iloc[0]
                best_val = float(g_means2.loc[g_means2["school"] == best_school, "mean_weight"].iloc[0])
                fig_w.add_annotation(
                    x=best_label,
                    y=best_val,
                    text="최댓값",
                    showarrow=True,
                    arrowhead=2,
                )
            if "하늘고" in g_means2["school"].values:
                h_label = g_means2.loc[g_means2["school"] == "하늘고", "label"].iloc[0]
                h_val = float(g_means2.loc[g_means2["school"] == "하늘고", "mean_weight"].iloc[0])
                fig_w.add_annotation(
                    x=h_label,
                    y=h_val,
                    text="목표 최적(EC 2.0)",
                    showarrow=True,
                    arrowhead=2,
                    yshift=20,
                )
            st.plotly_chart(fig_w, use_container_width=True)

    st.divider()

    st.subheader("EC별 생육 비교 (2x2)")
    if not g_means.empty:
        g_means_plot = g_means.copy()
        g_means_plot["EC 목표"] = g_means_plot["school"].map(EC_TARGETS)
        g_means_plot["label"] = g_means_plot.apply(
            lambda r: f"{r['school']} (EC {r['EC 목표']})" if pd.notna(r["EC 목표"]) else str(r["school"]),
            axis=1,
        )

        fig2 = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수(장)", "평균 지상부 길이(mm)", "개체수"),
        )

        fig2.add_trace(go.Bar(x=g_means_plot["label"], y=g_means_plot["mean_weight"], name="평균 생중량"), row=1, col=1)
        fig2.add_trace(go.Bar(x=g_means_plot["label"], y=g_means_plot["mean_leaf"], name="평균 잎 수"), row=1, col=2)
        fig2.add_trace(go.Bar(x=g_means_plot["label"], y=g_means_plot["mean_shoot"], name="평균 지상부 길이"), row=2, col=1)
        fig2.add_trace(go.Bar(x=g_means_plot["label"], y=g_means_plot["n"], name="개체수"), row=2, col=2)

        fig2.update_layout(
            height=720,
            barmode="group",
            title="EC(학교)별 생육 지표 비교",
            font=dict(family=PLOTLY_FONT_FAMILY),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("학교별 생중량 분포")
    growth_show = filter_by_school(growth_df, selected_school)
    if growth_show.empty:
        st.error("표시할 생육 데이터가 없습니다.")
    else:
        if "생중량(g)" in growth_show.columns:
            fig_box = px.box(
                growth_show,
                x="school",
                y="생중량(g)",
                title="학교별 생중량 분포(박스플롯)",
                points="all",
            )
            fig_box = fig_font(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.error("생육 데이터에 '생중량(g)' 컬럼이 없습니다.")

    st.divider()

    st.subheader("상관관계 분석(산점도)")
    if not growth_show.empty and "생중량(g)" in growth_show.columns:
        c1, c2 = st.columns(2)

        with c1:
            if "잎 수(장)" in growth_show.columns:
                fig_sc1 = px.scatter(
                    growth_show,
                    x="잎 수(장)",
                    y="생중량(g)",
                    color="school" if selected_school == "전체" else None,
                    title="잎 수 vs 생중량",
                )
                fig_sc1 = fig_font(fig_sc1)
                st.plotly_chart(fig_sc1, use_container_width=True)
            else:
                st.error("'잎 수(장)' 컬럼이 없습니다.")

        with c2:
            if "지상부 길이(mm)" in growth_show.columns:
                fig_sc2 = px.scatter(
                    growth_show,
                    x="지상부 길이(mm)",
                    y="생중량(g)",
                    color="school" if selected_school == "전체" else None,
                    title="지상부 길이 vs 생중량",
                )
                fig_sc2 = fig_font(fig_sc2)
                st.plotly_chart(fig_sc2, use_container_width=True)
            else:
                st.error("'지상부 길이(mm)' 컬럼이 없습니다.")
    else:
        st.error("상관 분석에 필요한 컬럼이 부족합니다. (생중량(g) 필수)")

    with st.expander("📄 학교별 생육 데이터 원본 + XLSX 다운로드"):
        if growth_show.empty:
            st.error("표시할 생육 데이터가 없습니다.")
        else:
            st.dataframe(growth_show, use_container_width=True)

            # XLSX download (BytesIO, no file path)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                if selected_school == "전체":
                    # write each school as a sheet (derived, not hardcoded)
                    for s in sorted(growth_df["school"].dropna().unique().tolist()):
                        df_s = growth_df[growth_df["school"] == s].copy()
                        # Excel sheet name limit 31
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
