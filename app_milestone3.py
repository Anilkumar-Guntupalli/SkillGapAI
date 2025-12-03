# ============================================================
# SkillGapAI – Milestone 3
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime

# Plotly / matplotlib
HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

# Embedding & vectorization backends
HAS_SENTBERT = False
HAS_SKLEARN = False
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTBERT = True
except Exception:
    HAS_SENTBERT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# PDF generation optional
HAS_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# Page config
st.set_page_config(
    page_title="SkillGapAI — Skill Match ",
    layout="wide",
    page_icon="🧠",
)

# ============================
# GLOBAL CSS (UI UPGRADE)
# ============================
CSS = """
<style>
:root{
  --bg:#040610;
  --panel:#071122;
  --accent1:#5b2c6f;
  --accent2:#7b2ff7;
  --muted:#9aa6b2;
}
html, body {
  background: radial-gradient(circle at top, #1c1440 0, #040610 45%, #02030a 100%);
  color: #e6eef8;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(15,23,42,0.8); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg,#7b2ff7,#00d0ff);
  border-radius: 999px;
}

/* Header */
.header {
  background: linear-gradient(90deg, #5b2c6f, #8a3be6);
  padding:16px;
  border-radius:14px;
  margin-bottom:16px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  box-shadow: 0 14px 40px rgba(0,0,0,0.55);
}
.header .left { display:flex; gap:12px; align-items:center; }
.logo {
  width:58px;
  height:58px;
  border-radius:16px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:900;
  font-size:22px;
  color:white;
  background: radial-gradient(circle at 20% 0%, #ffffff 0, #7b2ff7 35%, #3b1260 80%);
  box-shadow: 0 10px 28px rgba(0,0,0,0.55);
}
.h-title { font-size:18px; font-weight:800; color:#fff; letter-spacing:0.02em; }
.h-sub { color: rgba(255,255,255,0.90); font-size:12px; margin-top:4px; }

.panel {
  background: radial-gradient(circle at top left, rgba(123,47,247,0.06) 0, rgba(12,18,36,0.96) 40%, #030712 100%);
  padding:14px;
  border-radius:14px;
  border:1px solid rgba(148,163,184,0.18);
  box-shadow: 0 10px 30px rgba(0,0,0,0.55);
}

/* KPI row + cards (B-style glassmorphism) */
.kpi-row {
  display:flex;
  gap:14px;
  flex-wrap:wrap;
}
.kpi-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(0,0,0,0.32));
  border: 1px solid rgba(255,255,255,0.10);
  padding: 16px 18px;
  border-radius: 14px;
  min-width: 170px;
  box-shadow: 0 10px 32px rgba(0,0,0,0.6);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  transition: transform 0.18s ease-out, box-shadow 0.18s ease-out, border-color 0.18s;
}
.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 18px 40px rgba(0,0,0,0.85);
  border-color: rgba(180,198,255,0.55);
}
.kpi-label {
  font-size: 13px;
  font-weight: 700;
  color: #d5d9e0;
}
.kpi-value {
  font-size: 32px;
  font-weight: 900;
  margin-top: 8px;
  background: linear-gradient(90deg, #b073ff, #d4b3ff);
  -webkit-background-clip: text;
  color: transparent;
}
.kpi-sub {
  font-size: 12px;
  color: #a8b3c7;
  margin-top: 8px;
}

/* Badges */
.badge-high {
  background: linear-gradient(90deg,#2ecc71,#16a085);
  color:#04210f;
  padding:6px 10px;
  border-radius:8px;
  display:inline-block;
  font-weight:700;
}
.badge-mid {
  background: linear-gradient(90deg,#f1c40f,#e67e22);
  color:#2b1600;
  padding:6px 10px;
  border-radius:8px;
  display:inline-block;
  font-weight:700;
}
.badge-low {
  background: linear-gradient(90deg,#e74c3c,#ff6b6b);
  color:#2b0b07;
  padding:6px 10px;
  border-radius:8px;
  display:inline-block;
  font-weight:700;
}

/* Sidebar labels & inputs */
.sidebar-label {
  font-size:13px;
  font-weight:700;
  color:#e2e8f0;
  margin-bottom:6px;
  display:block;
}
.small-muted { color:#9aa6b2; font-size:12px; }

input[type="number"] {
  background: rgba(15,23,42,0.85);
  color: #e6eef8;
  border: 1px solid rgba(148,163,184,0.40);
  padding:6px;
  border-radius:8px;
  width:100%;
  font-size:13px;
}

/* Textareas */
textarea {
  background: radial-gradient(circle at top left, rgba(15,23,42,0.95), rgba(3,7,18,0.98));
  color: #e6eef8;
  border-radius:10px !important;
  border:1px solid rgba(148,163,184,0.45) !important;
  padding:8px !important;
  font-size:13px !important;
}

/* Download buttons – make them match theme */
button[kind="secondary"] {
  border-radius:999px !important;
}

/* Dataframe font */
[data-testid="stTable"], [data-testid="stDataFrame"] {
  font-size:13px;
}
.download-link { color:#9ad6ff; text-decoration: none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Header
st.markdown(
    f"""
    <div class="header">
      <div class="left">
        <div class="logo">SG</div>
        <div>
          <div class="h-title">Skill Extraction using NLP — SkillGapAI</div>
          <div class="h-sub">Milestone 3 • Skill Gap Analysis &amp; Similarity Matching</div>
        </div>
      </div>
      <div style="text-align:right;">
        <div class="small-muted">Module: spaCy &amp; BERT (hybrid) • {datetime.now().strftime('%b %d, %Y')}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================
# Utilities
# ============================
def normalize_skill_list(text: str):
    if not text:
        return []
    parts = re.split(r'[,\n]+', text)
    out = [p.strip() for p in parts if p.strip()]
    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            uniq.append(s)
            seen.add(k)
    return uniq

# Fallback simple token-overlap similarity if no libraries installed
def simple_overlap_similarity(list_a, list_b):
    def norm_tokens(s):
        return set(re.findall(r"[A-Za-z0-9\+\#]+", s.lower()))
    mat = np.zeros((len(list_a), len(list_b)), dtype=float)
    for i, a in enumerate(list_a):
        ta = norm_tokens(a)
        for j, b in enumerate(list_b):
            tb = norm_tokens(b)
            if not ta or not tb:
                sim = 0.0
            else:
                inter = len(ta & tb)
                uni = len(ta | tb)
                sim = inter / uni if uni > 0 else 0.0
            mat[i, j] = sim
    return mat

# Embedding / similarity wrapper
@st.cache_resource(show_spinner=False)
def load_sentbert_model():
    if not HAS_SENTBERT:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity_matrix(resume_skills, jd_skills):
    if HAS_SENTBERT:
        model = load_sentbert_model()
        r_emb = model.encode(resume_skills, convert_to_tensor=True)
        j_emb = model.encode(jd_skills, convert_to_tensor=True)
        sim = util.cos_sim(r_emb, j_emb).cpu().numpy()
        return sim
    if HAS_SKLEARN:
        vect = TfidfVectorizer().fit(resume_skills + jd_skills)
        r_mat = vect.transform(resume_skills).toarray()
        j_mat = vect.transform(jd_skills).toarray()
        sim = cosine_similarity(r_mat, j_mat)
        return sim
    return simple_overlap_similarity(resume_skills, jd_skills)

def classify_skills(sim_df, threshold_high=0.8, threshold_mid=0.5):
    matched, partial, missing = [], [], []
    for j in sim_df.columns:
        max_sim = sim_df[j].max()
        if max_sim >= threshold_high:
            matched.append(j)
        elif max_sim >= threshold_mid:
            partial.append(j)
        else:
            missing.append(j)
    return matched, partial, missing

def build_comparison_df(sim_df):
    rows = []
    for j in sim_df.columns:
        max_sim = float(sim_df[j].max())
        best_r = sim_df[j].idxmax()
        rows.append(
            {
                "Job Skill": j,
                "Closest Resume Skill": best_r,
                "Similarity (0-1)": round(max_sim, 4),
                "Similarity (%)": round(max_sim * 100, 2),
            }
        )
    return pd.DataFrame(rows)

# ============================
# Sidebar – Inputs & Options
# ============================
with st.sidebar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Inputs & Options</div>', unsafe_allow_html=True)

    upload = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Resume skills</div>', unsafe_allow_html=True)
    resume_text = st.text_area(
        "Resume skills (comma / newline separated)",
        value="Python, SQL, Machine Learning, Tableau",
        height=140,
        key="resume_skills",
    )

    st.markdown('<div class="sidebar-label">Job Description skills</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Job skills (comma / newline separated)",
        value="Python, Data Visualization, Deep Learning, Communication, AWS",
        height=140,
        key="jd_skills",
    )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Analysis thresholds</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        th_high = st.number_input(
            "High match (≥)",
            min_value=0.50,
            max_value=0.99,
            value=0.80,
            step=0.01,
            format="%.2f",
            help="Similarity ≥ this is considered matched",
        )
    with col2:
        th_mid = st.number_input(
            "Partial match (≥)",
            min_value=0.20,
            max_value=0.79,
            value=0.50,
            step=0.01,
            format="%.2f",
            help="Similarity ≥ this and < High is partial",
        )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Report & Export</div>', unsafe_allow_html=True)
    enable_pdf = (
        st.checkbox("Enable PDF report (ReportLab)", value=False)
        if not HAS_REPORTLAB
        else st.checkbox("Enable PDF report (ReportLab available)", value=True)
    )
    st.markdown(
        '<div class="small-muted">Upload CSV may include columns "resume_skills" and "jd_skills" (comma/newline values permitted).</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Parse CSV if uploaded
if upload:
    try:
        df_up = pd.read_csv(upload)
        if "resume_skills" in df_up.columns and "jd_skills" in df_up.columns:
            resume_text = "\n".join(df_up["resume_skills"].astype(str).dropna().tolist())
            jd_text = "\n".join(df_up["jd_skills"].astype(str).dropna().tolist())
        elif "resume" in df_up.columns or "jd" in df_up.columns:
            if "resume" in df_up.columns:
                resume_text = "\n".join(df_up["resume"].astype(str).dropna().tolist())
            if "jd" in df_up.columns:
                jd_text = "\n".join(df_up["jd"].astype(str).dropna().tolist())
    except Exception as e:
        st.sidebar.error("CSV parse failed: " + str(e))

resume_skills = normalize_skill_list(resume_text)
jd_skills = normalize_skill_list(jd_text)

# ============================
# Main – Analysis trigger
# ============================
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("## 🔍 Skill Gap Analysis ")
st.markdown("Provide resume & job skills in the sidebar and press **Analyze**.")
analyze = st.button("Analyze skills", key="analyze_button")
st.markdown("</div>", unsafe_allow_html=True)

if analyze:
    if not resume_skills or not jd_skills:
        st.warning("Please provide both resume skills and job skills (via paste or upload).")
    else:
        with st.spinner("Computing similarity..."):
            sim_mat = compute_similarity_matrix(resume_skills, jd_skills)
            sim_df = pd.DataFrame(sim_mat, index=resume_skills, columns=jd_skills)

        matched, partial, missing = classify_skills(
            sim_df, threshold_high=th_high, threshold_mid=th_mid
        )
        total_jd = max(len(jd_skills), 1)
        overall_match = round(
            ((len(matched) + 0.5 * len(partial)) / total_jd) * 100.0, 2
        )
        comp_df = build_comparison_df(sim_df)

        left, right = st.columns([3, 1.6])

        # LEFT: Visualizations
        with left:
            st.markdown("### 🔬 Similarity Visualizations")

            if HAS_PLOTLY:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=sim_df.values,
                        x=sim_df.columns,
                        y=sim_df.index,
                        colorscale="Viridis",
                        zmin=0,
                        zmax=1,
                        hovertemplate="Resume: %{y}<br>Job: %{x}<br>Sim: %{z:.2f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="Skill Similarity Heatmap",
                    height=420,
                    margin=dict(l=80, r=10, t=40, b=80),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 🔵 Skill Similarity Bubble Chart")
                bubble_rows = []
                for i, r in enumerate(sim_df.index):
                    for j, c in enumerate(sim_df.columns):
                        bubble_rows.append(
                            {"resume": r, "job": c, "sim": float(sim_df.iloc[i, j])}
                        )
                bdf = pd.DataFrame(bubble_rows)
                x_map = {job: idx for idx, job in enumerate(sim_df.columns)}
                y_map = {res: idx for idx, res in enumerate(sim_df.index)}
                bdf["x"] = bdf["job"].map(x_map)
                bdf["y"] = bdf["resume"].map(y_map)
                bdf["size"] = (bdf["sim"] * 50) + 8
                fig2 = px.scatter(
                    bdf,
                    x="x",
                    y="y",
                    size="size",
                    color="sim",
                    hover_name="resume",
                    hover_data=["job", "sim"],
                    color_continuous_scale="Turbo",
                    size_max=60,
                )
                fig2.update_layout(
                    xaxis=dict(
                        tickmode="array",
                        tickvals=list(x_map.values()),
                        ticktext=list(x_map.keys()),
                        title="Job Skills",
                    ),
                    yaxis=dict(
                        tickmode="array",
                        tickvals=list(y_map.values()),
                        ticktext=list(y_map.keys()),
                        title="Resume Skills",
                    ),
                    height=420,
                    margin=dict(l=80, r=10, t=30, b=80),
                    coloraxis_colorbar=dict(title="Similarity"),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                cax = ax.imshow(sim_df.values, vmin=0, vmax=1, cmap="viridis")
                ax.set_xticks(np.arange(len(sim_df.columns)))
                ax.set_yticks(np.arange(len(sim_df.index)))
                ax.set_xticklabels(sim_df.columns, rotation=45, ha="right")
                ax.set_yticklabels(sim_df.index)
                plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title("Skill Similarity Heatmap (fallback)")
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("### 📋 Detailed Comparison")
            st.dataframe(comp_df, height=320)

        # RIGHT: KPI overview & lists
        with right:
            st.markdown("### Skill Match Overview")

            st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Overall Match</div>
                  <div class="kpi-value">{overall_match}%</div>
                  <div class="kpi-sub">({len(matched)} matched • {len(partial)} partial • {len(missing)} missing)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Matched Skills</div>
                  <div class="kpi-value">{len(matched)}</div>
                  <div class="kpi-sub">Skills found exactly</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Partial Matches</div>
                  <div class="kpi-value">{len(partial)}</div>
                  <div class="kpi-sub">Close / related skills</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">Missing Skills</div>
                  <div class="kpi-value">{len(missing)}</div>
                  <div class="kpi-sub">Skills to add</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            labels = ["Matched", "Partial", "Missing"]
            sizes = [len(matched), len(partial), len(missing)]
            colors = ["#2ECC71", "#F1C40F", "#E74C3C"]

            if HAS_PLOTLY:
                figd = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=sizes,
                            hole=0.6,
                            marker=dict(colors=colors),
                            textinfo="none",
                        )
                    ]
                )
                figd.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(figd, use_container_width=True)
                legend_html = "<div style='display:flex; gap:8px; align-items:center; justify-content:center; margin-top:6px;'>"
                legend_html += f"<div style='display:flex; gap:6px; align-items:center;'><div style='width:14px;height:14px;background:{colors[0]};border-radius:3px;'></div> <div style='font-size:13px;color:#cde; margin-right:10px;'>Matched</div></div>"
                legend_html += f"<div style='display:flex; gap:6px; align-items:center;'><div style='width:14px;height:14px;background:{colors[1]};border-radius:3px;'></div> <div style='font-size:13px;color:#cde; margin-right:10px;'>Partial</div></div>"
                legend_html += f"<div style='display:flex; gap:6px; align-items:center;'><div style='width:14px;height:14px;background:{colors[2]};border-radius:3px;'></div> <div style='font-size:13px;color:#cde;'>Missing</div></div>"
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)
            else:
                fig2, ax2 = plt.subplots(figsize=(3.2, 3.2))
                ax2.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.4))
                ax2.axis("equal")
                st.pyplot(fig2)
                st.markdown(
                    "<div style='font-size:13px; margin-top:8px; color:#cde;'>Matched • Partial • Missing</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("#### Missing (job-needed) skills")
            if missing:
                for s in missing:
                    st.markdown(f"- ❌ **{s}**")
            else:
                st.success("No missing skills — resume covers the job requirements well.")

            st.markdown("---")
            st.markdown("#### Matched")
            if matched:
                for s in matched:
                    st.markdown(f"- ✅ **{s}**")
            else:
                st.info("No exact matches found.")

            st.markdown("---")
            st.markdown("#### Partial / Related")
            if partial:
                for s in partial:
                    st.markdown(f"- 🟡 **{s}**")
            else:
                st.info("No partial matches found.")

            st.markdown("---")
            csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download comparison CSV",
                csv_bytes,
                file_name="skill_comparison.csv",
                mime="text/csv",
            )

            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    pd.DataFrame({"resume_skill": resume_skills}).to_excel(
                        writer, sheet_name="resume_skills", index=False
                    )
                    pd.DataFrame({"job_skill": jd_skills}).to_excel(
                        writer, sheet_name="job_skills", index=False
                    )
                    comp_df.to_excel(writer, sheet_name="comparison", index=False)
                st.download_button(
                    "Download Excel (multi-sheet)",
                    excel_buffer.getvalue(),
                    file_name="skillgap_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                try:
                    import zipfile

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, mode="w") as zf:
                        zf.writestr(
                            "resume_skills.csv",
                            pd.DataFrame({"resume_skill": resume_skills}).to_csv(
                                index=False
                            ),
                        )
                        zf.writestr(
                            "job_skills.csv",
                            pd.DataFrame({"job_skill": jd_skills}).to_csv(index=False),
                        )
                        zf.writestr(
                            "comparison.csv", comp_df.to_csv(index=False)
                        )
                    zip_buf.seek(0)
                    st.download_button(
                        "Download ZIP of CSVs",
                        zip_buf.getvalue(),
                        file_name="skillgap_csvs.zip",
                    )
                except Exception as ex:
                    st.error("Export failed: " + str(ex))

            if enable_pdf and HAS_REPORTLAB:
                try:
                    def build_pdf_bytes():
                        buf = io.BytesIO()
                        c = canvas.Canvas(buf, pagesize=A4)
                        width, height = A4
                        left = 20 * mm
                        top = height - 20 * mm
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(left, top, "SkillGapAI — Skill Gap Analysis Report")
                        top -= 10 * mm
                        c.setFont("Helvetica", 9)
                        c.drawString(left, top, f"Generated: {datetime.now().isoformat()}")
                        top -= 8 * mm
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(left, top, f"Overall match: {overall_match}%")
                        top -= 8 * mm

                        c.setFont("Helvetica-Bold", 11)
                        c.drawString(left, top, "Matched Skills:")
                        top -= 6 * mm
                        c.setFont("Helvetica", 10)
                        for s in matched:
                            c.drawString(left + 6 * mm, top, f"- {s}")
                            top -= 5 * mm
                            if top < 30 * mm:
                                c.showPage()
                                top = height - 20 * mm

                        top -= 4 * mm
                        c.setFont("Helvetica-Bold", 11)
                        c.drawString(left, top, "Partial Skills:")
                        top -= 6 * mm
                        c.setFont("Helvetica", 10)
                        for s in partial:
                            c.drawString(left + 6 * mm, top, f"- {s}")
                            top -= 5 * mm
                            if top < 30 * mm:
                                c.showPage()
                                top = height - 20 * mm

                        top -= 4 * mm
                        c.setFont("Helvetica-Bold", 11)
                        c.drawString(left, top, "Missing Skills:")
                        top -= 6 * mm
                        c.setFont("Helvetica", 10)
                        for s in missing:
                            c.drawString(left + 6 * mm, top, f"- {s}")
                            top -= 5 * mm
                            if top < 30 * mm:
                                c.showPage()
                                top = height - 20 * mm

                        c.showPage()
                        c.save()
                        buf.seek(0)
                        return buf.getvalue()

                    pdf_bytes = build_pdf_bytes()
                    st.download_button(
                        "Download PDF report",
                        pdf_bytes,
                        file_name="skillgap_report.pdf",
                        mime="application/pdf",
                    )
                except Exception as ex:
                    st.error("PDF generation failed: " + str(ex))
            else:
                report_txt = io.StringIO()
                report_txt.write("SkillGapAI - Skill Gap Analysis Report\n")
                report_txt.write(f"Generated: {datetime.now().isoformat()}\n\n")
                report_txt.write(f"Overall match: {overall_match}%\n\n")
                report_txt.write("Matched Skills:\n")
                for s in matched:
                    report_txt.write(f"- {s}\n")
                report_txt.write("\nPartial Skills:\n")
                for s in partial:
                    report_txt.write(f"- {s}\n")
                report_txt.write("\nMissing Skills:\n")
                for s in missing:
                    report_txt.write(f"- {s}\n")
                st.download_button(
                    "Download text report",
                    report_txt.getvalue().encode("utf-8"),
                    file_name="skillgap_report.txt",
                    mime="text/plain",
                )

        st.success("Analysis complete — visualizations and exports are available.")
else:
    st.info("Ready. Provide skills in the sidebar and press Analyze.")

# Footer
st.markdown(
    "<div style='margin-top:18px; color:#9aa6b2; text-align:center;'>Milestone 3 • Skill Gap Analysis &amp; Similarity Matching • SkillGapAI • Developed for Anilkumar</div>",
    unsafe_allow_html=True,
)
