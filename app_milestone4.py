# ============================================================
# SkillGapAI – Milestone 4
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Optional plotting backends
HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

# Optional PDF backend
HAS_FPDF = False
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# -------------------------------------------
# PAGE CONFIG
# -------------------------------------------
st.set_page_config(
    page_title="SkillGapAI - Milestone 4 Dashboard",
    layout="wide",
    page_icon="📊",
)

# -------------------------------------------
# GLOBAL CSS (Premium UI)
# -------------------------------------------
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
  font-size:24px;
  color:white;
  background: radial-gradient(circle at 20% 0%, #ffffff 0, #7b2ff7 35%, #3b1260 80%);
  box-shadow: 0 10px 28px rgba(0,0,0,0.55);
}
.h-title { font-size:20px; font-weight:800; color:#fff; letter-spacing:0.02em; }
.h-sub { color: rgba(255,255,255,0.90); font-size:12px; margin-top:4px; }

/* Panel */
.panel {
  background: radial-gradient(circle at top left, rgba(123,47,247,0.06) 0, rgba(12,18,36,0.96) 40%, #030712 100%);
  padding:14px;
  border-radius:14px;
  border:1px solid rgba(148,163,184,0.18);
  box-shadow: 0 10px 30px rgba(0,0,0,0.55);
}

/* KPI row + cards */
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

/* Sidebar labels & text */
.sidebar-label {
  font-size:13px;
  font-weight:700;
  color:#e2e8f0;
  margin-bottom:6px;
  display:block;
}
.small-muted { color:#9aa6b2; font-size:12px; }

/* Text & select inputs */
textarea, input, select {
  font-size:13px !important;
}
textarea {
  background: radial-gradient(circle at top left, rgba(15,23,42,0.95), rgba(3,7,18,0.98));
  color: #e6eef8;
  border-radius:10px !important;
  border:1px solid rgba(148,163,184,0.45) !important;
  padding:8px !important;
}
button[kind="secondary"] {
  border-radius:999px !important;
}

/* Tables */
[data-testid="stTable"], [data-testid="stDataFrame"] {
  font-size:13px;
}

/* Tag chips */
.chip-positive {
  display:inline-block;
  padding:4px 8px;
  border-radius:999px;
  background:rgba(34,197,94,0.15);
  color:#bbf7d0;
  font-size:11px;
}
.chip-negative {
  display:inline-block;
  padding:4px 8px;
  border-radius:999px;
  background:rgba(248,113,113,0.15);
  color:#fecaca;
  font-size:11px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------------------------------
# HEADER
# -------------------------------------------
st.markdown(
    f"""
    <div class="header">
      <div class="left">
        <div class="logo">SG</div>
        <div>
          <div class="h-title">SkillGapAI — Milestone 4 Dashboard</div>
          <div class="h-sub">
            Final comparison dashboard • Role-wise skill gap visualization • Exportable reports
          </div>
        </div>
      </div>
      <div style="text-align:right;">
        <div class="small-muted">Module: Dashboard &amp; Report Export</div>
        <div class="small-muted">{datetime.now().strftime('%b %d, %Y')}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------
# DATA SOURCE (Demo / future hook for M3)
# -------------------------------------------

with st.sidebar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Data Source</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Choose data mode",
        ["Demo data ", "Upload CSV (from Milestone 3)"],
        index=0,
    )

    uploaded_file = None
    if mode == "Upload CSV (from Milestone 3)":
        st.markdown('<div class="small-muted">CSV should include columns like: skill, resume_score, job_score</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload skill comparison CSV", type=["csv"])

    st.markdown("</div>", unsafe_allow_html=True)

# --- Prepare data ---

if mode == "Demo data " or uploaded_file is None:
    # Same semantics as your sample, but as a single DataFrame
    resume_skills = {
        "Python": 92,
        "Machine Learning": 88,
        "TensorFlow": 85,
        "SQL": 65,
        "Statistics": 89,
        "Communication": 70,
        "AWS": 30,
        "Project Management": 40,
    }

    job_requirements = {
        "Python": 95,
        "Machine Learning": 90,
        "TensorFlow": 88,
        "SQL": 75,
        "Statistics": 90,
        "Communication": 85,
        "AWS": 80,
        "Project Management": 75,
    }

    skills_df = pd.DataFrame({
        "Skill": list(resume_skills.keys()),
        "Resume Score": list(resume_skills.values()),
        "Job Requirement Score": [job_requirements[k] for k in resume_skills.keys()],
    })
else:
    try:
        df_up = pd.read_csv(uploaded_file)
        # Try to detect columns
        possible_cols = [c.lower() for c in df_up.columns]
        # Expect something like: skill, resume_score, job_score
        skill_col = df_up.columns[possible_cols.index("skill")]
        resume_col = df_up.columns[possible_cols.index("resume_score")]
        job_col = df_up.columns[possible_cols.index("job_score")]
        skills_df = df_up[[skill_col, resume_col, job_col]].rename(
            columns={
                skill_col: "Skill",
                resume_col: "Resume Score",
                job_col: "Job Requirement Score",
            }
        )
    except Exception as e:
        st.sidebar.error(f"CSV format not recognized: {e}")
        st.stop()

# -------------------------------------------
# METRICS & DERIVED COLUMNS
# -------------------------------------------
skills_df["Gap (Resume - Job)"] = skills_df["Resume Score"] - skills_df["Job Requirement Score"]
skills_df["Absolute Gap"] = skills_df["Gap (Resume - Job)"].abs()
skills_df["Match %"] = np.round(
    np.clip(
        (skills_df["Resume Score"] / skills_df["Job Requirement Score"]) * 100, 0, 130
    ),
    1,
)

# Basic summary metrics
total_skills = len(skills_df)
matched_threshold = 0  # treat ">= job - 5" as matched-ish, we'll refine below
gap_threshold = -15

matched_mask = skills_df["Gap (Resume - Job)"] >= -5
gap_mask = skills_df["Gap (Resume - Job)"] <= gap_threshold

matched_skills_count = int(matched_mask.sum())
missing_skills_count = int(gap_mask.sum())

overall_match = float(skills_df["Match %"].mean().round(1))

# Interpret strength/gap lists
strengths = skills_df[skills_df["Gap (Resume - Job)"] >= 5].sort_values("Gap (Resume - Job)", ascending=False)
close_match = skills_df[(skills_df["Gap (Resume - Job)"] > -5) & (skills_df["Gap (Resume - Job)"] < 5)].sort_values("Match %", ascending=False)
gaps = skills_df[skills_df["Gap (Resume - Job)"] <= -5].sort_values("Gap (Resume - Job)")

# -------------------------------------------
# TOP PANEL – SUMMARY
# -------------------------------------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### 📌 Skill Match Overview")

# KPI cards
st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="kpi-card">
      <div class="kpi-label">Overall Match</div>
      <div class="kpi-value">{overall_match:.1f}%</div>
      <div class="kpi-sub">Average coverage across all required skills</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="kpi-card">
      <div class="kpi-label">Matched / Near Match</div>
      <div class="kpi-value">{matched_skills_count}</div>
      <div class="kpi-sub">Skills where resume ≈ job requirement</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="kpi-card">
      <div class="kpi-label">Skills with Gaps</div>
      <div class="kpi-value">{missing_skills_count}</div>
      <div class="kpi-sub">Skills needing focused upskilling</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="kpi-card">
      <div class="kpi-label">Total Skills Evaluated</div>
      <div class="kpi-value">{total_skills}</div>
      <div class="kpi-sub">Based on resume vs job description</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# LAYOUT: LEFT (Charts) / RIGHT (Heatmap + Lists)
# -------------------------------------------
left_col, right_col = st.columns([2.1, 1.4])

# -------------------------------------------
# LEFT: BAR CHARTS & RADAR
# -------------------------------------------
with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📊 Resume vs Job Requirement Comparison")

    if HAS_PLOTLY:
        # Grouped bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=skills_df["Skill"],
                y=skills_df["Resume Score"],
                name="Resume Score",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                x=skills_df["Skill"],
                y=skills_df["Job Requirement Score"],
                name="Job Requirement",
            )
        )
        fig_bar.update_layout(
            barmode="group",
            xaxis_title="Skill",
            yaxis_title="Score (%)",
            height=380,
            margin=dict(l=40, r=10, t=40, b=80),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### 🔻 Gap View (Where are we behind?)")

        gap_df = skills_df.sort_values("Gap (Resume - Job)")
        fig_gap = px.bar(
            gap_df,
            x="Skill",
            y="Gap (Resume - Job)",
            color="Gap (Resume - Job)",
            color_continuous_scale="RdYlGn",
            height=320,
        )
        fig_gap.update_layout(
            xaxis_title="Skill",
            yaxis_title="Gap (Resume - Job)",
            margin=dict(l=40, r=10, t=40, b=80),
        )
        st.plotly_chart(fig_gap, use_container_width=True)
    else:
        # Matplotlib fallback grouped bar
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(skills_df))
        width = 0.35
        ax.bar(x - width / 2, skills_df["Resume Score"], width, label="Resume")
        ax.bar(x + width / 2, skills_df["Job Requirement Score"], width, label="Job")
        ax.set_xticks(x)
        ax.set_xticklabels(skills_df["Skill"], rotation=45, ha="right")
        ax.set_ylabel("Score (%)")
        ax.set_title("Resume vs Job Requirements")
        ax.legend()
        st.pyplot(fig)

        st.markdown("#### 🔻 Gap View (Where are we behind?)")
        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        gap_df = skills_df.sort_values("Gap (Resume - Job)")
        ax2.bar(gap_df["Skill"], gap_df["Gap (Resume - Job)"])
        ax2.set_xticklabels(gap_df["Skill"], rotation=45, ha="right")
        ax2.set_ylabel("Gap (Resume - Job)")
        st.pyplot(fig2)

    # Radar chart for top 5 skills
    st.markdown("---")
    st.markdown("### 🎯 Role View Radar Chart")

    radar_df = skills_df.head(5)  # pick first 5 skills (or you can change to .nlargest)
    labels = radar_df["Skill"].tolist()
    resume_values = radar_df["Resume Score"].tolist()
    job_values = radar_df["Job Requirement Score"].tolist()

    # Close the loop
    labels += labels[:1]
    resume_values += resume_values[:1]
    job_values += job_values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    if not HAS_PLOTLY:
        # matplotlib radar
        fig_r = plt.figure(figsize=(5, 5))
        ax_r = plt.subplot(111, polar=True)
        ax_r.plot(angles, resume_values, linewidth=2, label="Current Profile")
        ax_r.fill(angles, resume_values, alpha=0.25)
        ax_r.plot(angles, job_values, linewidth=2, label="Job Requirement")
        ax_r.fill(angles, job_values, alpha=0.25)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(labels[:-1])
        ax_r.set_title("Role View")
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        st.pyplot(fig_r)
    else:
        # Plotly radar
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=resume_values, theta=labels, fill="toself", name="Current Profile"))
        fig_r.add_trace(go.Scatterpolar(r=job_values, theta=labels, fill="toself", name="Job Requirement"))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=420,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# RIGHT: HEATMAP + TABLES + RECOMMENDATIONS
# -------------------------------------------
with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🔥 Skill Match Heatmap")

    # "Similarity matrix" as 2xN heatmap: Resume vs Job
    if HAS_PLOTLY:
        heat_data = np.vstack([skills_df["Resume Score"].values, skills_df["Job Requirement Score"].values])
        fig_h = go.Figure(
            data=go.Heatmap(
                z=heat_data,
                x=skills_df["Skill"],
                y=["Resume", "Job Requirement"],
                colorscale="Viridis",
                zmin=0,
                zmax=100,
                hovertemplate="Source: %{y}<br>Skill: %{x}<br>Score: %{z:.1f}%<extra></extra>",
            )
        )
        fig_h.update_layout(
            height=260,
            margin=dict(l=60, r=10, t=20, b=60),
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        fig_h, ax_h = plt.subplots(figsize=(5, 3))
        heat_data = np.vstack([skills_df["Resume Score"].values, skills_df["Job Requirement Score"].values])
        cax = ax_h.imshow(heat_data, vmin=0, vmax=100, cmap="viridis")
        ax_h.set_yticks([0, 1])
        ax_h.set_yticklabels(["Resume", "Job Requirement"])
        ax_h.set_xticks(np.arange(len(skills_df["Skill"])))
        ax_h.set_xticklabels(skills_df["Skill"], rotation=45, ha="right")
        fig_h.colorbar(cax, ax=ax_h, fraction=0.046, pad=0.04)
        st.pyplot(fig_h)

    st.markdown("---")
    st.markdown("### 🚀 Strengths & Upskilling Targets")

    tab1, tab2, tab3 = st.tabs(["Top Strengths", "Close Matches", "Gaps / Missing"])

    with tab1:
        if len(strengths) == 0:
            st.info("No strong overshoot skills yet — focus on closing gaps first.")
        else:
            st.write("Skills where your resume score is **significantly higher** than the job requirement.")
            st.dataframe(
                strengths[["Skill", "Resume Score", "Job Requirement Score", "Gap (Resume - Job)", "Match %"]]
                .style.format({"Match %": "{:.1f}%"})
            )

    with tab2:
        if len(close_match) == 0:
            st.info("No very close matches — small tuning may be needed across skills.")
        else:
            st.write("Skills where your profile is **very close** to what the job expects.")
            st.dataframe(
                close_match[["Skill", "Resume Score", "Job Requirement Score", "Gap (Resume - Job)", "Match %"]]
                .style.format({"Match %": "{:.1f}%"})
            )

    with tab3:
        if len(gaps) == 0:
            st.success("No major gaps detected — your resume strongly covers this role.")
        else:
            st.write("Skills with **clear gaps** that should be prioritized for upskilling.")
            st.dataframe(
                gaps[["Skill", "Resume Score", "Job Requirement Score", "Gap (Resume - Job)", "Match %"]]
                .style.format({"Match %": "{:.1f}%"})
            )

            # Simple text recommendations
            st.markdown("#### Suggested Upskilling Focus")
            for _, row in gaps.iterrows():
                skill = row["Skill"]
                diff = row["Gap (Resume - Job)"]
                st.info(f"**{skill}** — increase by ~{abs(int(diff))} points to fully align with the role.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 🔥 **ADDED SECTION — UPSKILLING RECOMMENDATIONS**
# ============================================================

st.markdown("### 🚀 Upskilling Recommendations for Career Growth")

# AI-generated dynamic recommendations using gaps
recommendations = []

for skill in resume_skills:
    res = resume_skills[skill]
    req = job_requirements[skill]

    # If user is weak (less than 60%)
    if res < 60 and req >= 70:
        recommendations.append((
            skill,
            f"Your score in **{skill}** is low. Consider taking a structured certification or guided course."
        ))

    # If moderate (60–75)
    elif 60 <= res < 75:
        recommendations.append((
            skill,
            f"Improve your **{skill}** skills with intermediate-level projects and practice modules."
        ))

    # If near match but needs polishing
    elif 75 <= res < req:
        recommendations.append((
            skill,
            f"Your **{skill}** is good, but a few advanced topics will make it perfect match."
        ))

# Add fallback recommendations if AI generates none
if len(recommendations) == 0:
    recommendations = [
        ("Leadership", "Try improving leadership experience through team projects."),
        ("Cloud Skills", "Learn cloud fundamentals to strengthen technical profile."),
        ("Critical Thinking", "Practice case-study based skill challenges.")
    ]

# Display recommendations beautifully
for skill, advice in recommendations:
    st.info(f"**{skill}** — {advice}")


# -------------------------------------------
# EXPORTS (CSV + PDF)
# -------------------------------------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("### ⤓ Export Reports")

# CSV
csv_data = skills_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Skill Report (CSV)",
    data=csv_data,
    file_name="skillgap_report_m4.csv",
    mime="text/csv",
)

# PDF
def generate_pdf_bytes(df: pd.DataFrame) -> bytes:
    if not HAS_FPDF:
        return b""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "SkillGapAI - Milestone 4 Skill Gap Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Overall Match: {overall_match:.1f}%", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Matched / Near Match Skills: {matched_skills_count}", ln=True)
    pdf.cell(0, 8, f"Skills with Gaps: {missing_skills_count}", ln=True)
    pdf.cell(0, 8, f"Total Skills Evaluated: {total_skills}", ln=True)
    pdf.ln(4)

    # Table header
    pdf.set_font("Arial", "B", 11)
    pdf.cell(55, 8, "Skill", border=1)
    pdf.cell(25, 8, "Resume", border=1, align="C")
    pdf.cell(30, 8, "Job Req.", border=1, align="C")
    pdf.cell(25, 8, "Gap", border=1, align="C")
    pdf.cell(30, 8, "Match %", border=1, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 10)
    for _, row in df.iterrows():
        pdf.cell(55, 7, str(row["Skill"])[:30], border=1)
        pdf.cell(25, 7, f"{row['Resume Score']:.0f}", border=1, align="C")
        pdf.cell(30, 7, f"{row['Job Requirement Score']:.0f}", border=1, align="C")
        pdf.cell(25, 7, f"{row['Gap (Resume - Job)']:.0f}", border=1, align="C")
        pdf.cell(30, 7, f"{row['Match %']:.1f}%", border=1, align="C")
        pdf.ln(7)

    return pdf.output(dest="S").encode("latin1")

if HAS_FPDF:
    pdf_bytes = generate_pdf_bytes(skills_df)
    st.download_button(
        label="📄 Download Full Report (PDF)",
        data=pdf_bytes,
        file_name="skillgap_report_m4.pdf",
        mime="application/pdf",
    )
else:
    st.info("PDF export: `fpdf` library is not installed. Install with `pip install fpdf` to enable PDF reports.")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# FOOTER
# -------------------------------------------
st.markdown(
    "<div style='margin-top:18px; color:#9aa6b2; text-align:center;'>"
    "Milestone 4 • Dashboard & Report Export • SkillGapAI • Developed by Anilkumar"
    "</div>",
    unsafe_allow_html=True,
)
