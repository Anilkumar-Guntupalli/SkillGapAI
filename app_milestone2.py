# ============================================================
# SkillGapAI – Milestone 2
# ============================================================
import streamlit as st
import re
import os
import io
import json
import tempfile
from datetime import datetime

# plotting libraries
HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt  # fallback

# pdf/docx reading
import PyPDF2
import docx2txt

# CSV / data
import pandas as pd

# optional spaCy (auto-use if installed; silent)
HAS_SPACY = False
try:
    import spacy
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False

# PDF generation (ReportLab) - optional
HAS_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# --------------------------
# Developer sample path kept only for metadata
# NOTE: local path included per developer instruction (will be converted to URL in your tooling).
# Replace/transform as needed when calling external services.
# --------------------------
SAMPLE_PDF_URL = "/mnt/data/SkillGapAI Analyzing Resume & Job Post for Skill Gap.pdf"
# Example uploaded file path detected in history (developer instruction): included as reference
UPLOADED_FILE_URL = "/mnt/data/a6175bde-76d8-4eb7-9da8-16c9052ff6ce.png"

# --------------------------
# Page config & header (PROJECT TITLE ADDED HERE)
# --------------------------
st.set_page_config(page_title="Skill Extraction using NLP – Technical & Soft Skill Analyzer", layout="wide", page_icon="🧠")

# --------------------------
# Fixed theme (Dark) CSS
# --------------------------
DARK_CSS = """
<style>
:root{ --bg:#05060b; --muted:#9aa6b2; --accent1:#7b2ff7; --accent2:#00d0ff; }
html, body { background: linear-gradient(180deg,#04040a,#071022); color: #e6eef8; }
.header { display:flex; justify-content:space-between; align-items:center; padding:20px; border-radius:12px;
          background: linear-gradient(90deg, rgba(123,47,247,0.20), rgba(0,208,255,0.10));
          border:1px solid rgba(255,255,255,0.08); margin-bottom:14px;}
.logo { width:72px;height:72px;border-radius:18px;display:flex;align-items:center;justify-content:center;
        font-weight:900;font-size:26px;color:white;background: linear-gradient(135deg,var(--accent1),var(--accent2));
        box-shadow: 0 20px 60px rgba(11,12,20,0.7); }
.title { font-size:22px; font-weight:900; letter-spacing:0.02em; }
.subtitle { color:#a9bac6; font-size:13px; margin-top:6px; }
.glass { background: radial-gradient(circle at top left, rgba(123,47,247,0.12), rgba(0,0,0,0.72));
         padding:16px; border-radius:16px; border:1px solid rgba(255,255,255,0.06); }
.skill-badge { display:inline-block; padding:6px 10px; margin:6px 6px 6px 0; border-radius:999px;
               background: linear-gradient(90deg, rgba(123,47,247,0.16), rgba(0,208,255,0.08));
               color:#e6eef8; border:1px solid rgba(255,255,255,0.10); font-weight:600; font-size:12px; }
.highlight { background: rgba(123,47,247,0.35); padding:2px 5px; border-radius:6px; }
.small-muted { color:#94a3b8; font-size:12px; }
.chip-title { display:inline-block; padding:4px 10px; border-radius:999px;
              background: linear-gradient(90deg, rgba(123,47,247,0.30), rgba(0,208,255,0.20));
              font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#e5edff; }
.metric-card { background: radial-gradient(circle at top left, rgba(123,47,247,0.25), rgba(6,10,20,0.95));
               border-radius:16px; padding:10px 14px; border:1px solid rgba(255,255,255,0.12); }
.metric-label { font-size:11px; color:#cbd5f5; text-transform:uppercase; letter-spacing:0.08em; }
.metric-value { font-size:20px; font-weight:800; }
.metric-sub { font-size:11px; color:#9ca3af; }
.header-right { text-align:right; color:#9aa6b2; font-size:12px; }
</style>
"""
# apply dark css
st.markdown(DARK_CSS, unsafe_allow_html=True)
theme_choice = "Dark"

# Top header with the chosen project title (big, bold, centered-left)
st.markdown(
    f"""
    <div class="header">
      <div style="display:flex; gap:14px; align-items:center;">
        <div class="logo">SG</div>
        <div>
          <div class="title">Skill Extraction using NLP – Technical & Soft Skill Analyzer</div>
          <div class="subtitle">Module: Skill Extraction • spaCy-ready • BERT pipeline (future)</div>
        </div>
      </div>
      <div class="header-right">
        <div>Milestone 2 • {datetime.now().strftime('%b %d, %Y')}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Skill lists (defaults — unchangeable now)
# --------------------------
DEFAULT_TECHNICAL_SKILLS = [
    "python", "java", "c++", "c#", "sql", "html", "css", "javascript", "react",
    "node", "node.js", "tensorflow", "pytorch", "machine learning", "data analysis",
    "data visualization", "aws", "azure", "gcp", "power bi", "tableau", "django",
    "flask", "scikit-learn", "nlp", "pandas", "numpy", "matplotlib", "keras", "docker", "kubernetes"
]

DEFAULT_SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "problem solving", "time management",
    "adaptability", "critical thinking", "creativity", "collaboration", "decision making",
    "attention to detail", "organization", "empathy"
]

TECHNICAL_SKILLS = list(DEFAULT_TECHNICAL_SKILLS)
SOFT_SKILLS = list(DEFAULT_SOFT_SKILLS)

# --------------------------
# spaCy helper (auto-load, silent)
# --------------------------
def _load_spacy_model_silent():
    if not HAS_SPACY:
        return None
    if st.session_state.get("_spacy_model_cached", None) is not None:
        return st.session_state["_spacy_model_cached"]
    try:
        model = spacy.load("en_core_web_sm")
    except Exception:
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            model = spacy.load("en_core_web_sm")
        except Exception:
            model = None
    st.session_state["_spacy_model_cached"] = model
    return model

_spacy_model = _load_spacy_model_silent()

# --------------------------
# Text normalization & tokenization
# --------------------------
def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r'\r\n', '\n', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip().lower()

def tokens_from_text(text: str, use_spacy_auto=True):
    text = normalize_text(text)
    if use_spacy_auto and _spacy_model:
        doc = _spacy_model(text)
        tokens = set([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop])
        tokens |= set([ent.text.lower() for ent in doc.ents])
        return tokens
    return set(re.findall(r'[a-zA-Z\+\#\.]+', text.lower()))

def extract_skills_from_text(text: str, technical_list, soft_list, use_spacy_auto=True):
    # accepts lists (we pass the defaults)
    text_norm = normalize_text(text)
    tokens = tokens_from_text(text_norm, use_spacy_auto=use_spacy_auto)
    found_tech = []
    found_soft = []
    for skill in sorted(technical_list, key=lambda s: -len(s)):
        s_norm = skill.lower()
        if (s_norm in text_norm) or (s_norm in tokens):
            found_tech.append(skill.title())
    for skill in sorted(soft_list, key=lambda s: -len(s)):
        s_norm = skill.lower()
        if (s_norm in text_norm) or (s_norm in tokens):
            found_soft.append(skill.title())
    return sorted(list(set(found_tech))), sorted(list(set(found_soft)))

# --------------------------
# File reading helpers
# --------------------------
def read_pdf(file_obj_or_path):
    text = ""
    try:
        if isinstance(file_obj_or_path, str):
            f = open(file_obj_or_path, "rb")
            reader = PyPDF2.PdfReader(f)
        else:
            reader = PyPDF2.PdfReader(file_obj_or_path)
        for p in reader.pages:
            try:
                txt = p.extract_text()
            except Exception:
                txt = None
            if txt:
                text += txt + "\n"
        if isinstance(file_obj_or_path, str):
            f.close()
    except Exception as e:
        st.warning("PDF read error: " + str(e))
    return text

def read_docx(file_obj_or_path):
    try:
        if isinstance(file_obj_or_path, str):
            return docx2txt.process(file_obj_or_path)
        else:
            b = file_obj_or_path.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(b)
                tmp.flush()
                path = tmp.name
            text = docx2txt.process(path)
            try:
                os.remove(path)
            except Exception:
                pass
            return text
    except Exception as e:
        st.warning("DOCX read error: " + str(e))
        return ""

def read_txt(file_obj_or_path):
    try:
        if isinstance(file_obj_or_path, str):
            with open(file_obj_or_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        else:
            raw = file_obj_or_path.read()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
    except Exception as e:
        st.warning("TXT read error: " + str(e))
        return ""

def extract_uploaded_text(uploaded):
    if not uploaded:
        return ""
    name = getattr(uploaded, "name", str(uploaded))
    name = name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded)
    elif name.endswith(".docx"):
        return read_docx(uploaded)
    elif name.endswith(".txt"):
        return read_txt(uploaded)
    else:
        return ""

# --------------------------
# CSV parsing helper
# --------------------------
def extract_text_from_csv(uploaded_csv):
    try:
        if isinstance(uploaded_csv, str):
            df = pd.read_csv(uploaded_csv)
        else:
            df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.warning("CSV read error: " + str(e))
        return "", ""
    resume_text = ""
    jd_text = ""
    cols = {c.lower(): c for c in df.columns}
    if 'text' in cols:
        vals = df[cols['text']].dropna().astype(str).tolist()
        resume_text = "\n\n".join(vals)
    else:
        if 'resume' in cols:
            resume_text = "\n\n".join(df[cols['resume']].dropna().astype(str).tolist())
        if 'jd' in cols:
            jd_text = "\n\n".join(df[cols['jd']].dropna().astype(str).tolist())
    return resume_text, jd_text

# --------------------------
# Highlighting helper
# --------------------------
def highlight_text_html(text: str, skills: list):
    if not text:
        return ""
    text_esc = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    def repl(match):
        return f"<span class='highlight'>{match.group(0)}</span>"
    for s in sorted(skills, key=lambda x: -len(x)):
        try:
            pattern = re.compile(r'(' + re.escape(s) + r')', flags=re.IGNORECASE)
            text_esc = pattern.sub(repl, text_esc)
        except Exception:
            continue
    html = "<div style='line-height:1.5;'>" + text_esc.replace("\n", "<br/>") + "</div>"
    return html

# --------------------------
# Utility: extract sentence context for a skill
# --------------------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')
def find_skill_context_sentences(text: str, skills: list, max_sentences=1):
    contexts = {}
    if not text:
        return contexts
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    low_sentences = [s.lower() for s in sentences]
    for skill in skills:
        s_norm = skill.lower()
        found = []
        for i, ls in enumerate(low_sentences):
            if re.search(r'\b' + re.escape(s_norm) + r'\b', ls):
                found.append(sentences[i])
                if len(found) >= max_sentences:
                    break
        contexts[skill] = found or []
    return contexts

# --------------------------
# PDF report builder (ReportLab) - fallback to text file
# --------------------------
def generate_pdf_report_bytes(context):
    """
    context: dict containing metrics and lists
    returns bytes of PDF (or raises if cannot)
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    left_margin = 20 * mm
    top = height - 20 * mm
    line_h = 8 * mm

    def write_line(text, x=left_margin, y=None, size=10, bold=False):
        nonlocal top
        if y is None:
            top -= line_h
            y = top
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, text)

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, top, "SkillGapAI - Skill Extraction Report")
    top -= 12 * mm
    c.setFont("Helvetica", 9)
    c.drawString(left_margin, top, f"Generated: {datetime.now().isoformat()}")
    top -= 8 * mm

    # Snapshot metrics
    write_line("Snapshot:", bold=True)
    write_line(f"Candidate skills (total): {context.get('total_candidate',0)}")
    write_line(f"JD skills (total): {context.get('total_required',0)}")
    write_line(f"Missing skills (total): {context.get('missing_total',0)}")
    write_line(f"Match rate: {context.get('coverage_pct',0.0):.1f}%")
    top -= 4 * mm

    # Technical and Soft percentages
    write_line("Percentages:", bold=True)
    for k,v in context.get('percentages', {}).items():
        write_line(f"{k}: {v:.1f}%")
    top -= 4 * mm

    # Lists sections (truncate if too long)
    def write_list(title, items):
        nonlocal top
        write_line(title, bold=True)
        for it in items:
            write_line(f"- {it}", size=9)
            if top < 30*mm:
                c.showPage()
                top = height - 20*mm
        top -= 2*mm

    write_list("Candidate — Technical", context.get('tech_r', []))
    write_list("Candidate — Soft", context.get('soft_r', []))
    write_list("Job — Technical (required)", context.get('tech_j', []))
    write_list("Job — Soft (required)", context.get('soft_j', []))

    # Missing / Extra
    write_list("Missing Technical", context.get('missing_tech', []))
    write_list("Missing Soft", context.get('missing_soft', []))
    write_list("Extra Technical", context.get('extra_tech', []))
    write_list("Extra Soft", context.get('extra_soft', []))

    # Per-skill contexts (sample)
    write_line("Per-skill contexts (resume / job):", bold=True)
    for skill, rows in context.get('skill_context_rows', [])[:40]:  # limit
        write_line(f"{skill}:", bold=False)
        write_line(f"  Resume: {rows.get('candidate_sentence','')[:200]}", size=9)
        write_line(f"  JD: {rows.get('job_sentence','')[:200]}", size=9)
        if top < 30*mm:
            c.showPage()
            top = height - 20*mm

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# --------------------------
# Sidebar Controls (minimal; theme fixed & no custom lists)
# --------------------------
with st.sidebar:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.header("Controls")
    # kept concise: no theme toggle, no custom lists
    st.caption("Quick toggles")
    show_missing = st.checkbox("Show missing skills", value=True)
    highlight_toggle = st.checkbox("Highlight detected skills in preview", value=True)
    st.markdown("---")
    st.caption("Developer")
    st.markdown("SkillGapAI • Developed for Anilkumar")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Main UI: two columns
# --------------------------
left, right = st.columns([1, 2])

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<span class='chip-title'>Inputs</span>", unsafe_allow_html=True)
    st.subheader("Upload")
    uploaded_resume = st.file_uploader(
        "Upload Resume (PDF / DOCX / TXT / CSV)",
        type=["pdf", "docx", "txt", "csv"],
        key="resume_uploader",
    )
    uploaded_jd = st.file_uploader(
        "Upload Job Description (PDF / DOCX / TXT / CSV)",
        type=["pdf", "docx", "txt", "csv"],
        key="jd_uploader",
    )
    st.markdown("Or paste / edit text on the right preview panels.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<span class='chip-title'>Analysis & Output</span>", unsafe_allow_html=True)
    st.subheader("Preview & Results")

    # Pre-fill empty preview strings
    resume_text = ""
    jd_text = ""

    # Resume source: uploaded file or CSV contents
    if uploaded_resume:
        if str(uploaded_resume.name).lower().endswith(".csv"):
            r_text, j_text = extract_text_from_csv(uploaded_resume)
            resume_text = r_text or ""
            if j_text:
                jd_text = j_text
        else:
            resume_text = extract_uploaded_text(uploaded_resume)

    # JD source: uploaded file or embedded CSV
    if uploaded_jd:
        if str(uploaded_jd.name).lower().endswith(".csv"):
            r_text2, j_text2 = extract_text_from_csv(uploaded_jd)
            jd_text = j_text2 or jd_text
            if not resume_text and r_text2:
                resume_text = r_text2
        else:
            jd_text = extract_uploaded_text(uploaded_jd)

    # Editable preview areas
    with st.expander("Resume preview (editable)", expanded=True):
        resume_text = st.text_area(
            "Resume (editable)", resume_text, height=220, key="resume_preview"
        )

    with st.expander("Job Description preview (editable)", expanded=True):
        jd_text = st.text_area(
            "Job Description (editable)", jd_text, height=220, key="jd_preview"
        )

    # Analyze button
    if st.button("Analyze skills"):
        resume_text = st.session_state.get("resume_preview", "") or ""
        jd_text = st.session_state.get("jd_preview", "") or ""

        if not resume_text and not jd_text:
            st.warning("Please provide resume and/or job description text (upload, CSV, or paste).")
        else:
            # auto spaCy usage if model loaded
            use_spacy_auto = bool(_spacy_model)

            # Use default (fixed) skill lists
            tech_r, soft_r = (
                extract_skills_from_text(resume_text, TECHNICAL_SKILLS, SOFT_SKILLS, use_spacy_auto=use_spacy_auto)
                if resume_text
                else ([], [])
            )
            tech_j, soft_j = (
                extract_skills_from_text(jd_text, TECHNICAL_SKILLS, SOFT_SKILLS, use_spacy_auto=use_spacy_auto)
                if jd_text
                else ([], [])
            )

            total_candidate = len(tech_r) + len(soft_r)
            total_required = len(tech_j) + len(soft_j)

            missing_tech = [s for s in tech_j if s not in tech_r]
            missing_soft = [s for s in soft_j if s not in soft_r]
            extra_tech = [s for s in tech_r if s not in tech_j]
            extra_soft = [s for s in soft_r if s not in soft_j]

            overlap_tech = len([s for s in tech_j if s in tech_r])
            overlap_soft = len([s for s in soft_j if s in soft_r])
            total_overlap = overlap_tech + overlap_soft
            coverage_pct = (total_overlap / total_required * 100.0) if total_required > 0 else 0.0

            # Percentages breakdowns
            def pct(n, d):
                return (n / d * 100.0) if d > 0 else 0.0

            candidate_tech_pct = pct(len(tech_r), total_candidate)
            candidate_soft_pct = pct(len(soft_r), total_candidate)
            jd_tech_pct = pct(len(tech_j), total_required)
            jd_soft_pct = pct(len(soft_j), total_required)
            missing_tech_pct = pct(len(missing_tech), len(tech_j))
            missing_soft_pct = pct(len(missing_soft), len(soft_j))
            extra_tech_pct = pct(len(extra_tech), len(tech_r))
            extra_soft_pct = pct(len(extra_soft), len(soft_r))

            # Snapshot (colorful custom cards)
            st.markdown("### 🔎 Snapshot")
            mc1, mc2, mc3, mc4 = st.columns(4)

            with mc1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">Candidate skills</div>
                      <div class="metric-value">{total_candidate}</div>
                      <div class="metric-sub">Technical: {candidate_tech_pct:.1f}% • Soft: {candidate_soft_pct:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">JD skills</div>
                      <div class="metric-value">{total_required}</div>
                      <div class="metric-sub">Technical: {jd_tech_pct:.1f}% • Soft: {jd_soft_pct:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with mc3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">Missing skills</div>
                      <div class="metric-value">{len(missing_tech) + len(missing_soft)}</div>
                      <div class="metric-sub">Tech: {missing_tech_pct:.1f}% • Soft: {missing_soft_pct:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with mc4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">Match rate</div>
                      <div class="metric-value">{coverage_pct:.1f}%</div>
                      <div class="metric-sub">JD skills covered</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Donut visualizations
            try:
                if HAS_PLOTLY:
                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        specs=[[{"type": "domain"}, {"type": "domain"}]],
                        subplot_titles=("Candidate: Technical vs Soft", "Job: Technical vs Soft"),
                    )
                    fig.add_trace(
                        go.Pie(
                            labels=["Technical", "Soft"],
                            values=[len(tech_r), len(soft_r)],
                            hole=0.55,
                            textinfo="percent+label",
                            marker=dict(colors=["#7b2ff7", "#00d0ff"])
                        ),
                        1,
                        1,
                    )
                    fig.add_trace(
                        go.Pie(
                            labels=["Technical", "Soft"],
                            values=[len(tech_j), len(soft_j)],
                            hole=0.55,
                            textinfo="percent+label",
                            marker=dict(colors=["#ff7b2f", "#2fff7b"])
                        ),
                        1,
                        2,
                    )
                    fig.update_layout(
                        height=380,
                        showlegend=True,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
                    axes[0].pie(
                        [len(tech_r), len(soft_r)],
                        labels=["Technical", "Soft"],
                        autopct="%1.1f%%",
                    )
                    axes[0].set_title("Candidate")
                    axes[1].pie(
                        [len(tech_j), len(soft_j)],
                        labels=["Technical", "Soft"],
                        autopct="%1.1f%%",
                    )
                    axes[1].set_title("Job")
                    st.pyplot(fig)
            except Exception as e:
                st.warning("Visualization error: " + str(e))

            # Detailed lists
            st.markdown("### 📋 Detailed Skill Lists")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Candidate — Technical**")
                if tech_r:
                    for s in tech_r:
                        st.markdown(
                            f"<span class='skill-badge'>{s}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("None found")
                st.markdown("**Candidate — Soft**")
                if soft_r:
                    for s in soft_r:
                        st.markdown(
                            f"<span class='skill-badge'>{s}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("None found")
            with col_b:
                st.markdown("**Job — Technical (required)**")
                if tech_j:
                    for s in tech_j:
                        st.markdown(
                            f"<span class='skill-badge'>{s}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("None found")
                st.markdown("**Job — Soft (required)**")
                if soft_j:
                    for s in soft_j:
                        st.markdown(
                            f"<span class='skill-badge'>{s}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("None found")

            # Gaps & extras
            if show_missing:
                st.markdown("### ⚠️ Gaps & Extras")
                gap_col, extra_col = st.columns(2)
                with gap_col:
                    st.markdown("**Missing Technical**")
                    st.write(missing_tech or "None")
                    st.markdown("**Missing Soft**")
                    st.write(missing_soft or "None")
                with extra_col:
                    st.markdown("**Extra Technical (candidate has but JD doesn't)**")
                    st.write(extra_tech or "None")
                    st.markdown("**Extra Soft**")
                    st.write(extra_soft or "None")

            # Skill overlap (technical)
            st.markdown("### 📊 Skill Overlap Score (Technical)")
            all_tech = sorted(set(tech_r + tech_j))
            if all_tech:
                comp_df = pd.DataFrame(
                    {
                        "skill": all_tech,
                        "candidate": [1 if s in tech_r else 0 for s in all_tech],
                        "job": [1 if s in tech_j else 0 for s in all_tech],
                    }
                )
                try:
                    if HAS_PLOTLY:
                        fig_comp = go.Figure()
                        fig_comp.add_trace(
                            go.Bar(
                                x=comp_df["skill"],
                                y=comp_df["candidate"],
                                name="Candidate",
                            )
                        )
                        fig_comp.add_trace(
                            go.Bar(
                                x=comp_df["skill"],
                                y=comp_df["job"],
                                name="Job",
                            )
                        )
                        fig_comp.update_layout(
                            barmode="group",
                            xaxis_tickangle=-40,
                            height=380,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        ax = comp_df.plot(
                            kind="bar", x="skill", y=["candidate", "job"], figsize=(10, 4)
                        )
                        ax.set_xticklabels(all_tech, rotation=45, ha="right")
                        st.pyplot(ax.get_figure())
                except Exception as e:
                    st.warning("Comparison chart error: " + str(e))
            else:
                st.info("No technical skills to compare.")

            # Highlighted previews
            if highlight_toggle:
                st.markdown("### 🔦 Highlighted Previews")
                union_skills_for_highlight = sorted(
                    set([s.lower() for s in tech_r + soft_r + tech_j + soft_j]),
                    key=lambda x: -len(x),
                )
                st.markdown("**Resume (highlighted)**")
                st.markdown(
                    highlight_text_html(resume_text, union_skills_for_highlight),
                    unsafe_allow_html=True,
                )
                st.markdown("**Job Description (highlighted)**")
                st.markdown(
                    highlight_text_html(jd_text, union_skills_for_highlight),
                    unsafe_allow_html=True,
                )

            # Per-skill context extraction (for both resume & JD)
            union_all_skills = sorted(set(tech_r + soft_r + tech_j + soft_j))
            resume_contexts = find_skill_context_sentences(resume_text, union_all_skills, max_sentences=1)
            jd_contexts = find_skill_context_sentences(jd_text, union_all_skills, max_sentences=1)

            # Build detected-skills CSV (skill + source + context)
            skill_rows = []
            for s in union_all_skills:
                in_candidate = s in tech_r or s in soft_r
                in_job = s in tech_j or s in soft_j
                source = (
                    "Both"
                    if in_candidate and in_job
                    else ("Candidate" if in_candidate else ("Job" if in_job else "Unknown"))
                )
                skill_rows.append(
                    {
                        "skill": s,
                        "source": source,
                        "candidate_sentence": " | ".join(resume_contexts.get(s, [])),
                        "job_sentence": " | ".join(jd_contexts.get(s, [])),
                    }
                )

            df_skills_list = pd.DataFrame(skill_rows)

            # Interactive per-skill expanders inline
            st.markdown("### 🧾 Interactive Skill Context Explorer")
            for row in skill_rows:
                skill = row["skill"]
                with st.expander(f"{skill} — {row['source']}"):
                    st.write("**Candidate sentence:**")
                    st.write(row.get("candidate_sentence") or "—")
                    st.write("**Job sentence:**")
                    st.write(row.get("job_sentence") or "—")

            # Download: CSV list of detected skills (flat)
            try:
                csv_skills = df_skills_list.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download detected skills (CSV)",
                    csv_skills,
                    file_name="detected_skills_list.csv",
                )
            except Exception as e:
                st.warning("CSV skill list export failed: " + str(e))

            # Export: Excel multi-sheet with padding fix (ensure equal column lengths)
            try:
                # Candidate sheet: pad tech/soft to equal length
                t_len = len(tech_r)
                s_len = len(soft_r)
                max_candidate = max(t_len, s_len, 0)
                tech_col = tech_r + [""] * (max_candidate - t_len)
                soft_col = soft_r + [""] * (max_candidate - s_len)
                df_candidate = pd.DataFrame({"technical": tech_col, "soft": soft_col})

                # Job sheet: pad tech/soft to equal length
                jt_len = len(tech_j)
                js_len = len(soft_j)
                max_job = max(jt_len, js_len, 0)
                jtech_col = tech_j + [""] * (max_job - jt_len)
                jsoft_col = soft_j + [""] * (max_job - js_len)
                df_job = pd.DataFrame({"technical": jtech_col, "soft": jsoft_col})

                # Missing sheet: pad to equal length
                mt_len = len(missing_tech)
                ms_len = len(missing_soft)
                max_missing = max(mt_len, ms_len, 0)
                mtech_col = missing_tech + [""] * (max_missing - mt_len)
                msoft_col = missing_soft + [""] * (max_missing - ms_len)
                df_missing = pd.DataFrame({"missing_technical": mtech_col, "missing_soft": msoft_col})

                # Extra sheet: pad to equal length
                et_len = len(extra_tech)
                es_len = len(extra_soft)
                max_extra = max(et_len, es_len, 0)
                etech_col = extra_tech + [""] * (max_extra - et_len)
                esoft_col = extra_soft + [""] * (max_extra - es_len)
                df_extra = pd.DataFrame({"extra_technical": etech_col, "extra_soft": esoft_col})

                # Skill context sheet (already rectangular)
                df_context = df_skills_list.rename(columns={
                    "skill": "skill",
                    "source": "source",
                    "candidate_sentence": "candidate_sentence",
                    "job_sentence": "job_sentence",
                })

                excel_buffer = io.BytesIO()
                writer_engine = None
                last_exc = None
                for engine in ("openpyxl", "xlsxwriter"):
                    try:
                        excel_buffer.seek(0)
                        excel_buffer.truncate(0)
                        with pd.ExcelWriter(excel_buffer, engine=engine, datetime_format="yyyy-mm-ddThh:mm:ss") as writer:
                            df_candidate.to_excel(writer, sheet_name="candidate", index=False)
                            df_job.to_excel(writer, sheet_name="job", index=False)
                            df_missing.to_excel(writer, sheet_name="missing", index=False)
                            df_extra.to_excel(writer, sheet_name="extra", index=False)
                            df_context.to_excel(writer, sheet_name="skill_context", index=False)
                            meta = {
                                "generated_at": datetime.now().isoformat(),
                                "resume_source": ("uploaded" if uploaded_resume else "pasted"),
                                "jd_source": ("uploaded" if uploaded_jd else "pasted"),
                                "developer_sample_path": SAMPLE_PDF_URL,
                                "coverage_pct": coverage_pct
                            }
                            pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)
                        writer_engine = engine
                        break
                    except Exception as ex:
                        last_exc = ex
                        continue

                if writer_engine is None:
                    import zipfile
                    st.warning("Excel writer engines not available. Providing ZIP of CSVs instead.")
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, mode="w") as zf:
                        zf.writestr("candidate.csv", df_candidate.to_csv(index=False))
                        zf.writestr("job.csv", df_job.to_csv(index=False))
                        zf.writestr("missing.csv", df_missing.to_csv(index=False))
                        zf.writestr("extra.csv", df_extra.to_csv(index=False))
                        zf.writestr("skill_context.csv", df_context.to_csv(index=False))
                        meta = {
                            "generated_at": datetime.now().isoformat(),
                            "resume_source": ("uploaded" if uploaded_resume else "pasted"),
                            "jd_source": ("uploaded" if uploaded_jd else "pasted"),
                            "developer_sample_path": SAMPLE_PDF_URL,
                            "coverage_pct": coverage_pct
                        }
                        zf.writestr("meta.json", json.dumps(meta, indent=2))
                    zip_buffer.seek(0)
                    st.download_button("Download exports (ZIP of CSVs)", zip_buffer.getvalue(), file_name="skill_exports_zip.zip")
                else:
                    excel_buffer.seek(0)
                    st.download_button("Download full Excel export (multi-sheet)", excel_buffer.getvalue(), file_name="skill_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning(f"Excel export failed: {e}")

            # Prepare PDF report context
            percentages = {
                "candidate_tech_pct": candidate_tech_pct,
                "candidate_soft_pct": candidate_soft_pct,
                "jd_tech_pct": jd_tech_pct,
                "jd_soft_pct": jd_soft_pct,
                "missing_tech_pct": missing_tech_pct,
                "missing_soft_pct": missing_soft_pct,
                "extra_tech_pct": extra_tech_pct,
                "extra_soft_pct": extra_soft_pct
            }

            # Build skill_context_rows for PDF builder
            skill_context_rows = []
            for row in skill_rows:
                skill_context_rows.append((row["skill"], {
                    "candidate_sentence": row["candidate_sentence"],
                    "job_sentence": row["job_sentence"],
                    "source": row["source"]
                }))

            pdf_context = {
                "total_candidate": total_candidate,
                "total_required": total_required,
                "missing_total": len(missing_tech) + len(missing_soft),
                "coverage_pct": coverage_pct,
                "percentages": percentages,
                "tech_r": tech_r,
                "soft_r": soft_r,
                "tech_j": tech_j,
                "soft_j": soft_j,
                "missing_tech": missing_tech,
                "missing_soft": missing_soft,
                "extra_tech": extra_tech,
                "extra_soft": extra_soft,
                "skill_context_rows": skill_context_rows
            }

            # Download PDF report (try ReportLab, else fallback to text)
            try:
                if HAS_REPORTLAB:
                    pdf_bytes = generate_pdf_report_bytes(pdf_context)
                    st.download_button("Download PDF Report", pdf_bytes, file_name="skill_report.pdf", mime="application/pdf")
                else:
                    # fallback: create a plain-text report and offer as .txt
                    text_report = io.StringIO()
                    text_report.write("SkillGapAI - Skill Extraction Report\n")
                    text_report.write(f"Generated: {datetime.now().isoformat()}\n\n")
                    text_report.write("Snapshot:\n")
                    text_report.write(f" Candidate skills (total): {total_candidate}\n")
                    text_report.write(f" JD skills (total): {total_required}\n")
                    text_report.write(f" Missing skills (total): {len(missing_tech) + len(missing_soft)}\n")
                    text_report.write(f" Match rate: {coverage_pct:.1f}%\n\n")
                    text_report.write("Percentages:\n")
                    for k,v in percentages.items():
                        text_report.write(f" {k}: {v:.1f}%\n")
                    text_report.write("\nSkill list (candidate tech):\n")
                    for s in tech_r:
                        text_report.write(f" - {s}\n")
                    text_report.seek(0)
                    st.download_button("Download text Report (ReportLab not installed)", text_report.getvalue().encode("utf-8"), file_name="skill_report.txt", mime="text/plain")
            except Exception as e:
                st.warning(f"PDF export failed: {e}")

    else:
        st.info("Click 'Analyze skills' to extract and compare skills. Upload or paste text above.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div style='margin-top:16px; color:#95a3b3; text-align:center;'>Milestone 2 • Skill Extraction • SkillGapAI • Developed by Anilkumar</div>",
    unsafe_allow_html=True,
)


