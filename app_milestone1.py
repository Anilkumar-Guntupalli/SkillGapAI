# ============================================================
# SkillGapAI – Premium Neon-Galaxy UI (Clean + Bug-Free Version)
# - Emails Removed
# - Phone Section Removed
# - Raw HTML Bug Fixed
# - Front Page Improved
# Developed for: Anilkumar
# ============================================================

import streamlit as st
import docx2txt
import PyPDF2
import re
import time
from datetime import datetime

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="SkillGapAI — Premium Parser",
    layout="wide",
    page_icon="🧠",
)

# -----------------------------------------------------------
# PREMIUM UI CSS
# -----------------------------------------------------------
st.markdown("""
<style>
:root{
    --bg: #05060b;
    --accent1: #7b2ff7;
    --accent2: #00d0ff;
    --muted: #9aa6b2;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: #e6eef8;
}

/* HERO */
.hero-wrap{
    text-align:center;
    padding:30px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.05);
    background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    box-shadow:0 20px 60px rgba(0,0,0,0.7);
}
.logo-3d{
    width:85px;height:85px;
    border-radius:20px;
    background:linear-gradient(135deg,var(--accent1),var(--accent2));
    display:flex;align-items:center;justify-content:center;
    font-size:28px;font-weight:900;color:white;
}

/* CARDS */
.glass{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.05);
    padding:20px;border-radius:15px;
    margin-bottom:18px;
}
.file-card{
    background:rgba(255,255,255,0.04);
    border-radius:12px;padding:15px;
    display:flex;justify-content:space-between;
    border:1px solid rgba(255,255,255,0.05);
}
.file-icon{
    width:52px;height:52px;border-radius:12px;
    background:linear-gradient(135deg,var(--accent1),var(--accent2));
    display:flex;align-items:center;justify-content:center;
    font-size:24px;font-weight:700;color:white;
}
.badge{
    background:rgba(255,255,255,0.08);
    padding:6px 12px;border-radius:999px;
    margin-left:8px;font-size:12px;font-weight:800;
}
.skill-badge{
    background:rgba(255,255,255,0.05);
    padding:6px 12px;border-radius:999px;
    margin:6px 6px 0 0;
    font-size:13px;border:1px solid rgba(255,255,255,0.05);
}
.footer{
    text-align:center;
    margin-top:30px;
    color:var(--muted);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# HERO SECTION
# -----------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="logo-3d">SG</div>
    <h1 style="font-size:30px;margin-top:15px;">🧠 SkillGapAI — Smart Resume & JD Parser</h1>
    <p style="color:#9aa6b2;margin-top:-5px;">
        Upload files, extract clean text, auto-detect skills, and download results instantly.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to:", ["Upload File", "Paste JD", "About Project"])
show_counts = st.sidebar.checkbox("Show text counts", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Developer: **Anilkumar**")

# -----------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_text(uploaded):
    try:
        if uploaded.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded)
            text = ""
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    text += t + "\n"
            return clean_text(text)

        elif uploaded.name.endswith(".docx"):
            return clean_text(docx2txt.process(uploaded))

        elif uploaded.name.endswith(".txt"):
            return clean_text(uploaded.read().decode("utf-8", errors="ignore"))

        else:
            return ""
    except:
        return ""

def find_skills(text):
    keywords = ["python","java","sql","aws","docker","react","node",
                "django","flask","excel","pandas","numpy","machine learning","nlp"]
    text = text.lower()
    return [k for k in keywords if k in text]


# -----------------------------------------------------------
# PAGE: UPLOAD FILE
# -----------------------------------------------------------
if menu == "Upload File":
    
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📤 Upload Resume or Job Description")
    uploader = st.file_uploader("Select a file", type=["pdf","docx","txt"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🧾 Parsed Output")

    if uploader:
        with st.spinner("Extracting..."):
            time.sleep(0.3)
            extracted = extract_text(uploader)

        size = f"{uploader.size} bytes"
        chars = len(extracted)
        words = len(extracted.split())

        # FILE CARD FIXED (NO HTML BUG)
        st.markdown(f"""
        <div class="file-card">
            <div style="display:flex;gap:12px;align-items:center;">
                <div class="file-icon">📄</div>
                <div>
                    <div style="font-weight:900;">{uploader.name}</div>
                    <div style="color:#bbb;">Size: {size}</div>
                </div>
            </div>
            <div>
                {f'<span class="badge">Chars: {chars}</span>' if show_counts else ""}
                {f'<span class="badge">Words: {words}</span>' if show_counts else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.success("File parsed successfully!")

        # Skills Section
        skills = find_skills(extracted)
        st.subheader("🧰 Skills Identified")
        if skills:
            for s in skills:
                st.markdown(f"<span class='skill-badge'>{s}</span>", unsafe_allow_html=True)
        else:
            st.write("No common skills detected.")

        st.text_area("Extracted Text", extracted, height=350)

        st.download_button("💾 Download Text", extracted, f"parsed_{uploader.name}.txt")

    else:
        st.info("Upload a file to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# PAGE: PASTE JD
# -----------------------------------------------------------
elif menu == "Paste JD":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📋 Paste Job Description")

    jd = st.text_area("Paste JD here...", height=260)

    if jd:
        cleaned = clean_text(jd)
        skills = find_skills(cleaned)

        st.subheader("🧼 Cleaned Text")
        st.text_area("Output", cleaned, height=220)

        st.subheader("🧰 Skills")
        st.write(skills or "—")

        st.download_button("💾 Download JD", cleaned, "cleaned_jd.txt")
    else:
        st.info("Paste a JD to analyze.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------------
else:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("ℹ️ About SkillGapAI")
    st.write("""
    **SkillGapAI – Milestone 1 Completed**
    
    This tool extracts clean text from:
    - PDF  
    - DOCX  
    - TXT  

    Features:
    - Premium UI  
    - Clean text extraction  
    - Skill detection  
    - Download support  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
now = datetime.now().strftime("%b %d, %Y")
st.markdown(f"<div class='footer'>SkillGapAI • Developed by <b>Anilkumar</b> • {now}</div>", unsafe_allow_html=True)
