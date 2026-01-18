import os
import re

import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from fpdf import FPDF
from dotenv import load_dotenv


# ======================================================
# LOAD ENVIRONMENT VARIABLES
# ======================================================
load_dotenv()


# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="AI Exam Generator Pro",
    page_icon="📝",
    layout="wide"
)

st.title("📝 AI Exam Paper Generator")
st.write("Convert your syllabus PDF into a structured university-style exam paper.")


# ======================================================
# SIDEBAR CONFIGURATION
# ======================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    st.write("API Key Loaded:", bool(gemini_api_key))

    # 🔴 Model name kept exactly as requested
    model_name = "gemini-2.5-flash-lite"

    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    total_marks = st.selectbox("Exam Type", ["50 Marks", "75 Marks", "100 Marks"])


# ======================================================
# EMBEDDING MODEL
# ======================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embedding_model()


# ======================================================
# TEXT UTILITIES
# ======================================================
def clean_text(text: str) -> str:
    text = re.sub(r"Page\s+\d+", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_for_pdf(text: str) -> str:
    # Remove markdown artifacts
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Improve numbering spacing
    text = re.sub(r"\n(\d+\.)", r"\n\n\1", text)

    # Replace separators
    text = text.replace("---", "\n" + "-" * 45 + "\n")

    return text.strip()


def extract_topics(text: str, k: int = 15):
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 30]

    filtered = []
    for line in lines:
        if any(
            keyword in line.lower()
            for keyword in [
                "text books",
                "references",
                "bibliography",
                "course outcomes",
            ]
        ):
            break
        filtered.append(line)

    if not filtered:
        return []

    embeddings = np.array(embed_model.encode(filtered)).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    _, indices = index.search(centroid, min(k, len(filtered)))

    return [filtered[i] for i in indices[0]]


def exam_structure(marks: str) -> str:
    structures = {
        "50 Marks": (
            "Section A: 10 MCQs (1 mark each)\n"
            "Section B: 4 Short Answer Questions (5 marks each)\n"
            "Section C: 2 Long Answer Questions (10 marks each)"
        ),
        "75 Marks": (
            "Section A: 15 MCQs (1 mark each)\n"
            "Section B: 5 Short Answer Questions (5 marks each)\n"
            "Section C: 2 Long Answer Questions (15 marks each)\n"
            "Section D: 1 Essay Question (5 marks)"
        ),
        "100 Marks": (
            "Section A: 20 MCQs (1 mark each)\n"
            "Section B: 6 Short Answer Questions (5 marks each)\n"
            "Section C: 3 Long Answer Questions (15 marks each)\n"
            "Section D: 1 Essay Question (15 marks)"
        ),
    }
    return structures.get(marks, "")


# ======================================================
# PDF GENERATION
# ======================================================
def create_pdf(text: str, title: str):
    pdf = FPDF()
    pdf.add_page()

    margin = 20
    pdf.set_margins(margin, margin, margin)
    pdf.set_auto_page_break(auto=True, margin=margin)

    width = pdf.w - 2 * margin

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(width, 10, title.upper(), ln=True, align="C")

    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(
        width,
        8,
        f"Generated AI Exam | Difficulty: {difficulty}",
        ln=True,
        align="C",
    )
    pdf.ln(6)

    pdf.set_font("Helvetica", size=11)

    # Paragraph-based rendering
    for para in text.split("\n\n"):
        para = para.strip()

        if not para:
            pdf.ln(3)
            continue

        if para.upper().startswith("SECTION"):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
        else:
            pdf.set_font("Helvetica", size=11)

        safe_para = para.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(width, 9, safe_para)
        pdf.ln(1)

    pdf_output = pdf.output(dest="S")
    return (
        bytes(pdf_output)
        if isinstance(pdf_output, bytearray)
        else pdf_output.encode("latin-1")
    )


# ======================================================
# MAIN APPLICATION LOGIC
# ======================================================
uploaded_file = st.file_uploader(
    "📄 Step 1: Upload Syllabus (PDF)",
    type="pdf"
)

text_data = ""

if uploaded_file:
    with st.status("Reading PDF..."):
        doc = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )
        for page in doc:
            text_data += clean_text(page.get_text())

    st.success("✅ Syllabus Loaded")


if st.button("🚀 Step 2: Generate Question Paper"):
    if not gemini_api_key:
        st.error("❌ GEMINI_API_KEY missing")
        st.stop()

    if not text_data.strip():
        st.error("❌ Upload syllabus PDF first")
        st.stop()

    with st.spinner("Generating exam paper..."):
        try:
            topics = extract_topics(text_data)
            topics_summary = "\n".join(f"- {topic}" for topic in topics)

            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(model_name)

            prompt = f"""
You are a strict University Professor.

Create a formal {total_marks} examination paper.

TOPICS:
{topics_summary}

STRUCTURE:
{exam_structure(total_marks)}

DIFFICULTY: {difficulty}

RULES:
- Plain text only
- No answers
- MCQs must have (A)(B)(C)(D)
- University exam tone
"""

            response = model.generate_content(prompt)

            if not response.text.strip():
                st.error("❌ Empty AI response")
                st.stop()

            exam_text = normalize_for_pdf(response.text)

            st.subheader("📝 Exam Paper Preview")
            st.text_area("Preview", exam_text, height=450)

            pdf_bytes = create_pdf(
                exam_text,
                f"{total_marks} Examination"
            )

            st.download_button(
                "⬇️ Download Exam PDF",
                data=pdf_bytes,
                file_name="Question_Paper.pdf",
                mime="application/pdf",
            )

        except Exception as error:
            st.error(f"❌ System Error: {error}")
