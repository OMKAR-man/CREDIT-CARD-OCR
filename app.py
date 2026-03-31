import streamlit as st
import cv2
import numpy as np
from utils import detect_card, extract_text_and_lines, extract_details, save_to_csv
from PIL import Image
import pandas as pd
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Card OCR AI", layout="wide", page_icon="💳")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.stApp { background: #0f1117; color: #e8e8e8; }

.card-field {
    background: #1a1d27;
    border: 1px solid #2d3142;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.card-field .label {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
    font-family: 'Space Mono', monospace;
}
.card-field .value {
    font-size: 18px;
    font-weight: 600;
    color: #f0f0f0;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
}
.card-field .value.empty { color: #3d4052; font-style: italic; font-size: 14px; }

.scanbot-container {
    border: 2px solid #2d3142;
    border-radius: 16px;
    overflow: hidden;
    background: #0a0c12;
}
</style>
""", unsafe_allow_html=True)

CSV_FILE = "data.csv"

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Card OCR AI")
    st.markdown("---")
    option = st.selectbox("Mode", ["📤 Upload Image", "📷 Live Camera"])
    st.markdown("---")

    st.subheader("📁 Saved Records")
    if os.path.exists(CSV_FILE):
        try:
            df_saved = pd.read_csv(CSV_FILE, encoding="utf-8", on_bad_lines="skip")
            st.metric("Total Cards Saved", len(df_saved))
        except:
            df_saved = pd.read_csv(CSV_FILE, encoding="utf-8", errors="ignore")

        if st.button("🗑️ Clear All Records"):
            os.remove(CSV_FILE)
            st.success("Cleared.")
            st.rerun()
    else:
        st.info("No records yet.")

    show_debug = st.checkbox("🔍 Show Raw OCR Text", value=False)

# ── UI Helpers ─────────────────────────────────────────────────
def display_details_card(details):
    col1, col2 = st.columns(2)
    fields = [
        ("👤 Name", "Name", col1),
        ("💳 Account No", "Account No", col2),
        ("🏦 Bank Name", "Bank Name", col1),
        ("📅 Valid Thru", "Valid Thru", col2),
    ]
    for label, key, col in fields:
        val = details.get(key) or ""
        display_val = val if val else "Not detected"
        cls = "value" if val else "value empty"
        with col:
            st.markdown(f"""
            <div class="card-field">
                <div class="label">{label}</div>
                <div class="{cls}">{display_val}</div>
            </div>
            """, unsafe_allow_html=True)

def show_csv_table():
    if os.path.exists(CSV_FILE):
        st.subheader("📋 Saved Card Records")
        df = pd.read_csv(CSV_FILE, encoding="utf-8", on_bad_lines="skip")
        st.dataframe(df, use_container_width=True)

def handle_save(details):
    result = save_to_csv(details, CSV_FILE)
    if result == "duplicate":
        st.warning("⚠️ Already saved (duplicate card number).")
    elif result == "invalid":
        st.error("❌ No card number detected — cannot save.")
    else:
        st.success("✅ Saved successfully!")

# ══════════════════════════════════════════════════════════════
#  MODE: UPLOAD IMAGE
# ══════════════════════════════════════════════════════════════
if option == "📤 Upload Image":
    st.title("📤 Upload Card Image")
    st.caption("Flat, well-lit photo · Card fills most of the frame · No glare")

    file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).convert("RGB")

        # Resize improves OCR accuracy
        image = image.resize((900, 550))

        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        card_bgr, bbox = detect_card(frame_bgr)

        # IMPORTANT FIX: fallback
        if card_bgr is None:
            card_bgr = frame_bgr

        display = frame.copy()
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 130), 3)
            st.image(display, caption="✅ Card region detected", use_container_width=True)
        else:
            st.image(display, caption="⚠️ Using full image", use_container_width=True)

        if show_debug:
            st.image(card_bgr, caption="Detected Card")

        with st.spinner("Running OCR…"):
            lines, full_text = extract_text_and_lines(card_bgr)
            details = extract_details(full_text, lines)

        if show_debug:
            with st.expander("🔍 Raw OCR lines"):
                for i, ln in enumerate(lines):
                    st.code(f"Line {i+1}: {ln}")

        st.subheader("Extracted Details")
        display_details_card(details)

        with st.expander("✏️ Manually correct before saving"):
            details["Name"]       = st.text_input("Name", value=details.get("Name", ""))
            details["Account No"] = st.text_input("Account No", value=details.get("Account No", ""))
            details["Valid Thru"] = st.text_input("Valid Thru", value=details.get("Valid Thru", ""))
            details["Bank Name"]  = st.text_input("Bank Name", value=details.get("Bank Name", ""))

        if st.button("💾 Save to CSV"):
            handle_save(details)

    st.divider()
    show_csv_table()

# ══════════════════════════════════════════════════════════════
#  MODE: LIVE CAMERA
# ══════════════════════════════════════════════════════════════
elif option == "📷 Live Camera":
    st.title("📷 Live Card Scanner")

    html_path = os.path.join(os.path.dirname(__file__), "camera.html")

    if os.path.exists(html_path):
        # ✅ FIXED Unicode error here
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        components.html(html_content, height=720, scrolling=False)
    else:
        st.error("❌ camera.html not found")

    st.markdown("---")

    st.subheader("📝 Manual Entry")
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Name")
            bank = st.text_input("🏦 Bank Name")
        with col2:
            accno = st.text_input("💳 Account No")
            expiry = st.text_input("📅 Valid Thru")

        if st.form_submit_button("💾 Save to CSV"):
            handle_save({
                "Name": name,
                "Account No": accno,
                "Valid Thru": expiry,
                "Bank Name": bank,
            })

    st.divider()
    show_csv_table()