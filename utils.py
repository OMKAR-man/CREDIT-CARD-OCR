import cv2
import numpy as np
import re
import pandas as pd
import easyocr
import os

# Initialise EasyOCR once (GPU=False for broad compatibility)
reader = easyocr.Reader(['en'], gpu=False)

KNOWN_BANKS = [
    "HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YES BANK", "PNB", "CANARA",
    "BOB", "UNION", "INDUSIND", "IDFC", "RBL", "FEDERAL", "HSBC", "CITIBANK",
    "CHASE", "BARCLAYS", "WELLS FARGO", "BANK OF AMERICA", "AMERICAN EXPRESS",
    "AMEX", "VISA", "MASTERCARD", "RUPAY", "DISCOVER", "STANDARD CHARTERED",
]

SKIP_WORDS = {
    "VALID", "THRU", "FROM", "DEBIT", "CREDIT", "CARD", "BANK", "PLATINUM",
    "GOLD", "CLASSIC", "SIGNATURE", "MEMBER", "SINCE", "VISA", "MASTERCARD",
    "RUPAY", "AMEX", "DISCOVER", "AUTHORIZED", "INTERNATIONAL", "WORLDWIDE",
    "NETWORK", "SECURE", "VERIFIED", "EXPIRES", "EXPIRY", "DATE", "GOOD",
    "ONLY", "THROUGH", "ACCOUNT", "SAVINGS", "CURRENT", "PREMIER",
}


# ─────────────────────────────────────────────
# PREPROCESSING  — multiple attempts, best is chosen downstream
# ─────────────────────────────────────────────
def _upscale(gray, min_width=960):
    """Upscale image so width is at least min_width."""
    h, w = gray.shape
    if w < min_width:
        scale = min_width / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    return gray


def preprocess_for_ocr(image):
    """Return a list of preprocessed variants for the OCR to try."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _upscale(gray)

    variants = []

    # Variant 1: CLAHE + unsharp mask
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    variants.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))

    # Variant 2: Adaptive threshold (for embossed/low-contrast cards)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # Variant 3: Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    return variants


# ─────────────────────────────────────────────
# CARD DETECTION
# ─────────────────────────────────────────────
def detect_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    for low, high in [(30, 100), (50, 150), (80, 200)]:
        edged = cv2.Canny(blur, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(cnt)
            if area < frame.shape[0] * frame.shape[1] * 0.10:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / h if h else 0
                if 1.1 <= aspect <= 2.2:
                    pad = 8
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(frame.shape[1] - x, w + 2 * pad)
                    h = min(frame.shape[0] - y, h + 2 * pad)
                    return frame[y:y + h, x:x + w], (x, y, w, h)

    return frame, None


# ─────────────────────────────────────────────
# SPATIAL LINE GROUPING
# ─────────────────────────────────────────────
def group_results_into_lines(results, y_thresh=15):
    if not results:
        return [], ""

    items = []
    for (bbox, text, conf) in results:
        if conf < 0.20:          # slightly more permissive
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        items.append((cy, cx, text.strip(), conf))

    if not items:
        return [], ""

    items.sort(key=lambda i: (i[0], i[1]))

    lines, current_line = [], [items[0]]
    for item in items[1:]:
        if abs(item[0] - current_line[-1][0]) <= y_thresh:
            current_line.append(item)
        else:
            current_line.sort(key=lambda i: i[1])
            lines.append(" ".join(i[2] for i in current_line))
            current_line = [item]

    current_line.sort(key=lambda i: i[1])
    lines.append(" ".join(i[2] for i in current_line))

    full_text = "\n".join(lines)
    return lines, full_text


# ─────────────────────────────────────────────
# OCR — tries multiple preprocessed variants and
#        merges the best results.
# ─────────────────────────────────────────────
def extract_text_and_lines(image):
    """
    Runs EasyOCR on up to 3 preprocessed variants of the image
    and returns the result set that yields the most text (by char count).
    """
    variants = preprocess_for_ocr(image)
    best_lines, best_text = [], ""

    for variant in variants:
        results = reader.readtext(variant, detail=1, paragraph=False)
        lines, text = group_results_into_lines(results)

        # Keep whichever variant produced the most readable text
        if len(text) > len(best_text):
            best_lines, best_text = lines, text

        # Early-exit if we already found a 16-digit card number
        if re.search(r'\d[\s\-]*\d[\s\-]*\d[\s\-]*\d[\s\-]*'
                     r'\d[\s\-]*\d[\s\-]*\d[\s\-]*\d[\s\-]*'
                     r'\d[\s\-]*\d[\s\-]*\d[\s\-]*\d[\s\-]*'
                     r'\d[\s\-]*\d[\s\-]*\d[\s\-]*\d', text):
            break

    return best_lines, best_text


def extract_text(image):
    _, full_text = extract_text_and_lines(image)
    return full_text


# ─────────────────────────────────────────────
# BANK NAME
# ─────────────────────────────────────────────
def extract_bank_name(text):
    upper = text.upper()
    for bank in KNOWN_BANKS:
        if bank in upper:
            return bank.title()
    return ""


# ─────────────────────────────────────────────
# CARDHOLDER NAME
# ─────────────────────────────────────────────
def extract_cardholder_name(lines):
    candidates = []
    for line in lines:
        line_clean = re.sub(r"[^A-Za-z\s]", "", line).strip()
        words = line_clean.split()
        if len(words) < 2 or len(words) > 4:
            continue
        if not all(w.isalpha() for w in words):
            continue
        if any(w.upper() in SKIP_WORDS for w in words):
            continue
        if all(len(w) <= 2 for w in words):
            continue
        joined = " ".join(w.capitalize() for w in words)
        candidates.append(joined)

    return max(candidates, key=len) if candidates else ""


# ─────────────────────────────────────────────
# CARD NUMBER
# ─────────────────────────────────────────────
def extract_card_number(text):
    def ocr_clean(t):
        return (t.replace('O', '0').replace('o', '0')
                 .replace('I', '1').replace('l', '1')
                 .replace('S', '5').replace('s', '5'))

    cleaned = ocr_clean(re.sub(r'[^\dOoIlSs\s\-]', ' ', text))

    # Primary: 4×4 groups
    m = re.search(r'\b(\d{4})[\s\-]*(\d{4})[\s\-]*(\d{4})[\s\-]*(\d{4})\b', cleaned)
    if m:
        digits = ''.join(m.groups())
        return ' '.join(digits[i:i + 4] for i in range(0, 16, 4))

    # Fallback: concatenated 16 digits
    digits_only = re.sub(r'\D', '', ocr_clean(text))
    nums = re.findall(r'\d{16}', digits_only)
    if nums:
        d = nums[0]
        return ' '.join(d[i:i + 4] for i in range(0, 16, 4))

    return ""


# ─────────────────────────────────────────────
# EXPIRY DATE
# ─────────────────────────────────────────────
def extract_expiry(text):
    t = text.replace('O', '0').replace('o', '0')
    t = re.sub(r'\s*[/\-]\s*', '/', t)
    matches = re.findall(r'\b(0[1-9]|1[0-2])/(\d{2}|\d{4})\b', t)
    if matches:
        m, y = matches[-1]
        if len(y) == 4:
            y = y[2:]
        return f"{m}/{y}"
    return ""


# ─────────────────────────────────────────────
# MASTER EXTRACTION
# ─────────────────────────────────────────────
def extract_details(text, lines=None):
    if lines is None:
        lines = [l for l in text.split('\n') if l.strip()]
    return {
        "Name":       extract_cardholder_name(lines),
        "Account No": extract_card_number(text),
        "Valid Thru": extract_expiry(text),
        "Bank Name":  extract_bank_name(text),
    }


# ─────────────────────────────────────────────
# CSV SAVE
# ─────────────────────────────────────────────
def save_to_csv(data, filepath="data.csv"):
    if not data.get("Account No"):
        return "invalid"

    df_new = pd.DataFrame([data])

    if os.path.exists(filepath):
        df_old = pd.read_csv(filepath)
        if "Account No" in df_old.columns:
            # Normalise before comparing
            acc = str(data["Account No"]).replace(" ", "")
            existing = df_old["Account No"].astype(str).str.replace(" ", "", regex=False)
            if (existing == acc).any():
                return "duplicate"
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(filepath, index=False)
    return "saved"