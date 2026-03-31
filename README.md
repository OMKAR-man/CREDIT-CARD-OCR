# 💳 Card OCR AI Scanner

A smart **Credit/Debit Card OCR System** built using **Streamlit, OpenCV, and EasyOCR / Scanbot SDK** that extracts card details from images or live camera input.

---

## 🚀 Features

* 📤 Upload card images for OCR extraction
* 📷 Live camera scanner (browser-based)
* 🧠 Automatic detection of:

  * Card Number
  * Cardholder Name
  * Expiry Date
  * Bank Name
* ✏️ Manual correction before saving
* 💾 Save extracted data to CSV
* 🔍 Debug mode to view raw OCR output
* 🎯 Card boundary detection using OpenCV

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Computer Vision:** OpenCV
* **OCR Engine:** EasyOCR / Scanbot SDK
* **Data Handling:** Pandas
* **Language:** Python

---

## 📂 Project Structure

```
Card-OCR-AI/
│
├── app.py              # Main Streamlit application
├── utils.py            # OCR + detection logic
├── camera.html         # Live camera scanner (Scanbot Web SDK)
├── data.csv            # Stored card records
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/card-ocr-ai.git
cd card-ocr-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📸 Usage

### 🔹 Upload Mode

1. Select **Upload Image**
2. Upload card image
3. View extracted details
4. Edit (if needed)
5. Click **Save**

---

### 🔹 Live Camera Mode

1. Select **Live Camera**
2. Allow camera access
3. Scan your card
4. Data auto-fills
5. Save to CSV

---

## 📊 Output

Extracted data is saved in:

```
data.csv
```

Fields:

* Name
* Account No
* Valid Thru
* Bank Name

---

## 🔥 Key Highlights

* Real-time card detection
* Multi-pass OCR for higher accuracy
* Works with low-light and tilted images
* Clean UI with modern design

---

## ⚠️ Limitations

* OCR accuracy depends on image quality
* Scanbot SDK requires API key (paid)
* Live camera requires browser permissions

---

## 🚀 Future Enhancements

* 💳 Luhn validation for card numbers
* 🤖 AI-based name detection
* 📱 Mobile-first scanning UI
* ☁️ Database integration (MongoDB / Firebase)
* 🔐 Secure encryption for card data

---

## 👨‍💻 Author

**Omkar Mane**
GitHub: https://github.com/OMKAR-man

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!

---
