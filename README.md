# 🏥 Heart Failure Prediction API 🚀

##  Overview
The **Heart Failure Prediction API** is a **Flask-based machine learning model** that predicts the risk of heart failure using clinical data. It supports **real-time predictions** via JSON input and **batch predictions** through file uploads (CSV, Excel, JSON, TXT).

This project includes:
- **Data Preprocessing & Feature Encoding**
- **Machine Learning Model (Random Forest)**
- **REST API with Flask**

---

## Features
✅ **Real-time heart failure prediction** based on patient data  
✅ **Batch predictions** via file uploads  
✅ **Data pre-processing (encoding & scaling) handled automatically**  
✅ **GitHub Actions CI/CD for automated deployment**  

---

##  Technologies Used
- **Python 3.12+**
- **Flask** (for API)
- **Scikit-learn** (for machine learning)
- **Pandas & NumPy** (for data processing)
- **Joblib** (for model persistence)

---

## 📌 Installation & Setup
### 🔹 **1. Clone the Repository**
bash
git clone https://github.com/Ruthvik14/Heart_Failure_Prediction.git
cd Heart_Failure_Prediction

### 🔹 **2. Create a Virtual Environment
   python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate

 ### 🔹 **3. Install Dependencies
pip install -r requirements.txt

### 🔹 **4. Train the Model & Save Artifacts
python train_model.py
