# 🏨 Hotel Booking Demand & Cancellation Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3%2B-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-v2.0%2B-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end exploratory data analysis (EDA) and predictive modeling pipeline to forecast hotel booking cancellations. Built with **Pandas**, **Seaborn**, and **Scikit-Learn** using a leak-free `Pipeline` + `ColumnTransformer` architecture that addresses class imbalance.

---

## 📌 Project Overview

Cancellation rates disrupt hotel revenue forecasting, room allocation, and staffing. The goals of this project are:
1. **Analyze booking patterns** across hotel properties, seasonality, lead times, pricing (ADR), and market channels.
2. **Execute end-to-end data hygiene** by removing duplicates, handling missingness, filtering invalid zero-guest bookings, and addressing extreme pricing outliers.
3. **Build and evaluate production-ready ML pipelines** (`Logistic Regression` & `Random Forest`) to flag high-risk cancellations early for automated intervention.

---

## 📊 Dataset Summary

* **Source:** Hotel Booking Demand Dataset (`hotel_bookings.csv`)
* **Initial Records:** 119,390 rows × 32 columns
* **Post-Deduplication & Cleaning:** 87,228 distinct valid reservations
* **Target Variable:** `is_canceled` (0 = Check-Out, 1 = Canceled)
* **Cancellation Baseline Rate:** **27.52%** (post-cleaning)

---

## 🧹 Data Cleaning & Preprocessing Workflow

| Step | Action Taken | Rationale |
| :--- | :--- | :--- |
| **Duplicates Removal** | Dropped 31,994 exact duplicate rows | Prevents artificial inflation of training metrics. |
| **High Missingness** | Dropped `company` column (~94% nulls) | Avoids high cardinality and noisy sparse data. |
| **Target Leakage Prevention** | Dropped `reservation_status` & `reservation_status_date` | Prevents 100% false test accuracy. |
| **Redundant Features** | Dropped `arrival_date_year` & `arrival_date_week_number` | Prevents overfitting to historical sample years. |
| **Feature Engineering** | Created `total_stay`, `total_guests`, `room_mismatch` | Extracts signals on stay duration and guest commitment. |
| **Categorical Cardinality** | Grouped top 7 origin countries, binned the rest into `'Other'` | Controls dimensionality in One-Hot Encoding. |

---

## 📈 Exploratory Data Analysis & Visualizations

### 1. Target Distribution
After removing duplicate records, cancellations account for **27.52%** of all unique reservations.

<p align="center">
  <img src="https://github.com/nishant3937899/hotel-booking-data-analysis/blob/b18042d2cd99972a9b709fcb9ec6d537b0012237/imgs/target_dist%20(1).png" width="550"/>
</p>

---

### 2. Cancellations by Hotel Type
City hotels handle higher total booking volume but experience a substantially higher cancellation rate compared to Resort Hotels.

<p align="center">
  <img src="https://github.com/nishant3937899/hotel-booking-data-analysis/blob/b18042d2cd99972a9b709fcb9ec6d537b0012237/imgs/cancellation_by_hotel.png" width="650"/>
</p>

---

### 3. Seasonality & Monthly Demand Trend
Both properties experience a summer peak (July–August) corresponding with European holiday seasons, while winter months (November–January) record the lowest booking volume.

<p align="center">
  <img src="https://github.com/nishant3937899/hotel-booking-data-analysis/blob/b18042d2cd99972a9b709fcb9ec6d537b0012237/imgs/monthly_demand.png" width="850"/>
</p>

---

### 4. Average Daily Rate (ADR) Distribution
City Hotels command a higher median daily rate with steady pricing, whereas Resort Hotels exhibit wider seasonal price variance.

<p align="center">
  <img src="https://github.com/nishant3937899/hotel-booking-data-analysis/blob/b18042d2cd99972a9b709fcb9ec6d537b0012237/imgs/adr.png" width="600"/>
</p>

---

### 5. Lead Time Impact on Cancellations
Bookings made with short lead times (<30 days) have high fulfillment rates, whereas bookings made far in advance (>150 days) carry high cancellation risk.

<p align="center">
  <img src="https://github.com/nishant3937899/hotel-booking-data-analysis/blob/b18042d2cd99972a9b709fcb9ec6d537b0012237/imgs/leadtiome_vs_cancellation.png" width="700"/>
</p>

---

### 6. Geographic Distribution (Top 10 Origin Countries)
Portugal (`PRT`), United Kingdom (`GBR`), France (`FRA`), and Spain (`ESP`) lead in booking volume. Domestic Portuguese reservations have the highest cancellation rate.

<p align="center">
  <img src="https://github.com/nishant3937899/hotel-booking-data-analysis/blob/b18042d2cd99972a9b709fcb9ec6d537b0012237/imgs/top10.png" width="750"/>
</p>

---

### 7. Cancellation Rate by Market Segment
Online Travel Agents (Online TA) and Groups exhibit significantly higher cancellation rates compared to Direct and Corporate booking channels.

<p align="center">
  <img src="https://github.com/nishant3937899/hotel-booking-data-analysis/blob/b18042d2cd99972a9b709fcb9ec6d537b0012237/imgs/cancellation_rate.png" width="700"/>
</p>

---

## ⚙️ Machine Learning Pipeline Architecture

The machine learning workflow is bundled using Scikit-Learn's `Pipeline` and `ColumnTransformer` to ensure reproducibility:

```python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ]
)
```
## 🏆 Model Evaluation & Performance

The **Logistic Regression** model was trained using an end-to-end scikit-learn `Pipeline` and evaluated on an unseen **20% stratified test set** (17,446 reservations). By setting `class_weight='balanced'`, the classifier adjusted for class imbalance (27.52% baseline cancellation rate), achieving an **81.59% recall** on canceled bookings.

---

👤 Author
- Nishant Chandra Verma
