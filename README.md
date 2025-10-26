

# 🎬 Netflix Data Analysis & Recommendation System

This project analyzes and predicts Netflix content trends while also recommending similar shows and movies.

It includes:

* Data cleaning and visualization
* Machine learning models for content classification
* TF-IDF & Cosine Similarity–based recommender system
* Streamlit app for interactive use

---

## 🚀 Features

### 🔍 Data Analysis

* Cleaned dataset by handling missing values and incorrect entries
* Visualized Netflix content distribution by:

  * Type (Movies vs Shows)
  * Country of release
  * Ratings and release years

### 🤖 Machine Learning

* **Random Forest Classifier** used to predict Netflix title ratings
* Other models tested: Logistic Regression, Decision Tree, and XGBoost
* Tuned using GridSearchCV for best accuracy (~53%)

### 🎯 Recommendation System

* Built using **TF-IDF Vectorization** + **Cosine Similarity**
* Provides smart recommendations based on:

  * Title name
  * Genre
  * Country
  * Decade

### 🧠 Tech Stack

* **Python**
* **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
* **Scikit-learn**, **XGBoost**
* **Streamlit** for deployment

---

## 🗂️ Project Structure

netflix_project/
│
├── app.py                    # Streamlit app file
├── model/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── data/
│   └── netflix_data.csv
│
├── requirements.txt          # Dependencies
└── README.md                 # Project summary

---

## ⚙️ Installation

1. Clone or download this repository

2. Open terminal inside project folder

3. Install all dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Run the app:

   ```
   streamlit run app.py
   ```

---

## 📊 Results

* **Best Model:** Random Forest
* **Accuracy:** ~53% after hyperparameter tuning
* **Best Recommender Example:**

  * Input: *Stranger Things*
  * Output: `[October Faction, Trese, Curon, 46, Good Witch]`

---

## 🌐 Future Improvements

* Add hybrid recommender (content + collaborative)
* Use BERT embeddings for semantic similarity
* Host on Streamlit Cloud for public access

---

## 🧑‍💻 Author

**Prathamesh**
Machine Learning & Deep Learning Enthusiast

---


