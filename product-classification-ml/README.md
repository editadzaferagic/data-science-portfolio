# Product Category Classification (Machine Learning)

This project focuses on building a machine learning model to automatically classify products into categories based on their titles.

It simulates a real-world e-commerce scenario where thousands of products need to be categorized efficiently and accurately.

---

## Project Overview

* Analyzed a dataset of 30,000+ products
* Cleaned and preprocessed text data
* Applied feature engineering on product titles
* Trained and evaluated multiple machine learning models
* Selected and saved the best-performing model
* Built an interactive prediction system for real-time classification

---

## Technologies Used

* Python
* pandas
* scikit-learn
* NLP (TF-IDF Vectorization)
* Jupyter Notebook / Google Colab

---

## Model Evaluation

The model was evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-score

Detailed results are available in the `/reports` folder:

* Confusion matrix
* Classification report
* Metrics summary

---

## How to Use

### 1. Train the model

Run the training script:

```bash
python train_model.py
```

### 2. Predict category

Run the prediction script:

```bash
python predict_category.py
```

Enter a product title and the model will predict its category.

---

## Project Structure

```
product-classification-ml/
│
├── notebooks/
│   └── product_category_prediction.ipynb
│
├── models/
│   └── product_category_model.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── metrics_summary.json
│
├── train_model.py
├── predict_category.py
├── requirements.txt
├── README.md
```

---

## Key Skills Demonstrated

* End-to-end machine learning pipeline
* Text classification (NLP)
* Feature engineering
* Model evaluation and selection
* Working with real-world datasets
* Writing reusable and structured code

---

## Dataset

The dataset contains product titles and their corresponding categories.
It is not included in the repository due to size.

---

## Business Value

This solution reduces manual categorization effort, improves accuracy, and speeds up product listing in e-commerce systems.

---
