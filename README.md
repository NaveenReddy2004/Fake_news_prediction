
# Fake News Detection using ML & DL Models

This project aims to detect **fake vs real news articles** using various Machine Learning classifiers and Deep Learning techniques like **LSTM**. A **Streamlit web application** is also included for real-time predictions the model used in streamlit is SupportVectorClassifier whose accuracy was more than other models. 

---

## Features

- Binary classification of news as **real (0)** or **fake (1)**
- Text preprocessing pipeline (tokenization, stopword removal, stemming)
- Multiple classifiers:  
  ✅ Logistic Regression  
  ✅ SVC (Support Vector Classifier)  
  ✅ Random Forest  
  ✅ LSTM  
  ✅ BERT (Optional / for future work)  
- Confidence score with predictions  
- Live demo with **Streamlit UI**

---

## 📁 Dataset

Dataset used: A combination of fake and real news datasets from sources like Kaggle.

**Key columns used:**
- `title` or `text`: News content
- `label`: 1 (Fake), 0 (Real)

---

## Models Trained

| Model              | Accuracy | Notes                        |
|-------------------|----------|------------------------------|
| Logistic Regression |  Good baseline |
| SVC                 |  Accurate but slower |
| Random Forest       |  Best for interpretability |
| LSTM                |  Handles text sequences |
| BERT (Optional)     |  Future enhancement |

---

## Preprocessing

```python
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=300)
```

---

## Installation

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run main.py
```

Enter a news snippet in the textbox to predict if it's **real** or **fake** with confidence.

---

## Example Prediction Output

> 🟢 **Real News** (93.28% confidence)  
> 🔴 **Fake News** (87.45% confidence)

---

## 📁 Folder Structure

```
.
├── main.py                # Streamlit app
├── Fake_News_Pred.ipynb   # Training + Evaluation Notebook
├── models/                # Saved models (.pkl or .h5)
├── vectorizer/            # TF-IDF or tokenizer objects
├── requirements.txt
└── README.md
```

---

## 📚 Future Improvements

- BERT model fine-tuning and deployment
- Model evaluation dashboard
- Batch prediction from CSV input

---
