# bert-news-topic-classifier
A transformer-based NLP project to classify news headlines into categories (World, Sports, Business, Sci/Tech) using BERT and the AG News Dataset.# BERT News Topic Classifier

This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to classify news headlines into four categories: **World, Sports, Business, and Sci/Tech**. It leverages the **AG News dataset** and Hugging Face Transformers to fine-tune a pre-trained BERT model.

---

## 🚀 Features

- Load and explore the AG News dataset
- Preprocess text and tokenize using BERT tokenizer
- Fine-tune `bert-base-uncased` for news classification
- Evaluate model using **Accuracy** and **F1-score**
- Deploy a web app with **Streamlit** to predict categories for new headlines

---

## 📊 Dataset

- Dataset: [AG News Dataset](https://huggingface.co/datasets/ag_news)
- Train samples: 120,000
- Test samples: 7,600
- Classes: 4 (World, Sports, Business, Sci/Tech)
- Usage: `from datasets import load_dataset; dataset = load_dataset("ag_news")`

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/bert-news-topic-classifier.git
Navigate into the folder:
cd bert-news-topic-classifier
Install dependencies:
pip install -r requirements.txt

👩‍💻 Usage
Preprocess and train model:
python src/train_model.py
Make predictions on new headlines:
python src/predict.py
Run the web app:
streamlit run app/app.py
👤 Author
Tooba Nadeem
