#  Transformers - Lab 3 Overview

Welcome to **Lab 3** of the Deep Learning Applications course!  
This lab focuses on working with **Transformers** and the **HuggingFace ecosystem** to tackle NLP and vision tasks via feature extraction, fine-tuning, and zero-shot learning.

---

##  Subfolders

###  [`1-2/`](./1-2) — Sentiment Analysis with DistilBERT
Includes **Exercise 1** and **Exercise 2**, designed to be explored via interactive shell for a more engaging experience.

####  Exercise 1 — Feature Extraction
**Goal**: Use a pre-trained DistilBERT model to perform sentiment analysis on the *Rotten Tomatoes* movie review dataset.  
- `1.1`: Load and explore the dataset using HuggingFace `datasets`.  
- `1.2`: Load a pre-trained DistilBERT model and tokenizer. Test it on example text inputs.  
- `1.3`: Extract `[CLS]` token embeddings and train a simple classifier (e.g., SVM) as a **baseline**.

####  Exercise 2 — Fine-Tuning DistilBERT
**Goal**: Improve performance by fine-tuning DistilBERT for sequence classification.  
- `2.1`: Tokenize and preprocess the dataset using `Dataset.map`.  
- `2.2`: Load `AutoModelForSequenceClassification` with a custom classification head.  
- `2.3`: Fine-tune using HuggingFace `Trainer`, `DataCollatorWithPadding`, and scikit-learn metrics.


---

###  [`3.2/`](./3.2) — Fine-Tuning CLIP for Vision Tasks
**Goal**: Explore and fine-tune a **CLIP** model on a small image classification task.

- Use a pre-trained CLIP model (e.g., `openai/clip-vit-base-patch16`) for **zero-shot** evaluation on a dataset like ImageNette.
- Apply **parameter-efficient fine-tuning** techniques (e.g., adapters, LoRA) to adapt CLIP to the target task.
- Compare pre- and post-fine-tuning performance.
