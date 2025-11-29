# Twitter Bot Detection System Report

## 1. Project Overview
This project aims to detect bot accounts on Twitter using a hybrid approach combining natural language processing (NLP) and anomaly detection. The system leverages a pre-trained BERT model for text classification and an Isolation Forest algorithm to identify anomalous user behavior patterns. The solution is deployed as a scalable API with FastAPI and includes robust data encryption for sensitive fields.

## 2. Technical Components

### 2.1 Data Security
**Encryption:** Sensitive data (User IDs, Usernames) is encrypted using Fernet symmetric encryption.

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()  # Securely store in production
cipher = Fernet(key)
cipher.encrypt(data.encode()).decode()
```

### 2.2 NLP Classification with BERT
**Model:** A pre-trained `bert-base-uncased` model fine-tuned for binary classification (Bot/Human).

**Initialization:** Custom weight initialization (Xavier Uniform) for the classifier layer.

**Inference:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
outputs = model(**inputs)
confidence_score = torch.softmax(outputs.logits, dim=1)[0][1].item()
```

### 2.3 Anomaly Detection
**Features:** Retweet Count, Mention Count, Follower Count.

**Algorithm:** Isolation Forest with 10% contamination assumption.

```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
```

### 2.4 Scalability
- **Parallel Processing:** Dask is used for batch processing of large CSV files.
- **API Deployment:** FastAPI endpoint (`/detect`) enables real-time predictions.

## 3. Methodology

### 3.1 Workflow
**Input Processing:**
- Text is tokenized and classified by BERT.
- User activity features are scaled and analyzed for anomalies.

**Decision Fusion:**
- Combines BERT predictions (text-based) and anomaly scores (behavior-based).

**Evaluation:**
- **Metrics:** Precision, Recall, F1 Score, AUC-ROC.

### 3.2 Key Functions
- `process_input()`: Classifies text and returns confidence scores.
- `detect_anomalies()`: Flags unusual user activity.
- `evaluate_performance()`: Computes metrics for model validation.

## 4. Strengths
- **State-of-the-Art NLP:** BERT provides high-quality text classification.
- **Hybrid Approach:** Combines linguistic and behavioral signals for robust detection.
- **Security:** Encryption protects sensitive user data.
- **Scalability:**
  - FastAPI for low-latency API responses.
  - Dask for parallelized batch processing.

## 5. Limitations
- **Computational Cost:** BERT inference is resource-intensive (GPU recommended).
- **Static Thresholds:** Isolation Forest assumes 10% anomaly rate (`contamination=0.1`).
- **Data Dependency:** Requires CSV columns: User ID, Username, Tweet, Retweet Count, Mention Count, Follower Count, Bot Label.
- **Encryption Key Management:** Key is generated dynamically; requires secure storage in production.

## 6. Performance Metrics

| Metric      | Description                                    |
|------------|-----------------------------------------------|
| Precision  | Measures correctness of bot predictions.      |
| Recall     | Evaluates detection of all actual bots.       |
| F1 Score   | Harmonic mean of precision and recall.        |
| AUC-ROC    | Assesses model’s ability to distinguish bots. |

## 7. Future Improvements

### Model Optimization
- Quantize BERT or use DistilBERT for faster inference.
- Implement dynamic thresholding for anomaly detection.

### Feature Engineering
- Add temporal features (e.g., posting frequency).

### Security Enhancements
- Use AWS KMS or HashiCorp Vault for encryption key management.

### Deployment
- Containerize with Docker for cloud deployment.
- Add rate limiting and authentication to the API.

## 8. Conclusion
This project demonstrates a robust pipeline for Twitter bot detection by integrating cutting-edge NLP techniques with anomaly detection. While the current implementation achieves functional goals, future work should focus on optimizing computational efficiency and enhancing security practices for production readiness. The modular design allows easy integration with existing social media analytics platforms.

## Appendix: Dependencies
- **Python Libraries:** torch, transformers, scikit-learn, dask, cryptography, fastapi
- **Model:** `bert-base-uncased` (Hugging Face)
- **Hardware:** GPU acceleration recommended for BERT inference.
