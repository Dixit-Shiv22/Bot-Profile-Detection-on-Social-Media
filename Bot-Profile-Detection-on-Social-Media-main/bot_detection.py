import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from cryptography.fernet import Fernet

# Streamlit page configuration
st.set_page_config(
    page_title="Twitter Bot Detector",
    page_icon="🤖",
    layout="wide"
)

# Generate encryption key (store securely in production)
key = Fernet.generate_key()
cipher = Fernet(key)

# Function to encrypt sensitive data
def encrypt_data(data):
    data = str(data)  # Ensure data is always a string
    return cipher.encrypt(data.encode()).decode()

@st.cache_resource
def load_model_and_tokenizer():
    """Load and cache the BERT model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def process_input(text, tokenizer, model, device):
    """Process individual text input"""
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    confidence = torch.softmax(output.logits, dim=1)[0][1].item()
    prediction = torch.argmax(output.logits, dim=1).item()
    return "Bot" if prediction == 1 else "Human", confidence

def analyze_data(_user_data, tokenizer, model, device):
    """Enhanced analysis function with temporal and hashtag features"""
    # Anonymize sensitive data
    _user_data['User ID'] = _user_data['User ID'].apply(encrypt_data)
    _user_data['Username'] = _user_data['Username'].apply(encrypt_data)
    
    # Convert creation dates to datetime
    _user_data['Created At'] = pd.to_datetime(_user_data['Created At'], dayfirst=True)
    
    # Add temporal features
    _user_data['Account Age'] = (pd.to_datetime('today') - _user_data['Created At']).dt.days
    _user_data['Posting Frequency'] = _user_data.groupby('User ID')['Created At'].diff().dt.total_seconds().fillna(0)
    
    # Add hashtag analysis
    _user_data['Hashtag Count'] = _user_data['Hashtags'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    
    # Detect anomalies with enhanced features
    features = ['Retweet Count', 'Mention Count', 'Follower Count', 
                'Account Age', 'Posting Frequency', 'Hashtag Count']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(_user_data[features])
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_results = iso_forest.fit_predict(scaled_features)
    
    # Process tweets
    predictions, confidences = [], []
    progress_bar = st.progress(0)
    
    for idx, row in _user_data.iterrows():
        tweet_text = row['Tweet']
        result, confidence = process_input(tweet_text, tokenizer, model, device)
        predictions.append(1 if result == "Bot" else 0)
        confidences.append(confidence)
        progress_bar.progress((idx + 1) / len(_user_data))
    
    # Combine predictions
    _user_data['Text Confidence'] = confidences
    _user_data['Behavior Score'] = (1 - (anomaly_results + 1) / 2)  # Convert to 0-1 scale
    _user_data['Composite Score'] = 0.6 * _user_data['Text Confidence'] + 0.4 * _user_data['Behavior Score']
    _user_data['Prediction'] = (_user_data['Composite Score'] > 0.65).astype(int)
    
    return _user_data

def main():
    st.title("🤖 Twitter Bot Detection System")
    st.markdown("Upload a CSV file containing Twitter data to analyze for bot accounts")

    # Add explanation toggle
    with st.expander("How does the detection work?"):
        st.markdown("""
        **Our system uses a multi-modal approach:**
        1. **Text Analysis**: BERT model analyzes tweet content for bot-like patterns
        2. **Behavior Analysis**: Isolation Forest detects anomalous activity patterns
        3. **Temporal Analysis**: Account age and posting frequency metrics
        4. **Hashtag Analysis**: Measures hashtag spam behavior
        
        Final scores combine all factors with weighting:
        - 60% Text confidence
        - 40% Behavioral anomalies
        """)
    
    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            required_columns = ['User ID', 'Username', 'Tweet', 'Retweet Count',
                              'Mention Count', 'Follower Count', 'Bot Label',
                              'Created At', 'Hashtags']
            
            if not all(col in user_data.columns for col in required_columns):
                missing = set(required_columns) - set(user_data.columns)
                st.error(f"Missing columns: {', '.join(missing)}")
                st.write("Uploaded columns:", list(user_data.columns))
                return
                
            model, tokenizer, device = load_model_and_tokenizer()
            
            # Add confidence threshold control
            threshold = st.slider("Detection Threshold", 0.5, 0.9, 0.65, 0.05,
                                help="Higher values reduce false positives but may miss subtle bots")
            
            if st.button("Start Analysis"):
                with st.spinner("Analyzing data..."):
                    analyzed_data = analyze_data(user_data, tokenizer, model, device)
                
                # Update predictions with user-selected threshold
                analyzed_data['Prediction'] = (analyzed_data['Composite Score'] > threshold).astype(int)
                
                # Metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    analyzed_data['Bot Label'], analyzed_data['Prediction'], average='binary', zero_division=0
                )
                auc_roc = roc_auc_score(analyzed_data['Bot Label'], analyzed_data['Composite Score'])
                
                # Display results
                st.success("Analysis Complete!")
                
                # Metrics columns
                cols = st.columns(4)
                cols[0].metric("Precision", f"{precision:.2%}", help="Correct bot identifications")
                cols[1].metric("Recall", f"{recall:.2%}", help="Percentage of bots detected")
                cols[2].metric("F1 Score", f"{f1:.2%}", help="Balance of precision and recall")
                cols[3].metric("AUC-ROC", f"{auc_roc:.2%}", help="Overall model discrimination")
                
                # Show sample explanations
                st.subheader("Example Detections")
                sample = analyzed_data.sample(3)
                for _, row in sample.iterrows():
                    with st.expander(f"User {row['Username']} - {'Bot' if row['Prediction'] else 'Human'}"):
                        st.markdown(f"""
                        - **Text Confidence**: {row['Text Confidence']:.0%}
                        - **Behavior Score**: {row['Behavior Score']:.0%}
                        - **Key Factors**: 
                          {'High hashtag count' if row['Hashtag Count'] > 5 else ''}
                          {'Sparse posting pattern' if row['Posting Frequency'] < 300 else ''}
                          {'Anomalous follower ratio' if row['Behavior Score'] > 0.7 else ''}
                        """)
                
                # Data download
                csv = analyzed_data.to_csv(index=False).encode()
                st.download_button(
                    label="Download Full Results",
                    data=csv,
                    file_name="bot_detection_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure: 1) Correct date format (DD-MM-YYYY), 2) Valid CSV structure")

if __name__ == "__main__":
    main()

#Trial hit apprach of this solution without deployment

# import os
# import pandas as pd
# import torch
# import torch.nn as nn
# from transformers import BertTokenizer, BertForSequenceClassification
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# import numpy as np
# import dask.dataframe as dd
# from dask.diagnostics import ProgressBar
# from cryptography.fernet import Fernet
# from fastapi import FastAPI
# from pydantic import BaseModel

# # FastAPI initialization
# app = FastAPI()

# # Generate encryption key (store securely in production)
# key = Fernet.generate_key()
# cipher = Fernet(key)

# # Function to encrypt sensitive data
# def encrypt_data(data):
#     if not isinstance(data, str):  
#         data = str(data)  # Convert to string if it's not
#     return cipher.encrypt(data.encode()).decode()

# # Load Pre-Trained Tokenizer and Model
# def load_model_and_tokenizer():
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
#     nn.init.xavier_uniform_(model.classifier.weight)
#     nn.init.zeros_(model.classifier.bias)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     return model, tokenizer, device

# # Function to Process Input
# def process_input(text, tokenizer, model, device):
#     encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
#     with torch.no_grad():
#         output = model(input_ids, attention_mask=attention_mask)
#     confidence = torch.softmax(output.logits, dim=1)[0][1].item()
#     prediction = torch.argmax(output.logits, dim=1).item()
#     return "Bot" if prediction == 1 else "Human", confidence

# # Function to Detect Anomalies
# def detect_anomalies(features):
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)
#     iso_forest = IsolationForest(contamination=0.1, random_state=42)
#     anomaly_scores = iso_forest.fit_predict(scaled_features)
#     return anomaly_scores

# # Function to Evaluate Performance
# def evaluate_performance(true_labels, predictions, confidences):
#     precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
#     auc_roc = roc_auc_score(true_labels, confidences)
#     return precision, recall, f1, auc_roc

# # Function to Analyze User Data
# def analyze_user_data(user_data, tokenizer, model, device):
#     user_activity = np.array(user_data[['Retweet Count', 'Mention Count', 'Follower Count']])
#     anomaly_results = detect_anomalies(user_activity)
#     predictions, true_labels, confidences = [], [], []
    
#     for index, row in user_data.iterrows():
#         tweet_text = row['Tweet']
#         result, confidence = process_input(tweet_text, tokenizer, model, device)
        
#         true_labels.append(row['Bot Label'])
#         predictions.append(1 if result == "Bot" else 0)
#         confidences.append(confidence)
    
#     precision, recall, f1, auc_roc = evaluate_performance(true_labels, predictions, confidences)
    
#     results = {
#         "Precision": precision,
#         "Recall": recall,
#         "F1 Score": f1,
#         "AUC-ROC": auc_roc
#     }
    
#     return results

# # FastAPI endpoint to process user input for bot detection
# class InputData(BaseModel):
#     tweet: str
#     retweet_count: int
#     mention_count: int
#     follower_count: int

# @app.post("/detect")
# def detect_bot(data: InputData):
#     model, tokenizer, device = load_model_and_tokenizer()
#     result, confidence = process_input(data.tweet, tokenizer, model, device)
#     return {"prediction": result, "confidence": confidence}

# # Function to Process CSV (used in case you need to batch process)
# def process_csv(file_path, tokenizer, model, device):
#     user_data = dd.read_csv(file_path)
#     required_columns = ['User ID', 'Username', 'Tweet', 'Retweet Count', 'Mention Count', 'Follower Count', 'Bot Label']
#     for col in required_columns:
#         if col not in user_data.columns:
#             print(f"Missing column: {col}")
#             return
    
#     # Anonymize sensitive data
#     user_data['User ID'] = user_data['User ID'].apply(encrypt_data, meta=('User ID', 'str'))
#     user_data['Username'] = user_data['Username'].apply(encrypt_data, meta=('Username', 'str'))
    
#     # Perform bot detection
#     with ProgressBar():
#         results = analyze_user_data(user_data, tokenizer, model, device)
    
#     print("Analysis Results:")
#     print(results)
    
#     output_file = os.path.splitext(file_path)[0] + "_results.csv"
#     pd.DataFrame([results]).to_csv(output_file, index=False)
#     print(f"Results saved to: {output_file}")

# # If running this script as main
# if __name__ == "__main__":
#     import uvicorn
#     # Run the FastAPI app on host and port 8000
#     uvicorn.run(app, host="0.0.0.0", port=8000)
