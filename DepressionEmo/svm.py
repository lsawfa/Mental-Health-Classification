import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import time
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer
from .file_io import *

class SVM_MentalHealthClassifier:
    def __init__(self, train_path, model_path, emotion_list, val_path=None, test_path=None, label_list=None, dataset_type=None):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.emotion_list = emotion_list
        self.label_list = label_list
        self.stop_words = stopwords.words("english")
        self.lemma = nltk.WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.dataset_type = dataset_type

    def convert_labels(self, labels):
        labels2 = []
        for label in labels:
            try:
                label = str(label[0])
            except:
                label = str(label)
            
            num = len(self.emotion_list) - len(label)
            if num == 0:
                temp = [int(x) for x in label]
            else:
                temp = [int(x) for x in ''.join(['0'] * num) + label]
            labels2.append(temp)
        return labels2
    
    def convert_to_multilabel_indicator(self, labels):
        num_classes = len(self.label_list)
        multilabels = np.zeros((len(labels), num_classes), dtype=int)

        for idx, label in enumerate(labels):
            multilabels[idx, label] = 1

        # Convert each row to a binary string
        binary_labels = [''.join(map(str, row)) for row in multilabels]
        return binary_labels

    def align_columns_list_based(self, train_data, test_data):
        train_header = train_data[0]
        test_header = test_data[0]

        extra_in_train = [col for col in train_header if col not in test_header]
        extra_in_test = [col for col in test_header if col not in train_header]

        print(f"Extra columns in train: {extra_in_train}")
        print(f"Extra columns in test: {extra_in_test}")

        for row in test_data[1:]: 
            for col in extra_in_train:
                row.append(None) 
        for row in train_data[1:]: 
            for col in extra_in_test:
                row.append(None)  

        test_reordered = [train_header]  
        test_col_idx_map = [test_header.index(col) if col in test_header else None for col in train_header]
        for row in test_data[1:]:
            reordered_row = [row[idx] if idx is not None else None for idx in test_col_idx_map]
            test_reordered.append(reordered_row)

        return train_data, test_reordered
    
    def preprocess_data(self, dataset):
        cleaned_data = []
        text = ''
        for item in dataset:
            if 'text' in item:
                text = item['text']
            elif self.dataset_type=='dataset2':
                text = item[0]
            elif self.dataset_type=='dataset3':
                text = item[3]
            text = self.tokenizer.encode_plus(
                text, max_length=256, add_special_tokens=True, return_token_type_ids=False,
                padding="max_length", return_attention_mask=True, return_tensors='pt')
            text = ' '.join(str(x) for x in text['input_ids'].tolist()[0])
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[\^\w\s]', '', text, re.UNICODE)  # Remove punctuation
            text = re.sub("[^a-zA-Z0-9]", " ", text)
            text = nltk.word_tokenize(text.lower())
            text = [self.lemma.lemmatize(word) for word in text if word not in self.stop_words]
            cleaned_data.append(" ".join(text))
        return cleaned_data

    def load_and_process_data(self):
        def load_file(path):
            if path.endswith('.json') or path.endswith('.jsonl'):
                return read_list_from_jsonl_file(path)
            elif path.endswith('.csv'):
                return read_list_from_csv_file(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")

        train_set = load_file(self.train_path)
        val_set = load_file(self.val_path) if self.val_path else []
        test_set = load_file(self.test_path) if self.test_path else []
        
        if self.dataset_type=='dataset3':
            train_set, test_set = self.align_columns_list_based(train_set, test_set)
            test_set=test_set[1:]
        full_dataset = train_set + val_set + test_set
        
        # Extract labels
        if isinstance(full_dataset[0], dict):  # JSON-style dictionary data
            labels = [item['label_id'] for item in full_dataset]
        elif isinstance(full_dataset[0], list):  # CSV-style list data
            if self.dataset_type=='dataset2':
                label_idx = 1
            elif self.dataset_type=='dataset3':
                label_idx = 5
            full_dataset = full_dataset[1:]
            labels = [int(row[label_idx]) for row in full_dataset]
            labels = self.convert_to_multilabel_indicator(labels)
        df_labels = pd.DataFrame(labels, columns=['labels'])
        cleaned_data = self.preprocess_data(full_dataset)

        bow = self.vectorizer.fit_transform(cleaned_data)
        x_train, x_test, y_train, y_test = train_test_split(bow, np.asarray(df_labels), test_size=0.3, random_state=0)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def train_model(self, x_train, y_train):
        model = SVC()
        start_time = time.time()
        model.fit(x_train, y_train)
        end_time = time.time()
        print(f"Fitting SVC took {round(end_time - start_time, 2)} seconds")
        joblib.dump(model, self.model_path)
        return model

    def evaluate_model(self, model, x_test, y_test):
        predictions = model.predict(x_test)
        y_test = self.convert_labels(y_test)
        predictions = self.convert_labels(predictions)

        metrics = {
            'f1_micro': f1_score(y_true=y_test, y_pred=predictions, average='micro'),
            'recall_micro': recall_score(y_true=y_test, y_pred=predictions, average='micro'),
            'precision_micro': precision_score(y_true=y_test, y_pred=predictions, average='micro'),
            'f1_macro': f1_score(y_true=y_test, y_pred=predictions, average='macro'),
            'recall_macro': recall_score(y_true=y_test, y_pred=predictions, average='macro'),
            'precision_macro': precision_score(y_true=y_test, y_pred=predictions, average='macro')
        }

        print(metrics)
        return metrics

# Usage example
# classifier = SVM_MentalHealthClassifier(
#     train_path='Dataset/train.json',
#     val_path='Dataset/val.json',
#     test_path='Dataset/test.json',
#     model_path='./Model/svc_model.pkl'
# )

# x_train, x_val, x_test, y_train, y_val, y_test = classifier.load_and_process_data()
# model = classifier.train_model(x_train, y_train)
# metrics = classifier.evaluate_model(model, x_test, y_test)
