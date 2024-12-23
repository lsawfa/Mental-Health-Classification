import pandas as pd
import numpy as np
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from DepressionEmo.file_io import *
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class LightGBM_MentalHealthClassifier:
    def __init__(self, train_path, model_save_path, emotion_list, val_path=None, test_path=None, dataset_type=None, label_list=None):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_save_path = model_save_path
        self.emotion_list=emotion_list
        self.label_list=label_list
        self.dataset_type=dataset_type
        self.stop_words = set(stopwords.words('english'))
        self.lemma = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.model = LGBMClassifier(objective='multiclass', learning_rate=0.5, verbose=-1)
        self.le = LabelEncoder()

    def convert_labels(self, labels):
        labels2 = []
        for idx, label in enumerate(labels):
            try:
                label = str(label[0])
            except:
                label = str(label)
                pass

            temp = []
            num = len(self.emotion_list) - len(label)
            if num == 0:
                temp = [int(x) for x in label]
            else:
                temp = ''.join(['0'] * num) + str(label)
                temp = [int(x) for x in temp]

            labels2.append(temp)
        return labels2
    
    def convert_to_multilabel_indicator(self, labels):
        num_classes = len(self.label_list)
        labels = [int(label) for label in labels]
    
        multi_label = []
        for label in labels:
            encoding = [0] * num_classes 
            encoding[label] = 1         
            multi_label.append(''.join(map(str, encoding))) 
        
        return multi_label
    
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

    def clean_text(self, text):
        '''text = tokenizer.encode_plus(text, max_length=256, add_special_tokens=True, return_token_type_ids=False, padding = "max_length", return_attention_mask=True, return_tensors='pt')
        text = ' '.join(str(x) for x in text['input_ids'].tolist()[0])'''

        '''text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text, re.UNICODE) # remove punctuation

        # cleaning everything except alphabetical and numerical characters
        text = re.sub("[^a-zA-Z0-9]"," ",text)

        # tokenizing and lemmatizing
        text = nltk.word_tokenize(text.lower())
        text = [self.lemma.lemmatize(word) for word in text]

        # removing stopwords
        text = [word for word in text if word not in self.stop_words]

        # joining
        text = " ".join(text)'''

        return text

    def prepare_data(self):
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
        
        train_set = train_set + test_set + val_set
        
        for item in train_set:
            if self.dataset_type=='dataset1':
                item['text'] = self.clean_text(item['text'])
                data_labels = [item['label_id'] for item in train_set]
                df_labels = pd.DataFrame(data_labels, columns=['labels'])
            elif self.dataset_type=='dataset2':
                item[0] = self.clean_text(item[0])
                data_labels = [item[1] for item in train_set]
                data_labels=data_labels[1:]
                data_labels = self.convert_to_multilabel_indicator(data_labels)
                df_labels = pd.DataFrame(data_labels, columns=['labels'])
            elif self.dataset_type=='dataset3':
                item[3] = self.clean_text(item[3])
                data_labels = [item[5] for item in train_set]
                data_labels=data_labels[1:]
                data_labels = self.convert_to_multilabel_indicator(data_labels)
                df_labels = pd.DataFrame(data_labels, columns=['labels'])
        if self.dataset_type=='dataset1':
            data_features = self.vectorizer.fit_transform([item['text'] for item in train_set])
            data_features = data_features.toarray()
        elif self.dataset_type=='dataset2':
            df_labels['labels'] = df_labels['labels'].apply(lambda x: ''.join(map(str, x)))
            train_set = train_set[1:]
            data_features = self.vectorizer.fit_transform([item[0] for item in train_set])
            data_features = data_features.toarray()
        elif self.dataset_type=='dataset3':
            train_set = train_set[1:]
            data_features = self.vectorizer.fit_transform([item[3] for item in train_set])
            data_features = data_features.toarray()
        
        return data_features, np.asarray(df_labels)

    def train_model(self):
        data_features, df_labels = self.prepare_data()
        x_train, x_test, y_train, y_test = train_test_split(
            data_features, df_labels, test_size=0.3, train_size=0.7, random_state=0
        )
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=0.5, random_state=0
        )
        
        if len(np.unique(y_train))==2:
            self.model = LGBMClassifier(objective='binary', learning_rate=0.5, verbose=-1)

        y_train = self.le.fit_transform(y_train.ravel())

        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)
        predictions = self.le.inverse_transform(predictions)
        predictions = self.convert_labels(predictions)
        self.model.booster_.save_model(self.model_save_path)

        y_test = self.convert_labels(y_test)

        f1_mi = f1_score(y_true=y_test, y_pred=predictions, average='micro')
        re_mi = recall_score(y_true=y_test, y_pred=predictions, average='micro')
        pre_mi = precision_score(y_true=y_test, y_pred=predictions, average='micro')

        f1_mac = f1_score(y_true=y_test, y_pred=predictions, average='macro')
        re_mac = recall_score(y_true=y_test, y_pred=predictions, average='macro')
        pre_mac = precision_score(y_true=y_test, y_pred=predictions, average='macro')

        result = {
            'f1_micro': f1_mi,
            'recall_micro': re_mi,
            'precision_micro': pre_mi,
            'f1_macro': f1_mac,
            'recall_macro': re_mac,
            'precision_macro': pre_mac
        }

        print(result)
        return result

# Example usage:
# classifier = TextClassifier(
#     train_path='Dataset/train.json',
#     val_path='Dataset/val.json',
#     test_path='Dataset/test.json',
#     model_save_path='./Model/LightGBM.txt',
#     emotion_list=['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
# )

# classifier.train_model()
