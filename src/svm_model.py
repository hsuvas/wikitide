#Statistical model: SVM

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
vectorizer =TfidfVectorizer()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import pickle
import json
import os
from os.path import join as pj

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')



def extract_features(sentence1, sentence2):
    combined = sentence1 + " " + sentence2
    feature_vector = vectorizer.transform([combined]).toarray()
    return feature_vector

def train_model(input_path,output_path):
  train_df = pd.read_csv(input_path+'/svm_train_test_data/train_df.csv') #'data/train_test_data/train_df.csv'
  unl_df = pd.read_csv(input_path+'/unlabelled/10k_test_data.csv') #'data/unlabelled/10k_test_data.csv'
  test_df = pd.read_csv(input_path+'/svm_train_test_data/test_df.csv') #'data/train_test_data/test_df.csv'


  X_train = train_df[['def_first', 'def_end']]
  y_train = train_df['label']

  # Convert def_first and def_end to a single column and fit the vectorizer
  vectorizer.fit(train_df['def_first'] + ' '+ train_df['def_end'])

  # Transform the def_first and def_end columns into numerical features
  X_train = vectorizer.transform(train_df['def_first'] + ' '+ train_df['def_end']).toarray()

  svm_model = SVC(kernel='linear',probability=True)
  svm_model.fit(X_train, y_train)
  dataset = [[a, b] for a, b in zip(unl_df['def_first'],unl_df['def_end'])]

  #test set fitting
  X_test = vectorizer.transform(test_df['def_first'] + test_df['def_end'])
  y_test_pred = svm_model.predict(X_test.toarray())
  y_test = test_df['label']
  acc_test = accuracy_score(y_test, y_test_pred)
  f1 = f1_score(y_test, y_test_pred)
  tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
  logging.info("Initial Accuracy  is: ", str(acc_test))
  logging.info("Initial F1 score is: ", str(f1))
  logging.info("Initial Confusion Matrix is(tn,fp,fn,tp): ", str(tn), str(fp), str(fn), str(tp))

  print("Initial Accuracy  is: ", str(acc_test))
  print("Initial F1 score is: ", str(f1))
  print("Initial Confusion Matrix is(tn,fp,fn,tp): ", str(tn), str(fp), str(fn), str(tp))

  acc_res={}
  acc_res[0] = acc_test

  f1_res = {}
  f1_res[0] = f1

  for item in dataset:
    sentence1 = item[0]
    sentence2 = item[-1]
    #Appending the row to dataframe
    #train_df.loc[len(train_df)] =  {'def_first': sentence1, 'def_end': sentence2}

    # Predict the probability of classifying the pair as 0 or 1
    prob_0 = svm_model.predict_proba(extract_features(sentence1, sentence2).reshape(1, -1))[0][0]
    prob_1 = svm_model.predict_proba(extract_features(sentence1, sentence2).reshape(1, -1))[0][1]

    # Extract features based on the predicted probability
    if prob_0 > prob_1:
        feature_vector = extract_features(sentence1, sentence2) # Extract features based on the probability of being 0
        label = 0
    else:
        feature_vector = extract_features(sentence1, sentence2) # Extract features based on the probability of being 1
        label = 1
    X_train = np.vstack((X_train,feature_vector))
    y_train = np.append(y_train, label)
    X_train = np.vstack((X_train, feature_vector))
    y_train = np.append(y_train, 1 - label)

  svm_model.fit(X_train, y_train)
  X_test = vectorizer.transform(test_df['def_first'] + test_df['def_end'])
  y_test_pred = svm_model.predict(X_test.toarray())
  y_test = test_df['label']

  #calculate accuracy, f1 score and confusion matrix
  acc_test = accuracy_score(y_test, y_test_pred)
  logging.info("Final accuracy is: ", str(acc_test))
  acc_res[dataset.index(item)+1] = acc_test
  f1 = f1_score(y_test, y_test_pred)
  f1_res[dataset.index(item)+1] = f1
  logging.info("Final F1 score  is: ", str(f1))
  tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
  logging.info("Final Confusion Matrix is(tn,fp,fn,tp): ", str(tn), str(fp), str(fn), str(tp))

  print("Final accuracy is: ", str(acc_test))
  print("Final F1 score  is: ", str(f1))
  print("Final Confusion Matrix is(tn,fp,fn,tp): ", str(tn), str(fp), str(fn), str(tp))

  #Save the accuracy
  outpath = pj(output_path, '/acc_svm_res.json')
  logging.info(json.dumps(acc_res, indent=4))
  with open(outpath, "w") as outfile:
    json.dump(acc_res, outfile)
  #Save the f1 score
  outpath = pj(output_path, '/f1_svm_res.json')
  logging.info(json.dumps(f1_res, indent=4))
  with open(outpath, "w") as outfile:
    json.dump(f1_res, outfile)
  #Save the confusion matrix
  outpath = pj(output_path, '/confusion_matrix_svm.json')
  with open(outpath, "w") as outfile:
    json.dump(confusion_matrix(y_test, y_test_pred).tolist(), outfile)

  
  #Save the model
  outpath = pj(output_path, '/svm_vectorizer.pkl')
  with open(outpath, 'wb') as f:
    pickle.dump(vectorizer, f)
  outpath = pj(output_path, '/svm_model.pkl')
  with open(outpath, 'wb') as f:
    pickle.dump(svm_model, f)
if __name__ == '__main__':
  input_path = 'data'
  output_path = 'output/svm'
  #train_model('data','data')
  train_model(input_path,output_path)

