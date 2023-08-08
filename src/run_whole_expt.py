import data_prep,svm_model
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
from sklearn.metrics import accuracy_score

if __name__== "__main__":
    datapath = 'data'
    df, train_df,val_df,test_df =  data_prep.create_splits(datapath)
    unlabelled_df = pd.read_csv(datapath+'/unlabelled/10k_test_data.csv')
    unlabelled_df = unlabelled_df.rename(columns = {'timestamp_1': 'timestamp_first','definition_1':'def_first','timestamp_2':'timestamp_end','definition_2': 'def_end'})
    unlabelled_df = data_prep.append_special_characters(unlabelled_df)
    unlabelled_df['label'] = '[MASK]'

    #Training SVM model: TODO: need to upgrade the code for further application of algorithm

    svm_model= svm_model.svm(train_df)
    X_val = vectorizer.transform(test_df['def_first'] + test_df['def_end'])
    y_val_pred = svm_model.predict(X_val)
    y_test = test_df['label']
    acc_test = accuracy_score(y_test, y_val_pred)
    print("Accuracy:", acc_test)