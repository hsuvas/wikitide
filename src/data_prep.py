import pandas as pd
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# Append dates and special characters into the definitions
def append_special_characters(df):
  df.def_first =  '<y>'+ ' '+ df.timestamp_first+ ' '+ '</y>' +' '+ df.def_first
  df.def_end =  '<y>'+ ' '+ df.timestamp_end+ ' '+ '</y>' +' '+ df.def_end
  def_first= []
  def_end = []
  for (title,def1,def2) in zip(df.title.tolist(),df.def_first.tolist(),df.def_end.tolist()):
    def_first.append(def1.replace(title,'<t>'+' '+ title + ' ' +'</t>'))
    def_end.append(def2.replace(title,'<t>'+' '+ title + ' '+'</t>'))
  df['def_first'] = def_first
  df['def_end'] = def_end
  # We drop the timestamps now?
  df = df.drop(['timestamp_first','timestamp_end'],axis = 1)
  return df

def data_prepare(datapath):
  # Read the csv files inside the folder and put in one dataframe
  files = glob.glob(datapath+"/*.csv")
  df = []
  for f in files:
      csv = pd.read_csv(f)
      if 'def_last' in csv.columns:
        csv = csv.rename(columns = {'def_last' : 'def_end'})
      df.append(csv)
  df = pd.concat(df)

  # Delete unncessary rows and convert the float labels to int
  if 'comment' in df.columns:
    df = df.drop(['comment'],axis =1)
  df = df.dropna()
  df.label = df.label.apply(lambda x: int(x))

  df = append_special_characters(df)
  return df

def split_dataframe_by_column(df, column_name, test_size=0.2, val_size=0.2, random_state=None):
    # Get the unique values of the column
    unique_values = df[column_name].unique()
    
    # Split the unique values into train and test sets
    train_val_values, test_values = train_test_split(unique_values, test_size=test_size, random_state=random_state)
    
    # Split the train set into train and validation sets
    train_values, val_values = train_test_split(train_val_values, test_size=val_size, random_state=random_state)
    
    # Create the train, validation, and test dataframes
    train_df = df[df[column_name].isin(train_values)]
    val_df = df[df[column_name].isin(val_values)]
    test_df = df[df[column_name].isin(test_values)]
    
    return train_df, val_df, test_df

def create_splits(datapath):

    input_path = datapath+ '/labelled'
    df = data_prepare(input_path)
    train_df,val_df,test_df = split_dataframe_by_column(df, 'title', test_size=0.2, val_size=0.2, random_state=None)

    outpath = datapath+'/train_test_data'
    Path(outpath).mkdir(parents=True, exist_ok=True)

    train_df.to_csv(outpath+'/train_df.csv',index=False)
    val_df.to_csv(outpath+'/val_df.csv',index=False)
    test_df.to_csv(outpath+'/test_df.csv',index=False)

    return df, train_df,val_df,test_df
