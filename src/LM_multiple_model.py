import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] ='1'
os.environ['WANDB_DISABLED'] = 'True'

import torch
import transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification, AutoModelForMaskedLM,DefaultDataCollator,Trainer,TrainingArguments,DataCollatorWithPadding,DataCollatorForTokenClassification
from datasets import load_dataset, load_from_disk,load_metric,Dataset,concatenate_datasets 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import json
import argparse
from os.path import join as pjoin


# import logging
# logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


parser = argparse.ArgumentParser()
#parser.add_argument('-i','--input_path',help='Input path to the data', required=True)
parser.add_argument('-o','--output_path',help='Path to save results', required=True)
# Parse the argument
args = parser.parse_args()


import numpy as np
from datasets import load_metric

metric_accuracy = load_metric("accuracy")
metric_f1 = load_metric("f1")
metric_pre = load_metric("precision")
metric_re = load_metric("recall")

def compute_metric_all(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    macro = {
    'f1': {},
    'recall': {},
    'precision': {}
    }
    for label in range(logits.shape[1]):
        label_predictions = predictions == label
        label_references = labels == label
        macro['f1'][label] = metric_f1.compute(predictions=label_predictions, references=label_references, average='weighted')['f1']
        macro['recall'][label] = metric_re.compute(predictions=label_predictions, references=label_references, average='weighted')['recall']
        macro['precision'][label] = metric_pre.compute(predictions=label_predictions, references=label_references, average='weighted')['precision']

    results = {
        'macro': macro,
        'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
    }
    return results
  

def preprocess_data(data_to_process):
    # Get all the dialogues
    inputs = [dialogue for dialogue in data_to_process['defs']]
    
    # Tokenize the dialogues
    model_inputs = tokenizer(inputs,  max_length=max_input, padding='longest', truncation=True)
    
    # Set labels
    label_index = [0, 1, 2]
    labels = [label_index.index(label_val) for label_val in data_to_process['label']]
    
    model_inputs['labels'] = torch.tensor(labels)  # convert to tensor
    #print(model_inputs['labels'])
    return model_inputs

class feat_dataset(torch.utils.data.Dataset):
    def __init__(self, texts, features, labels):
        self.texts = texts
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Encode the text
        inputs = tokenizer(self.texts[idx], padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        inputs = {key:val[0] for key,val in inputs.items()}
        
        # Add the updated features
        inputs['updated_features'] = torch.tensor(self.features[idx])
        
        # Return the inputs and labels as a dictionary
        return {'input_ids':inputs['input_ids'], 
                'attention_mask':inputs['attention_mask'],
                'updated_features':inputs['updated_features'],
                'labels':torch.tensor(self.labels[idx])}
    


model_list = ['roberta-large'] #,prajjwal1/bert-tiny 'xlm-roberta-base' 'bert-base-cased','xlm-roberta-base'

for model_name in model_list:       
        #set up all the paths
        # input_path_l = pjoin(args.input_path,'/nn_train_test_data/')
        # input_path_u = pjoin(args.input_path,'/unlabelled/')

    os.makedirs(os.path.join(args.output_path, model_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, model_name,'best_model'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'train_logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'eval_logs'), exist_ok=True)

    output_path_model = os.path.join(args.output_path, model_name) 
    output_path_best_model = os.path.join(args.output_path, model_name,'best_model')
    output_log_training = os.path.join(args.output_path, 'train_logs')
    output_log_eval = os.path.join(args.output_path, 'eval_logs')

    print(output_path_model)
    print(output_path_best_model)
    print(output_log_training)
    print(output_log_eval)
    
    # define the file path and name
    file_path = output_log_training
    file_name = str(model_name)+'_eval_log_boostrapped.json'

    # check if the file exists, if not, create a new empty dict to store the metrics
    if os.path.exists(os.path.join(file_path, file_name)):
        with open(os.path.join(file_path, file_name), 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}

    # Set the maximum length of the input and target sequences
    max_input = 128
    max_target = 1
    batch_size = 16
    print('Initiating Process')

    model_checkpoints = model_name
    path_to_model = os.path.join('/lustre/projects/cardiff',model_name.split('-')[0],model_name)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels=3)
    special_tokens_dict = {'additional_special_tokens': ['<y>','</y>','<t>','</t>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


    dataset = load_dataset("csv", data_files={"train": '/lustre/projects/cardiff/husu_expts/wiki_weakly_supervised_classifier-main/data/nn_train_test_data/train_df.csv',
                                    "test" : '/lustre/projects/cardiff/husu_expts/wiki_weakly_supervised_classifier-main/data/nn_train_test_data/test_df.csv'})

    tokenized_data = dataset.map(preprocess_data, batched = True,remove_columns= ['label','defs'])
    data_collator = DataCollatorWithPadding(tokenizer)
    print('Data Loaded')

    model_name = model_checkpoints

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir= output_path_model,          
        evaluation_strategy = "no",  
        #save_strategy="epoch",
        learning_rate=2e-6,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,    
        num_train_epochs=5,              
        weight_decay=0.01,               
        push_to_hub=False,
        logging_dir= output_log_training,           
        logging_steps=10,   
        save_total_limit=2,           
        #load_best_model_at_end=True,    
        metric_for_best_model="f1_macro",
        gradient_accumulation_steps = 8,   
        do_eval=False,       
        )


    # Define the Trainer object
    trainer = Trainer(
        model=model,                     
        args=training_args,             
        train_dataset=tokenized_data['train'],      # training dataset
        #eval_dataset=tokenized_data['val'], 
        data_collator = data_collator,  
        compute_metrics=compute_metric_all # function to compute metrics
        )

    trainer.train()
    trained_model = trainer.model
    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(output_log_training+'/no_bootstrapped_train_log.csv',index=False)
    model_inst =  trainer.model

    trainer = Trainer(
            model=model_inst,
            args=TrainingArguments(output_dir=output_path_model, evaluation_strategy="no", seed=42),
            train_dataset=tokenized_data['train'],
            eval_dataset=tokenized_data['test'],
            compute_metrics=compute_metric_all
            )
    metric = trainer.evaluate()

    with open(output_log_eval+'/'+model_checkpoints+'_no_bootstrapped_eval_log.json','w') as fp:
        json.dump(metric, fp)



    val_df = pd.read_csv('/lustre/projects/cardiff/husu_expts/wiki_weakly_supervised_classifier-main/data/unlabelled/v2_7karticles_randomsampled.csv')
    #val_df = val_df.drop(['title'],axis = 1)
    val_df['label'] =  0
    val_df['defs'] = val_df['def_first'].str.cat(val_df['def_end'], sep='[SEP]')
    val_df = val_df.drop(['title','timestamp_first','timestamp_end','def_first', 'def_end'],axis=1)
    val_df = val_df.dropna()



    old_model = trainer.model  #Taking the model trained on 3k data
    temperature = 1.0
    batch_size = 100 #number of batches to divide the Dev data (val_df)
    n = 10 # number of sentences to add for each label
    new_train_data = tokenized_data['train'] #Taking the training data to merge with the 
    iteration = 1
    batch_size = 100
    n=20
    tf=val_df
    val_batches= []
    print('===============================================================================')
    print('Initial Data Shape:')
    print(new_train_data)
    #Continue the iteration until the val df is empty
    for i in range(0, len(val_df), batch_size):
        model = old_model
        val_batches.append(val_df.iloc[i:i+batch_size]) 
        
    for idx, val_batch in enumerate(val_batches):
        while len(val_batch) > 0:
            print("Iteration:", iteration)
            print('===============================================================================')
            new_rows = []
            model = old_model
            inputs_list = [tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512) for input_text in val_batch['defs']]
            inputs_list = [{k: v.to(device) for k, v in inputs.items()} for inputs in inputs_list]
            logits_list = [trained_model(**inputs).logits for inputs in inputs_list]

            # Apply temperature scaling
            scaled_logits_list = [logits / temperature for logits in logits_list]
            probs_list = [F.softmax(scaled_logits, dim=-1) for scaled_logits in scaled_logits_list]
            p_0 = []
            p_1 = []
            p_2 = []
            for j, probs in enumerate(probs_list):
                probs_np = probs.detach().cpu().numpy()
                p = probs_np[0].tolist()
                p_0.append(p[0])
                p_1.append(p[1])
                p_2.append(p[2])
                
            val_batch['prob_0'] = p_0
            val_batch['prob_1'] = p_1
            val_batch['prob_2'] = p_2

            n=10

            val_batch_0 =  val_batch.sort_values(by='prob_0').drop(['prob_1','prob_2'],axis=1)
            val_batch_0['label'] = 0
            val_batch_0 = val_batch_0.rename(columns={'prob_0':'prob'})

            val_batch_1 =  val_batch.sort_values(by='prob_1').drop(['prob_0','prob_2'],axis=1)
            val_batch_1['label'] = 1
            val_batch_1 = val_batch_1.rename(columns={'prob_1':'prob'})

            val_batch_2 =  val_batch.sort_values(by='prob_2').drop(['prob_0','prob_1'],axis=1)
            val_batch_2['label'] = 2
            val_batch_2 = val_batch_2.rename(columns={'prob_2':'prob'})
            
            df0 = val_batch_0.reset_index(drop=True)
            df1 = val_batch_1.reset_index(drop=True)
            df2 = val_batch_2.reset_index(drop=True)

            df = pd.concat([df0[:20],df1[:20],df2[:20]],ignore_index=True)
            val_batch_final = df.drop_duplicates(subset=['label', 'defs']).groupby('label').head(10)

            if not val_batch_final.empty:
                new_dataset_tokenized = preprocess_data(val_batch_final) 
                new_dataset = Dataset.from_dict(new_dataset_tokenized)
                new_train_data = concatenate_datasets([new_train_data, new_dataset])
            else:
                print("No valid sentences for tokenization in this iteration ")

            print('Updated Training data shape')
            print(new_train_data.shape)
            print('===============================================================================')


            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset= new_train_data,
                data_collator= data_collator,
                compute_metrics=compute_metric_all,
                )

            trainer.train()
            df = pd.DataFrame(trainer.state.log_history)
            df.to_csv(output_log_eval+model_checkpoints+'_iter_'+str(iteration)+'_bootstrapped_train_log.csv',index=False)
            
            model= trainer.model

            # evaluation on bootstrapped model
            trainer = Trainer(
                model=model,
                args=TrainingArguments(output_dir= output_path_model, evaluation_strategy="no", seed=42),
                train_dataset=tokenized_data['train'],
                eval_dataset=tokenized_data['test'],
                compute_metrics=compute_metric_all
                )
            metric = trainer.evaluate()
            
            metrics[str(iteration)] = metric
            with open(output_log_eval+model_checkpoints+'_iter_'+str(iteration)+'_bootstrapped_eval_log.json','w') as fp:
                json.dump(metric, fp)
            print('Bootstrapped model Performance on Test Data: \n')
            print(metric)
            print('===================================================================================')

            # merge val_df and val_batch_final
            #merged_val_df = val_df.merge(val_batch_final, on='defs', how='left', indicator=True)
            merged_batch_df = val_batch.merge(val_batch_final, on='defs', how='left', indicator=True)
            # filter rows only in val_df
            #val_df = merged_val_df.loc[merged_val_df['_merge'] == 'left_only', val_df.columns.difference(['label'])]
            val_batch = merged_batch_df.loc[merged_batch_df['_merge'] == 'left_only', val_batch.columns.difference(['label'])]


            iteration+=1
            print("Remaining sentences in val set no.:",idx,' is: ', len(val_batch))
            print('===============================================================================')


            del model