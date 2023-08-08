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

metric_accuracy = load_metric("accuracy")
metric_f1 = load_metric("f1")
metric_pre = load_metric("precision")
metric_re = load_metric("recall")

def compute_metric_search(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_f1.compute(predictions=predictions, references=labels, average='micro')


def compute_metric_all(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'f1_micro': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
        'recall_micro': metric_re.compute(predictions=predictions, references=labels, average='micro')['recall'],
        'precision_micro': metric_pre.compute(predictions=predictions, references=labels, average='micro')['precision'],
        'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
        'recall_macro': metric_re.compute(predictions=predictions, references=labels, average='macro')['recall'],
        'precision_macro': metric_pre.compute(predictions=predictions, references=labels, average='macro')['precision'],
        'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
    }


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
    


model_list = ['roberta-large','xlm-roberta-base'] #,prajjwal1/bert-tiny 

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



    #bootstrapping
    temperature = 1.0
    batch_size = 100
    new_rows = []
    new_train_data = tokenized_data['train']
    old_model = trainer.model
    for i in range(0, len(val_df), batch_size):
        model = old_model
        batch = val_df.iloc[i:i+batch_size]

        inputs_list = [tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512) for input_text in batch['defs']]
        inputs_list = [{k: v.to(device) for k, v in inputs.items()} for inputs in inputs_list]

        logits_list = [trained_model(**inputs).logits for inputs in inputs_list]

        # Apply temperature scaling
        scaled_logits_list = [logits / temperature for logits in logits_list]
        probs_list = [F.softmax(scaled_logits, dim=-1) for scaled_logits in scaled_logits_list]

        for j, probs in enumerate(probs_list):
            probs_np = probs.detach().cpu().numpy()
            p = probs_np[0].tolist()

            updated_features_0 = probs_np[:, p.index(max(p))]
            new_row_0 = batch.iloc[j].copy() # create a copy of the original row
            #new_row_0['updated_prob'] = updated_features_0.tolist()
            new_row_0['label'] = p.index(max(p))
            new_rows.append(new_row_0)

        new_train_df = pd.DataFrame(new_rows, columns=['label','defs'])
        new_dataset_tokenized = preprocess_data(new_train_df) 
        new_dataset = Dataset.from_dict(new_dataset_tokenized)
        new_train_data = concatenate_datasets([new_train_data, new_dataset])

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset= new_train_data,
            data_collator= data_collator,
            compute_metrics=compute_metric_all,
        )

        trainer.train()
        old_model= trainer.model #keeping the prev model
        df = pd.DataFrame(trainer.state.log_history)
        df.to_csv(output_log_training+'/'+model_checkpoints+'_'+str(i)+'_bootstrapped_train_log.csv',index=False)

        model= trainer.model
        # evaluation on bootstrapped model
        trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=output_path_model, evaluation_strategy="no", seed=42),
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        compute_metrics=compute_metric_all)
        metric = trainer.evaluate()
        with open(output_log_eval+'/'+model_checkpoints+'_'+str(i)+'_bootstrapped_eval_log.json','w') as fp:
            json.dump(metric, fp)

        print(metric)
        del model