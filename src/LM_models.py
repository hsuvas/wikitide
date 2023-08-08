
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
import numpy as np
from datasets import load_metric
from sklearn.metrics import accuracy_score, f1_score
import multiprocessing
import json
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os
os.environ['WANDB_DISABLED'] = 'True'

import torch
import torch.nn.functional as F

import ray
from ray import tune
ray.init(ignore_reinit_error=True, num_cpus=1)

import logging
from os.path import join as pj
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

max_input = 128
max_target = 1
batch_size = 8
print('Initiating Process')
model_checkpoints = 'roberta-base' #'/lustre/projects/cardiff/roberta/roberta-base' #'roberta-base' #'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoints)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoints)
special_tokens_dict = {'additional_special_tokens': ['<y>','</y>','<t>','</t>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


def preprocess_data(data_to_process):
    #get all the dialogues
    inputs = [dialogue for dialogue in data_to_process['defs']]
    #print('inputs:', inputs)
    #tokenize the dialogues
    model_inputs = tokenizer(inputs,  max_length=max_input, padding='longest', truncation=True)
    #print(model_inputs.shape)
    label = torch.tensor(data_to_process['label'])
    print(label.shape)
    #set labels
    model_inputs['labels'] = label 
    return model_inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    accuracy_metric.add_batch(predictions=predictions, references=labels)
    f1_metric.add_batch(predictions=predictions, references=labels)
    return {'accuracy': accuracy, 'f1': f1}


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

def train_model(train_path, val_path,test_path,unl_path, output_dir, output_model_dir,local_dir,final_model_dir, log_dir,parallel=False):

    dataset = load_dataset("csv", data_files={"train": train_path,
                                            "val": val_path,
                                            "test": test_path})
    tokenized_data = dataset.map(preprocess_data, batched = True,remove_columns= ['label','defs'])
    data_collator = DataCollatorWithPadding(tokenizer)



    # # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,       
        evaluation_strategy = "epoch",  
        save_strategy="epoch",
        # learning_rate=5e-5,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,    
        # num_train_epochs=20,              
        weight_decay=0.01,               
        push_to_hub=False,
        logging_dir= log_dir,            
        logging_steps=500,              
        load_best_model_at_end=True,    
        metric_for_best_model="accuracy",
        greater_is_better=True           
    )


    trainer = Trainer(
        model=model_checkpoints,                     
        args=training_args,              
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['val'], 
        data_collator = data_collator,       
        compute_metrics=compute_metrics,
        model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoints, num_labels=2, return_dict=True))
    if parallel:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(5,30))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=local_dir,
                direction="maximize",
                backend="ray",
                n_trials=10,
                resources_per_trial={'cpu': multiprocessing.cpu_count(), "gpu": torch.cuda.device_count()})
            
    else:
        best_run = trainer.hyperparameter_search(
                    hp_space=lambda x: {
                        "learning_rate": tune.loguniform(1e-6, 1e-4),
                        "num_train_epochs": tune.choice(list(range(5,30))),
                        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                    },
                    local_dir=local_dir,
                    direction="maximize",
                    backend="ray",
                    n_trials=10)

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
    trainer.train()
    trainer.save_model(output_model_dir)
    trained_model = trainer.model

    df = pd.read_csv(unl_path)

    df['defs'] = df['def_first'].str.cat(df['def_end'], sep='[SEP]')
    df = df.drop(['title','timestamp_first','timestamp_end','def_first', 'def_end'],axis=1)
    df = df.dropna()

    df['label'] =  0
    df['features'] = 0

    temperature = 1.0

    for i, input_text in enumerate(df['defs']):
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Pass the inputs through the model to get the logits
        logits = trained_model(**inputs).logits

        # Apply temperature scaling
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        probs_np = probs.detach().cpu().numpy()
        if probs_np[0][0]< probs_np[0][-1]:
                updated_features = probs_np[:, 1] 
                df.at[i, 'updated_features'] = updated_features.tolist()
                df.at[i,'label'] = 1
        else:
                updated_features = probs_np[:, 0] 
                df.at[i, 'updated_features'] = updated_features.tolist()
                df.at[i,'label'] = 0


    updated_features = np.array(df['updated_features'].tolist())
    updated_features = updated_features.reshape(updated_features.shape[0], -1)

    #model= trainer.model

    trainer = Trainer(
        model = trained_model,
        args=training_args,
        train_dataset= feat_dataset(df['defs'].tolist(), updated_features, df['label'].tolist()),
        eval_dataset = tokenized_data['test'],
        data_collator= data_collator,
        compute_metrics=compute_metrics 
        )

    trainer.train()
    trainer.save_model(final_model_dir)
    model_upd = trainer.model
    model = AutoModelForSequenceClassification.from_pretrained(model_upd, num_labels=2)
    trainer = Trainer(
        model=model_upd,
        args=TrainingArguments(output_dir=output_dir, evaluation_strategy="no", seed=42),
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        compute_metrics=compute_metrics)

    metric_file = pj(output_dir, "/eval.json")
    metric = trainer.evaluate()
    logging.info(json.dumps(metric, indent=4))
    with open(metric_file, 'w') as f:
        json.dump(metric, f)




if __name__ == '__main__':
    train_path = 'data/train_test_data/train_df.csv'
    val_path = 'data/train_test_data/val_df.csv'
    test_path = 'data/train_test_data/test_df.csv'
    unl_path=  'data/unlabelled/10k_test_data.csv'
    output_dir ='output'
    output_model_dir = 'output/best_ft_model'
    local_dir ='output/ray_models'
    final_model_dir = '/lustre/projects/cardiff/husu_expts/wiki_classifier/final_model'
    log_dir = '/lustre/projects/cardiff/husu_expts/wiki_classifier/best_ft_model/log'
    #parallel=True
    train_model(train_path, val_path,test_path,unl_path, output_dir,output_model_dir, local_dir, final_model_dir,log_dir, parallel=False)
