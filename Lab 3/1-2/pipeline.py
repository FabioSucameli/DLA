import torch
import time
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import evaluate

#  Inizializza la sentiment analysis pipeline.
class SentimentAnalysisPipeline:
    def __init__(self, model_id='distilbert/distilbert-base-uncased', 
                 dataset_id='cornell-movie-review-data/rotten_tomatoes'):
        
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    # Eservizio 1.1: Load dataset splits
    def load_data(self):
        print("Loading dataset...")
        self.ds_train = load_dataset(self.dataset_id, split='train')
        self.ds_val = load_dataset(self.dataset_id, split='validation')
        self.ds_test = load_dataset(self.dataset_id, split='test')
        print(f"Train: {len(self.ds_train)}, Val: {len(self.ds_val)}, Test: {len(self.ds_test)}")
        
    #Esercizio 1.2 & 1.3: Crea feature extractor baseline
    def create_baseline(self):
        print("\nCreating baseline model...")
        
        # Carica base model
        model = AutoModel.from_pretrained(self.model_id)
        
        # Crea la pipeline per l'estrazione delle feature
        extractor = pipeline('feature-extraction', model=model, tokenizer=self.tokenizer)
        
        # Funzione per estrarre le feature
        def extract_features(dataset):
            features = extractor(dataset['text'], return_tensors='pt')
            # Estrai il token CLS dall'ultimo layer
            return torch.vstack([feature[0][0] for feature in features])
        
        print("Extracting features...")
        # Estrazione delle feature per tutti gli split
        X_train = extract_features(self.ds_train)
        y_train = self.ds_train['label']
        X_val = extract_features(self.ds_val)
        y_val = self.ds_val['label']
        X_test = extract_features(self.ds_test)
        y_test = self.ds_test['label']
        
        # Addestrare il classificatore SVM
        print("Training SVM classifier...")
        svc = LinearSVC()
        svc.fit(X_train, y_train)
        
        # Valutare il modello
        print("\nBaseline Results - Validation Set:")
        print(classification_report(y_val, svc.predict(X_val)))
        print("\nBaseline Results - Test Set:")
        print(classification_report(y_test, svc.predict(X_test)))
        
        return svc
    
    #Esercizi 2.1 e 2.2: Prepara i dati e il modello per il fine-tuning
    def prepare_for_finetuning(self):
        print("\nPreparing for fine-tuning...")
        
        # Funzione di tokenizzazione
        def preprocess_function(examples):
            return self.tokenizer(examples['text'], padding=True, truncation=True)
        
        # Tokenizza i dataset
        self.tokenized_train = self.ds_train.map(preprocess_function, batched=True)
        self.tokenized_val = self.ds_val.map(preprocess_function, batched=True)
        self.tokenized_test = self.ds_test.map(preprocess_function, batched=True)
        
        # Carica il modello per la classificazione di sequenze
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, 
            num_labels=2
        )
        
        # Crea il data collator per il padding automatico
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        print("Model and data prepared for fine-tuning")
    
    # Calcola le metriche di valutazione"
    def compute_metrics(self, eval_pred):
        load_accuracy = evaluate.load("accuracy")
        load_f1 = evaluate.load("f1")
        
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)['accuracy']
        f1 = load_f1.compute(predictions=predictions, references=labels)['f1']
        
        return {"accuracy": accuracy, "f1": f1}
    
    # Esercizio 2.3: Fine-tuning del modello
    def finetune_model(self, output_dir='./output', num_epochs=4, batch_size=64, use_gpu=True, use_wandb=False):
        # Verifica CUDA 
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"\nFine-tuning model for {num_epochs} epochs on {device}...")
        
        self.model = self.model.to(device)
        
        run_id = f"distilbert-ep{num_epochs}-bs{batch_size}-{int(time.time())}"
        
        # Argomenti per l'addestramento
        training_args = TrainingArguments(
            output_dir=f"./{run_id}",     
            run_name=run_id,                    
            learning_rate=2e-5,
            lr_scheduler_type="linear",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy='epoch',
            eval_strategy='epoch', 
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            push_to_hub=False,
            report_to='wandb' if use_wandb else 'none',  
            fp16=torch.cuda.is_available() and use_gpu,  
            no_cuda=not use_gpu 
        )
        
        # Crea il trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_val,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Avvia l'addestramento
        trainer.train()
        
        # Valutazione sul validation set
        print("\nFine-tuned Results - Validation Set:")
        val_results = trainer.evaluate()
        print(f"Accuracy: {val_results['eval_accuracy']:.4f}, F1: {val_results['eval_f1']:.4f}")
        
        # Valutazione sul test set
        print("\nFine-tuned Results - Test Set:")
        test_results = trainer.evaluate(self.tokenized_test)
        print(f"Accuracy: {test_results['eval_accuracy']:.4f}, F1: {test_results['eval_f1']:.4f}")
        
        return trainer
    
    def run_full_pipeline(self, use_wandb=False):
        # Esegui l'intera pipeline: baseline + fine-tuning
        # Carica i dati
        self.load_data()
        
        # Crea e valuta il modello baseline
        print("\n" + "="*50)
        print("BASELINE MODEL (Feature Extraction + SVM)")
        print("="*50)
        baseline_model = self.create_baseline()
        
        # Prepara per il fine-tuning
        print("\n" + "="*50)
        print("FINE-TUNING MODEL")
        print("="*50)
        self.prepare_for_finetuning()
        
        # Esegui il fine-tuning
        trainer = self.finetune_model(use_wandb=use_wandb)
        
        return baseline_model, trainer