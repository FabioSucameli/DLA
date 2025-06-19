from pipeline import SentimentAnalysisPipeline

def main():
    # Verifica CUDA
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Inizializza pipeline
    pipeline = SentimentAnalysisPipeline()
    
    while True:
        # Menu interattivo
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS PIPELINE - MENU")
        print("="*50)
        print("1. Full pipeline (all exercises)")
        print("2. Baseline only (exercises 1.1–1.3)")
        print("3. Fine-tuning only (exercises 2.1–2.3)")
        print("4. Exit")
        
        scelta = input("\nEnter a number (1–4): ")
        
        if scelta == '1':
            # Esegui tutto
            print("\nRunning full pipeline...")
            use_wandb = input("Enable Wandb? (y/n): ").lower() == 'y'
            baseline, trainer = pipeline.run_full_pipeline(use_wandb=use_wandb)
            
        elif scelta == '2':
            # Solo baseline
            print("\nRunning baseline only...")
            pipeline.load_data()
            baseline_model = pipeline.create_baseline()
            
        elif scelta == '3':
            # Solo fine-tuning
            print("\nRunning fine-tuning only...")
            pipeline.load_data()
            pipeline.prepare_for_finetuning()
            
            # Parametri personalizzabili
            print("\nFine-tuning parameters:")
            epochs = int(input("Number of epochs (default 4): ") or "4")
            batch_size = int(input("Batch size (default 64): ") or "64")
            use_wandb = input("Enable Wandb? (y/n): ").lower() == 'y'
            
            trainer = pipeline.finetune_model(
                num_epochs=epochs,
                batch_size=batch_size,
                use_wandb=use_wandb
            )
            
        elif scelta == '4':
            print("\nExiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 5.")


if __name__ == "__main__":
    main()
