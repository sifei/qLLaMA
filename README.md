# qLLaMA

This repository contains the qLLaMA model, which aims to fine-tune the LLaMA model for the downstream task (QQP). More information will be provided once the paper is accepted by PLOS ONE.

Finetune commond:
```python finetune.py --base_model='baffo32/decapoda-research-llama-7b-hf' --data_path='fine-tune-dataset/quora_100k_finetune.json' --output_dir='./qLLaMA' --batch_size=128 --num_epochs 10```

Running prediction on GLUE dataset:
```python prediction_glue.py --load_8bit --base_model 'baffo32/decapoda-research-llama-7b-hf' --lora_weights './qLLaMA'```


