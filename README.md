# qLLaMA

# Playground_LLM

Alpaca_LoRA:https://github.com/tloen/alpaca-lora
```python generate.py --load_8bit --base_model 'decapoda-research/llama-7b-hf' --lora_weights 'tloen/alpaca-lora-7b'```

SSL Tunnel from remote server to local
```ssh -L 7860:localhost:7860 username@server```

Download ggml for embedding
see download_ggml.py

Generate sentence embedding
see create_embedding.py


Running Alpaca-lora:
```python generate.py --load_8bit --base_model 'decapoda-research/llama-7b-hf' --lora_weights 'tloen/alpaca-lora-7b'```

pref version
```
pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
```

Finetune alpac-lora:
```python finetune.py --base_model='baffo32/decapoda-research-llama-7b-hf' --data_path='fine-tune-dataset/quora_100k_finetune.json' --output_dir='./quora-lora-alpaca' --batch_size=128 --num_epochs 10```

Running finetune alpaca-lora:
```python generate.py --load_8bit --base_model 'baffo32/decapoda-research-llama-7b-hf' --lora_weights './quora-lora-alpaca/'```

Running prediction quora-lora:
```python prediction.py --load_8bit --base_model 'baffo32/decapoda-research-llama-7b-hf' --lora_weights './quora-lora-7/'```

Command on HPC:
```srun -p gpuq --mem=200G --gres=gpu:a100:4 --pty bash```

adapter_model.bin not being updated issue:
```
pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
```
```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install accelerate==0.19.0 
pip install bitsandbytes==0.37.2
```

```
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=g-10-11 example_5shot.py --ckpt_dir /mnt/isilon/tsui_lab/LLM/LLAMA_MODEL/65B/65B/ --tokenizer_path /mnt/isilon/tsui_lab/LLM/LLAMA_MODEL/65B/tokenizer.model --filename /home/hans2/Quora/Quora_test_10.csv --ofilename Quora_test_10_5shot_65B.csv
```
