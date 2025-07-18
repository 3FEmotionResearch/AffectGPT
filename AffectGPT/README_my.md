# Env Setup
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 tokenizers==0.21.0 vllm==0.6.1
pip install flash-attn==2.7.2.post1 --no-build-isolation

conda env update --file environment.yml
```