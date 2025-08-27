# Env Setup
```
# Create a conda env, activate it.
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 tokenizers==0.21.0 vllm==0.6.1
pip install flash-attn==2.7.2.post1 --no-build-isolation

conda env update --file conda_env_my.yaml
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"

huggingface-cli login
git config --global credential.helper store
```

# Put dataset and AffectGPT checkpoint (aka. ckpt) into this repo with the exact names.
```
dataset/mer2025-dataset

$ ls -a dataset/mer2025-dataset
.              subtitle_chieng.csv              track3_train_ovmerd.csv
..             track2_train_mercaptionplus.csv  track_all_candidates.csv
audio          track2_train_ovmerd.csv          video
openface_face  track3_train_mercaptionplus.csv
```
```
# This magical "emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz" folder name is actually the `cfg-path` flag in `inference_hybird.py`.
AffectGPT/output/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz
```

# Run inference on dataset. More can check `README.md`.
```
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml --options "inference.test_epoch=60" --ask_reasoning
```
```
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml --options "inference.test_epoch=60" --ask_reasoning --block_description "Image->Last"
```
```
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml --options "inference.test_epoch=60" --ask_reasoning --block_description "Subtitle->Last"
```

# Run evaluation on inference result. Check `AffectGPT/evaluation-scoreonly-my.py`.

# Utils
## Read .npz
Use `UtilScripts/read_inference_results.py`.
