# Image Generation and Modality Weight Prior-Aware Driven Multi-Modal Entity Alignment
## Requirements
```txt
torch==2.0.0+cu118
torchvision==0.15.1+cu118
torchaudio==2.0.1+cu118
numpy==1.26.3
pandas==2.2.3
scikit-learn==1.6.0
transformers==4.47.1
tokenizers==0.21.0
safetensors==0.5.0
tensorboard==2.18.0
```
## Data

Data available here: https://drive.google.com/file/d/11v7b_fljlYC3arwul1JvKu0bVtMZfadn/view?usp=drive_link

## How to Run

```txt
bash run_IGMP.sh 0 OEA_D_W_15K_V1 norm 0.2 0 0.05
bash run_IGMP.sh 0 DBP15K zh_en 0.3 0 0.05
```
