# LoRA-Unlearn


This study addresses the challenge of machine unlearning in light of growing privacy regulations and the need for adaptable AI systems. We present a novel approach, PruneLoRA: Using LoRA to fine-tune sparse models.

## Abstract

Due to increasing privacy regulations and regulatory compli- ance, Machine Unlearning (MU) has become essential. The goal of unlearning is to remove information related to a spe- cific class from a model. Traditional approaches achieve exact unlearning by retraining the model on the remaining dataset, but incur high computational costs. This has driven the de- velopment of more efficient unlearning techniques, includ- ing model sparsification techniques, which boost computa- tional efficiency, but degrade the model’s performance on the remaining classes. To mitigate these issues, we propose a novel method, PruneLoRA which introduces a new MU paradigm, termed prune first, then adapt, then unlearn. LoRA (Hu et al. 2022)reduces the need for large-scale parameter up- dates by applying low-rank updates to the model. We leverage LoRA to selectively modify a subset of the pruned model’s parameters, thereby reducing the computational cost, mem- ory requirements and improving the model’s ability to retain performance on the remaining classes. Experimental Results across various metrics showcase that our method outperforms other approximate MU methods and bridges the gap between exact and approximate unlearning

## Model Weights
The model weights can be found in the following Google Drive folder:
[**Google Drive - Model Weights**](https://drive.google.com/drive/folders/1t4KTGH4lcaoS6XmFAtGGHJlSUtu3xCxa?usp=sharing)

## Directory Structure:
Please ensure that the model weights are stored inside the model_weights folder in the following structure:

```plaintext
model_weights/
│
├── vitbase10_finetune5.pth
├── vitbase10_finetune10.pth
├── resnet10_finetune5.pth
└── ...
```

Make sure to download the weights and place them in the correct directory structure before using the models.

### Usage

After ensuring the model weights are downloaded and set up as follows for the base ViT and base ResNet models, you can perform various operations on the models:

- **Fine-tuning**: Run `finetune.py` to fine-tune the model to unlearn.
- **Pruning and Fine-tuning**: Run `pruneft.py` to prune the model and then fine-tune it.
- **LoRA Fine-tuning**: Run `lora.py` to fine-tune the model using LoRA (Low-Rank Adaptation).
- **Pruning with LoRA Fine-tuning**: Run `loraprune.py` to prune the model and fine-tune it using LoRA.
  
For **evaluation**, you can run `evaluate.py` to assess the performance of the models.