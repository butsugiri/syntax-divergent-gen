# [WIP] syntax-divergent-gen
This is re-implementation of following papers:

- Shu, Raphael, and Hideki Nakayama. "Discrete Structural Planning for Generating Diverse Translations." (2018). [PDF](https://openreview.net/pdf?id=SJG1wjRqFQ)

**Note: still work in progress!!**

## Requirements
- Python 3.6 (or newer)
- logzero
- PyTorch 1.0.1
- sentencepiece
- stanfordnlp

## How-to-Use
### Get Coarse POS Tags
- Download `stanfordnlp` model file for English (EWT) from [here](https://stanfordnlp.github.io/stanfordnlp/installation_download.html)
- Convert your text to sequence of pos tags as follows: `python convert_to_pos.py -i <your_corpus> -m <path_to_model_file>`
- Get coarse pos tags as follows: `python coarse_pos.py --input work/train.en.pos  > work/train.en.pos.coarse`

### Build Vocabulary
- Build source language vocabulary:
  - `spm_train --input <train_corpus> --model_prefix iwslt2016_de --vocab_size=20000  --model_type=bpe  --hard_vocab_limit=false`
- Build pos vocabulary:
  - `spm_train --input work/train.en.pos.coarse --model_prefix iwslt2016 --vocab_size=10000  --model_type=word  --hard_vocab_limit=false`

### Code Learning

```train.sh
CUDA_VISIBLE_DEVICES=1 python -u train_code.py \
 --train_code <your_coarse_pos_tags> \
 --train_source <your_source_lang> \
 --valid_code <your_coarse_pos_tags> \
 --valid_source <your_source_lang> \
 --spm_code_model work/iwslt2016_pos.model \
 --spm_source_model work/iwslt2016_de.model \
 --epoch 20 \
 --gpu 1 \
 --batch_size 256
```

### Train NMT Model with Code
TBA...