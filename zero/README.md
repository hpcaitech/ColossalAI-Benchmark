# GPT2 ZeRO Benchmark
GPT2 ZeRO benchmark with data parallelism to evaluate Colossal-AI, DeepSpeed, FairScale and PatrickStar.

## Setup
1. Install dependencies

```
pip install -r requirements.txt
```

2. Download PatrickStar from github

```shell
git clone https://github.com/Tencent/PatrickStar.git
```

3. Install PatrickStar

```
cd PatrickStar
pip install .
```

## Usage
1. Prepare datasets and tokenizers from HuggingFace Hub if necessary (e.g. we provide an example of training `wikitext-2`).

2. Run benchmark with one of the systems to evaluate
```
DATA=/PATH/TO/DATASET TOKENIZER=/PATH/TO/TOKENIZER LOG=/PATH/TO/LOG torchrun --nproc_per_node=NUM_GPUS run.py --config=CONFIG_FILE
```