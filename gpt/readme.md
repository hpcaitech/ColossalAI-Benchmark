# Step1: Datasets
We do not host any datasets for GPT or BERT training, however, we detail their collection so that our results may be reproduced.


## Option1: Use A Toy Dataset
If you just want to go through the process quickly, you can download the a toy data (size 80MB).
Down load it from [Google Cloud](https://drive.google.com/file/d/1eCY30B9g-I3oPdtQHR8rmIxx64Js_LZh/view?usp=sharing).


## Option2: Prepare the Webtest Data
### Collecting GPT Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [shenggan's](https://github.com/Shenggan/openwebtext)（modified  from [eukaryote31's](https://github.com/eukaryote31/openwebtext)) work to download urls. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in Megatron's [openwebtext](./tools/openwebtext) directory. 

#### Install necessary packages

```
    pip install ftfy langdetect numpy torch pandas nltk sentencepiece boto3 tqdm regex bs4 newspaper3k htmlmin tldextract cached-path
    git clone git@github.com:Shenggan/LSH.git
    cd LSH
    python setup.py install   
```

#### Download Data

1. Download the deduplicated URLs `<raw_urls>` from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ) 

2. Remove blacklisted URLs.

       git clone git@github.com:WANG-CR/Megatron-LM.git
       python Megatron-LM/tools/openwebtext/blacklist_urls.py  <input file raw_urls>  <output file clean_urls.txt>

3. Download the content from the clean urls and Merge the contents into one loose json file with 1 json per newline of the format `{'text': text, 'url': unique_url}`. The output content will be called <one.json>

   ```
   git clone git@github.com:Shenggan/openwebtext.git
   python openwebtext/download.py <input file clean_urls.txt> --n_procs 50
   ```

#### Prepare Data for GPT Training

1. Perform ftfy, English detection and remove documents with less than 128 tokens. This step can be sharded and run on shards.

   ```
   cd Megatron-LM/tools/openwebtext
   python cleanup_dataset.py <input file one.json> <output file clean.json>
   ```

   Additional cleanup (e.g. remove documents less than 512 characters or dataset specific cleaning like stories, realnews datasets) can be done using `cleanup_fix_dataset.py`. More details can be found by running `python cleanup_fix_dataset.py --help`

2. Using LSH, find possible duplicates and store then in a file for later processing. The code supports saving and loading fingerprints for recurrent deduplications, and is also multithreaded for faster processing. More details are can be found by `python find_duplicate.py --help`.

   ```
   python find_duplicates.py --inputs <pairlist list of input cleaned data files and keys, e.g. cc.json cc_id news.json news_id> --output <output possible duplicate urls filename>
   ```

3. Based on similarity measure defind inside function `is_similar` (default: 0.9), group urls that are similar. Basically, for each group, only one url we should keep and remove the rest.

   ```
   python group_duplicate_urls.py <possible duplicate urls file> <output file containing similar urls>
   ```

4. Remove similar documents that were detected in the last step.

   ```
   python remove_group_duplicates.py <file containing simialr documents.json> <cleaned data file.json> <outputfile containing deduplicate data>
   ```

5. shuffle the dataset

   ```
   shuf <input file clean.json> -o <output file train_data.json>
   ```


## Step2: Training

Run GPT training using 4 GPUs with vanilla parallel strategy. You can try other strategies by using different files from ./configs.
The training last for 10 steps.

```python
NUM_GPUS_PER_NODE=4
NUM_NODES=1
NODE_RANK=0

export EXEC="torchrun"
export CONFIG="./configs/gpt2_vanilla.py"

DATA=/your_own_path/small-gpt-dataset.json ${EXEC} --nproc_per_node=${NUM_GPUS_PER_NODE} \
                                 --nnodes=${NUM_NODES} \
                                 --node_rank=${NODE_RANK} \
                                 train.py --from_torch --config=${CONFIG}
```