# DecomP
Repository for Decomposed Prompting


# Setup

1. Install the required dependencies
    ```shell
    conda create -n decomp python=3.8
    pip install -r requirements.txt
    ```
2. Set the necessary env variables
    ```shell
    export PYTHONPATH=src/
    export OPENAI_API_KEY=<YOUR_API_KEY>
    ```

# Running Inference

## Letter Concatenation
Run the letter concatenation experiments with one of the prompts
```shell
    promptid=p1 python -m decomp.inference.configurable_inference \
      --input datasets/letter_cat/n5_eg100_pos2_space.json \
      --config configs/inference/letter_cat/letter_cat_greedy_search.jsonnet \
      --output output/predictions/letter_cat_n5_eg100_pos2_space_decomp_predictions.json
      
    python -m decomp.datasets_utils.drop_eval \
      --gold_path datasets/letter_cat/n5_eg100_pos2_space.json \
      --prediction_path output/predictions/letter_cat_n5_eg100_pos2_space_decomp_predictions.json
```

Change promptid to p2 or p3 to try other prompts. Change config to cot_letter_cat_greedy_search to
evaluate the CoT baseline and cot_rollout_letter_cat_greedy_search to evaluate the CoT w/ rollout
baseline.

## List Reversal
Run the list reversal experiments on sequences of length `LEN` 
using prompting strategy `STRAT` (`STRAT` should be one of `decomp`, `cot`, `unrolled`).
```shell
# Generate the data
mkdir -p datasets/reverse
LEN=10
STRAT=decomp
python src/decomp/datasets_utils/build_reverse_dataset.py \
  --listop=reversed \
  --elts=words \
  --out_dir=datasets/reverse
  --length=$LEN \
  --num_examples=1000

python -m decomp.inference.configurable_inference \
  --input datasets/reverse/test_${LEN}_normal_words.json \
  --config configs/inference/reverse/${STRAT}.jsonnet \
  --output output/predictions/reverse_${LEN}_${STRAT}_predictions.json
  
python -m decomp.datasets_utils.drop_eval \
  --gold_path datasets/reverse/test_${LEN}_normal_words.json \
  --prediction_path output/predictions/reverse_${LEN}_${STRAT}_predictions.json 
```

## CommAQA
Run the CommAQA experiments with one of the prompts
```shell
    promptid=p1 python -m decomp.inference.configurable_inference \
      --input datasets/commaqa_e/iid_test_eg100.json \
      --config configs/inference/commaqa/commaqae_fine_greedy_search.jsonnet \
      --output output/predictions/commaqa_e_iid_test_eg100_fine_decomp_predictions.json
      
    python -m decomp.datasets_utils.drop_eval \
      --gold_path datasets/commaqa_e/iid_test_eg100.json \
      --prediction_path output/predictions/commaqa_e_iid_test_eg100_fine_decomp_predictions.json
```

Change promptid to p2 or p3 to try other prompts. Change config to cot_commaqae_greedy_search to
evaluate the CoT baseline and commaqae_coarse_greedy_search to evaluate the coarse decomposition.
Change the test file to datasets/commaqa_e/compgen_test_eg300.json to evaluate the compositional
generalization split. 
