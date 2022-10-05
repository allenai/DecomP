local gen_params = {"gen_model": "gpt3"};
{
  "start_state": "gen",
  "end_state": "[EOQ]",
  "models": {
    "gen": {
      "name": "llmqadecomp",
      "prompt_file": "configs/prompts/letter_cat/letter_cat_" + std.extVar("promptid") + "/cot_rollout_letter_cat_n3_ex3_decomp.txt",
      "next_model": "extract",
      "max_tokens": 400,
      "end_state": "[EOQ]",
      "max_steps": 1
    } + gen_params,
    "extract": {
      "name": "answer_extractor",
      "regex": ".* outputs (.*)\\.",
    }
  },
  "reader": {
    "name": "drop"
  }
}