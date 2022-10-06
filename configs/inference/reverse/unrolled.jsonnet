local gen_params = { "gen_model": "gpt3", "engine": "text-davinci-001"};
{
  "start_state": "reverse",
    "end_state": "[EOQ]",
    "models": {
      "reverse": {
        "name": "llmqadecomp",
        "prompt_file": "configs/prompts/reverse/unrolled.txt",
        "next_model": "ans_ext",
        "max_tokens": 500,
        "end_state": "[EOQ]",
      } + gen_params,
      "ans_ext": {
        "name": "answer_extractor",
        "regex": ".*he answer is (.*)\\.",
      },
      "execute": {
        "name": "execute_router"
      },
    },
    "reader": {
      "name": "drop",
      "add_paras": true
    }
}

