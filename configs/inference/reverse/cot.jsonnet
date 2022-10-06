local gen_params = { "gen_model": "gpt3", "engine": "text-davinci-001"};
{
  "start_state": "reverse",
    "end_state": "[EOQ]",
    "models": {
      "reverse": {
        "name": "llmqadecomp",
        "prompt_file": "configs/prompts/reverse/cot.txt",
        "next_model": "ans_ext",
        "max_tokens": 300,
        "end_state": "[EOQ]",
        "max_steps": 1,
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

