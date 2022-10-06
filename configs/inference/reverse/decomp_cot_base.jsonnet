local gen_params = { "gen_model": "gpt3", "engine": "text-davinci-001"};
{
  "start_state": "reverse",
    "end_state": "[EOQ]",
    "models": {
      "reverse": {
        "name": "llmqadecomp",
        "prompt_file": "configs/prompts/reverse/decomp.txt",
        "next_model": "execute",
        "max_tokens": 400,
        "end_state": "[EOQ]",
      } + gen_params,
      "cot": {
        "name": "llmqadecomp",
        "prompt_file": "configs/prompts/reverse/cot.txt",
        "next_model": "extract",
        "max_tokens": 400,
        "end_state": "[EOQ]",
        "max_steps": 1,
      } + gen_params,
      "extract": {
        "name": "answer_extractor",
        "regex": ".* (?:is|reverse) \"(.*)\"\\.",
      },
      "execute": {
        "name": "execute_router"
      },
      "join": {
        "name": "llmqa",
        "prompt_file": "configs/prompts/reverse/join.txt",
      } + gen_params,
      "remove_numbers": {
        "name": "llmqa",
        "prompt_file": "configs/prompts/reverse/rm_num.txt",
      } + gen_params,
    },
    "reader": {
      "name": "drop",
      "add_paras": true
    }
}

