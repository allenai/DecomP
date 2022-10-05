{
  "start_state": "gen",
  "end_state": "[EOQ]",
  "models": {
    "gen": {
      "name": "llmqadecomp",
      "prompt_file": "configs/prompts/commaqa_e/commaqa_e_" + std.extVar("promptid") + "/cot.txt",
      "next_model": "extract",
      "max_tokens": 300,
      "end_state": "[EOQ]",
      "gen_model": "gpt3",
      "max_steps": 1,
      "add_context": true,
    },
    "extract": {
      "name": "answer_extractor",
      "regex": ".* answer is (.*)\\.",
    }
  },
  "reader": {
    "name": "drop",
    "add_paras": true
  }
}