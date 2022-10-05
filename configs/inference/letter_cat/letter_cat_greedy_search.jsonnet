local base_primitives = import "../primitives.libsonnet";
local gen_params = {"gen_model": "gpt3"};
local primitives = {[p]: base_primitives[p] + gen_params for p in std.objectFields(base_primitives)};
{
  "start_state": "gen",
  "end_state": "[EOQ]",
  "models": {
    "gen": {
      "name": "llmqadecomp",
      "prompt_file": "configs/prompts/letter_cat/letter_cat_" + std.extVar("promptid") + "/letter_cat_n3_ex3_decomp.txt",
      "next_model": "execute",
      "end_state": "[EOQ]",
    } + gen_params,
    "execute": {
      "name": "execute_router"
    },
  } + primitives,
  "reader": {
    "name": "drop"
  }
}