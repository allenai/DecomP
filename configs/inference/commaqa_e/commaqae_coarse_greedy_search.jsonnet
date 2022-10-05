local base_primitives = import "../primitives.libsonnet";
local gen_params = {"gen_model": "gpt3"};
local primitives = {[p]: base_primitives[p] + gen_params for p in std.objectFields(base_primitives)};
{
  "start_state": "gen",
  "end_state": "[EOQ]",
  "models": {
    "gen": {
      "name": "llmqadecomp",
      "prompt_file": "configs/prompts/commaqa_e/commaqa_e_" + std.extVar("promptid") + "/decomp_coarse.txt",
      "next_model": "execute",
      "end_state": "[EOQ]",
      "add_context": false,
    } + gen_params,
     "qa": {
      "name": "llmqa",
      "prompt_file": "configs/prompts/commaqa_e/commaqa_e_" + std.extVar("promptid") + "/qa.txt",
      "add_context": true,
    } + gen_params,
    "execute": {
      "name": "execute_router"
    },
  } + primitives,
  "reader": {
    "name": "drop",
    "add_paras": true
  }
}