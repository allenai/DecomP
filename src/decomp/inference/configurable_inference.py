import argparse
import json
import logging
import os

import _jsonnet

from decomp.inference.constants import MODEL_NAME_CLASS, READER_NAME_CLASS
from decomp.inference.dataset_readers import DatasetReader
from decomp.inference.model_search import (
    ModelController,
    BestFirstDecomposer)
from decomp.inference.data_instances import StructuredDataInstance
from decomp.inference.utils import get_environment_variables

logger = logging.getLogger(__name__)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Convert HotPotQA dataset into SQUAD format')
    arg_parser.add_argument('--input', type=str, required=False, help="Input QA file")
    arg_parser.add_argument('--output', type=str, required=False, help="Output file")
    arg_parser.add_argument('--config', type=str, required=True, help="Model configs")
    arg_parser.add_argument('--reader', type=str, required=False, help="Dataset reader",
                            choices=READER_NAME_CLASS.keys())
    arg_parser.add_argument('--debug', action='store_true', default=False,
                            help="Debug output")
    arg_parser.add_argument('--demo', action='store_true', default=False,
                            help="Demo mode")
    arg_parser.add_argument('--threads', default=1, type=int,
                            help="Number of threads (use MP if set to >1)")
    arg_parser.add_argument('--n-examples', type=int)
    return arg_parser.parse_args()


def build_decomposer_and_models(config_map):
    print("loading participant models (might take a while)...")
    model_map = {}
    for key, value in config_map["models"].items():
        class_name = value.pop("name")
        if class_name not in MODEL_NAME_CLASS:
            raise ValueError("No class mapped to model name: {} in MODEL_NAME_CLASS:{}".format(
                class_name, MODEL_NAME_CLASS))
        model = MODEL_NAME_CLASS[class_name](**value)
        if key in config_map:
            raise ValueError("Overriding key: {} with value: {} using instantiated model of type:"
                             " {}".format(key, config_map[key], class_name))
        config_map[key] = model.query
        model_map[key] = model

    ## instantiating
    controller = ModelController(config_map, data_class=StructuredDataInstance)
    decomposer = BestFirstDecomposer(controller)
    return decomposer, model_map


def load_config(config_file):
    if config_file.endswith(".jsonnet"):
        ext_vars = get_environment_variables()
        logger.info("Parsing config with external variables: {}".format(ext_vars))
        config_map = json.loads(_jsonnet.evaluate_file(config_file, ext_vars=ext_vars))
    else:
        with open(config_file, "r") as input_fp:
            config_map = json.load(input_fp)
    return config_map


def load_reader(args, config_map):
    if "reader" in config_map:
        reader_config = config_map["reader"]
        reader_name = reader_config.pop("name")
        reader: DatasetReader = READER_NAME_CLASS[reader_name](**reader_config)
    else:
        reader: DatasetReader = READER_NAME_CLASS[args.example_reader]()
    return reader


def demo_mode(args, reader, decomposer):
    qid_example_map = {}
    if args.input:
        for eg in reader.read_examples(args.input):
            qid = eg["qid"]
            question = eg["query"]
            answer = eg["answer"]
            output_eg = {
                "qid": qid,
                "query": question,
                "question": question,
            }
            if "paras" in eg:
                output_eg["paras"] = eg["paras"]
            qid_example_map[qid] = answer, output_eg
    while True:
        qid = input("QID: ")
        if qid in qid_example_map:
            answer, example = qid_example_map[qid]
            print("using example from input file: " + json.dumps(example, indent=2))
        else:
            question = input("Question: ")
            answer = None
            example = {
                "qid": qid,
                "query": question,
                "question": question,
            }
        final_state, other_states = decomposer.find_answer_decomp(example, debug=args.debug)
        if final_state is None:
            print("FAILED!")
        else:
            if args.debug:
                for other_state in other_states:
                    data = other_state.data
                    print(data.get_printable_reasoning_chain())
                    print("Score: " + str(other_state._score))
            data = final_state._data
            chain = example["question"]
            chain += "\n" + data.get_printable_reasoning_chain()
            chain += " S: " + str(final_state._score)
            chain += "\nG: {}".format(answer)
            print(chain)


def inference_mode(args, reader, decomposer, model_map):
    print("Running decomposer on examples")
    qid_answer_chains = []

    if not args.input:
        raise ValueError("Input file must be specified when run in non-demo mode")
    examples = reader.read_n_examples(args.input, args.n_examples)
    if args.threads > 1:
        import multiprocessing as mp

        mp.set_start_method("spawn")
        with mp.Pool(args.threads) as p:
            qid_answer_chains = p.map(decomposer.return_qid_prediction, examples)
    else:
        for example in examples:
            qid_answer_chains.append(
                decomposer.return_qid_prediction(example, debug=args.debug))

    num_call_metrics = {}
    for pname, participant in model_map.items():
        for model, num_calls in participant.return_model_calls().items():
            print("Number of calls to {}: {}".format(pname + "." + model, num_calls))
            num_call_metrics[pname + "." + model] = num_calls
    metrics_json = {
        "num_calls": num_call_metrics
    }
    metrics_file = os.path.join(os.path.dirname(args.output), "metrics.json")

    with open(metrics_file, "w") as output_fp:
        json.dump(metrics_json, output_fp)

    predictions = {x[0]: x[1] for x in qid_answer_chains}
    with open(args.output, "w") as output_fp:
        json.dump(predictions, output_fp)

    chains = [x[2] for x in qid_answer_chains]
    ext_index = args.output.rfind(".")
    chain_tsv = args.output[:ext_index] + "_chains.tsv"
    with open(chain_tsv, "w") as output_fp:
        for chain in chains:
            output_fp.write(chain + "\n")


if __name__ == "__main__":

    parsed_args = parse_arguments()
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    config_map = load_config(parsed_args.config)

    decomposer, model_map = build_decomposer_and_models(config_map)

    example_reader = load_reader(args=parsed_args, config_map=config_map)

    if parsed_args.demo:
        demo_mode(args=parsed_args, reader=example_reader, decomposer=decomposer)
    else:
        inference_mode(args=parsed_args, reader=example_reader,
                       decomposer=decomposer, model_map=model_map)
