import logging
import math
import random
import re
from itertools import product, permutations

from decomp.inference.data_instances import QuestionGenerationStep, Task
from decomp.inference.model_search import ParticipantModel
from decomp.inference.utils import get_sequence_representation, stem_filter_tokenization, BLANK, \
    stop_words_set
from decomp.models.generator import LMGenerator
from decomp.models.gpt3generator import GPT3Generator
from decomp.models.llm_client_generator import LLMClientGenerator

logger = logging.getLogger(__name__)


class QuestionGenParticipant(ParticipantModel):

    def __init__(self, scale_by_step=1, add_eos=False, prompt_file="", next_model="execute",
                 end_state="[EOQ]", use_special_number_format=False, add_context=False,
                 max_steps=10, gen_model="lm", **kwargs):
        self.scale_by_step = scale_by_step
        self.add_eos = add_eos
        if prompt_file:
            with open(prompt_file, "r") as input_fp:
                self.prompt = "".join(input_fp.readlines())
        else:
            self.prompt = None
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0
        self.use_special_number_format = use_special_number_format
        self.add_context = add_context
        self.max_steps = max_steps
        self.gen_model = gen_model
        if gen_model == "lm":
            self.generator = LMGenerator(**kwargs)
        elif gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMClientGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)

    def return_model_calls(self):
        return {self.gen_model + "gen": self.num_calls}

    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param data: should have a dictionary as input containing
          mutable data
        :type data: dict
        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        :raises: ValueError
        """
        ## first checks state of `json_input` to figure out how to format things
        ## the first question
        data = state.data
        question_seq = data.get_current_qseq()
        answer_seq = data.get_current_aseq()
        # avoid long chains since this is costly and most likely an error
        if len(question_seq) >= self.max_steps:
            new_state = state.copy()
            output = self.end_state
            new_state.next = self.end_state
            new_state.data.add_qgen(QuestionGenerationStep(
                question=output,
                score=0,
                participant=state.next
            ))
            return new_state

        if self.use_special_number_format:
            gen_seq = "QC: " + data.get_current_inference_data()["query"]
            for qidx, (ques, ans) in enumerate(zip(question_seq, answer_seq)):
                gen_seq += "\nQ{}: {}\n#{}: {}".format(qidx + 1, ques, qidx + 1, ans)
            gen_seq += "\nQ{}:".format(len(question_seq) + 1)
        else:
            gen_seq = get_sequence_representation(origq=data.get_current_inference_data()["query"],
                                                  question_seq=question_seq,
                                                  answer_seq=answer_seq, compq_marker="QC: ",
                                                  simpq_marker="\nQS: ", answer_marker="\nA: ",
                                                  interq_marker="\nQS: ")
        if self.add_context and "paras" in state.data:
            gen_seq = "\n".join(state.data["paras"]) + "\n\n" + gen_seq

        if self.prompt:
            gen_seq = self.prompt + "\n\n" + gen_seq

        ## eventual output
        new_states = []
        ## go through generated questions
        output_seq_scores = self.generator.generate_text_sequence(gen_seq)
        self.num_calls += 1
        observed_outputs = set()
        for (output_seq, score) in output_seq_scores:
            if debug:
                print("--> " + output_seq + " : " + str(score))
            output = output_seq.strip()
            # catch potentially spurious duplicates
            if output in observed_outputs:
                continue
            else:
                observed_outputs.add(output)
            # copy state
            new_state = state.copy()

            # lower is better, same as the scores returned by generate_text_sequence
            assert score >= 0, "Score from generation assumed to be +ve. Got: {}! Needs to be " \
                               "+ve to ensure monotonically increasing scores as expected by the" \
                               " search.".format(score)
            new_state._score += score

            new_state.data.add_qgen(QuestionGenerationStep(
                question=output,
                score=score,
                participant=state.next
            ))

            if output == self.end_state:
                new_state.next = self.end_state
            else:
                new_state.data.add_task(Task(task_question=None,
                                             task_participant=new_state.next))
                new_state.next = self.next_model

            new_states.append(new_state)

        return new_states


class RandomGenParticipant(ParticipantModel):

    def __init__(self, operations_file, model_questions_file, sample_operations, sample_questions,
                 max_steps=6, next_model="execute", topk_questions=True, end_state="[EOQ]"):
        self.operations = self.load_operations(operations_file)
        self.model_questions = self.load_model_questions(model_questions_file)
        self.sample_operations = sample_operations
        self.sample_questions = sample_questions
        self.end_state = end_state
        self.next_model = next_model
        self.max_steps = max_steps
        self.num_calls = 0
        self.topk_questions = topk_questions

    def return_model_calls(self):
        return {"randomgen": self.num_calls}

    def load_operations(self, operations_file):
        with open(operations_file, "r") as input_fp:
            ops = [x.strip() for x in input_fp.readlines()]
        return ops

    def load_model_questions(self, model_questions_file):
        model_question_list = {}
        with open(model_questions_file, "r") as input_fp:
            for line in input_fp:
                fields = line.strip().split("\t")
                model = fields[0]
                if model not in model_question_list:
                    model_question_list[model] = []
                for question in fields[1:]:
                    question_entities = self.find_question_entities(question)
                    for q_ent in question_entities:
                        question = question.replace(q_ent, BLANK)
                    model_question_list[model].append(question)
        # get unique questions
        output_model_questions = []
        for model_key, question_list in model_question_list.items():
            unique_questions = list(set(question_list))
            for q in unique_questions:
                output_model_questions.append((model_key, q))
                print(model_key, q)
            logger.info("{} Questions in {} language".format(len(unique_questions),
                                                             model_key))

        return output_model_questions

    def select(self, population, sample_size_or_prop, samplek=True):
        if sample_size_or_prop >= 1:
            if samplek:
                return random.sample(population, k=sample_size_or_prop)
            else:
                return population[:sample_size_or_prop]
        else:
            if samplek:
                return random.sample(population, k=math.ceil(sample_size_or_prop * len(population)))
            else:
                return population[:math.ceil(sample_size_or_prop * len(population))]

    def build_end_state(self, state):
        new_state = state.copy()
        new_state.next = self.end_state
        output = self.end_state
        new_state.data.add_qgen(QuestionGenerationStep(
            question=output,
            score=0,
            participant=state.next
        ))
        return new_state

    def score_question(self, sub_question, complex_question):
        sub_question_tokens = set(stem_filter_tokenization(sub_question))
        if len(sub_question_tokens) == 0:
            logger.debug("No tokens found in sub_question: {}!!".format(sub_question))
            return 0.0
        complex_question_tokens = set(stem_filter_tokenization(complex_question))
        overlap = sub_question_tokens.intersection(complex_question_tokens)
        # only penalized for sub-question length
        return len(overlap) / len(sub_question_tokens)

    def find_question_entities(self, origq):
        entities = []
        for m in re.finditer("\\b([A-Z]\w+)", origq):
            if m.group(1).lower() not in stop_words_set:
                entities.append(m.group(1))

        for m in re.finditer("([0-9\.]+)", origq):
            entities.append(m.group(1))
        return entities

    def replace_blanks(self, blanked_str, fillers):
        num_blanks = blanked_str.count(BLANK)
        output_strings = []
        if num_blanks > 0:
            filler_permutations = permutations(fillers, num_blanks)
            for permutation in filler_permutations:
                new_str = blanked_str
                for filler_val in permutation:
                    new_str = new_str.replace(BLANK, filler_val, 1)
                output_strings.append(new_str)
        else:
            output_strings = [blanked_str]
        return output_strings

    def query(self, state, debug=False):
        data = state.data
        num_steps = len(data.get_current_qseq())
        # push for one extra step so that all shorter chains have been explored
        if num_steps > self.max_steps:
            return [self.build_end_state(state)]
        origq = data["query"]
        answer_strs = []
        if num_steps == 0:
            # hard-coded to only consider select in the first step
            ops = ["select"]
        else:
            for x in range(num_steps):
                answer_strs.append("#" + str(x + 1))
            operations_pool = []
            for op in self.operations:
                operations_pool.extend(self.replace_blanks(op, answer_strs))
            ops = self.select(operations_pool, self.sample_operations)

        question_entities = self.find_question_entities(origq)
        # hack to only use a filler in one of the steps
        potential_fillers = question_entities + answer_strs
        filler_pool = []
        for filler in potential_fillers:
            found_match = False
            for question in state.data.get_current_subqseq():
                if filler in question:
                    found_match = True
                    break
            if not found_match:
                filler_pool.append(filler)

        questions_pool = [(m, newq) for (m, q) in self.model_questions
                          for newq in self.replace_blanks(q, filler_pool)]
        if self.topk_questions:
            sorted_model_questions = sorted(questions_pool, reverse=True,
                                            key=lambda x: self.score_question(x[1], origq))
            model_questions = self.select(sorted_model_questions, self.sample_questions,
                                          samplek=False)
        else:
            model_questions = self.select(questions_pool, self.sample_questions, samplek=True)
        op_model_qs_prod = product(ops, model_questions)
        ## eventual output
        new_states = []
        self.num_calls += 1
        for (op, model_qs) in op_model_qs_prod:
            (model, question) = model_qs

            # no point repeating the exact same question
            if question in state.data.get_current_subqseq():
                continue
            # copy state

            new_state = state.copy()
            new_state.next = self.next_model
            new_state._score += 1
            output = "({}) [{}] {}".format(op, model, question)
            ## add new question to question_seq
            new_state.data.add_qgen(QuestionGenerationStep(
                question=output,
                score=1,
                participant=state.next
            ))
            # new_state.data["question_seq"].append(output)
            # new_state.data["score_seq"].append(1)
            # new_state.data["command_seq"].append("gen")
            ## mark the last output
            # new_state.last_output = output
            new_states.append(new_state)
        ##
        # if len(data["question_seq"]) > 0:
        #     new_states.append(self.build_end_state(state))
        return new_states
