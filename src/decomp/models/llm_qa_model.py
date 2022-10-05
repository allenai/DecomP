import json
import re
from decomp.models.gpt3generator import GPT3Generator
from decomp.models.llm_client_generator import LLMClientGenerator


class LLMQAModel:
    def __init__(self, prompt_file="", regex_extract=None, add_context=True, gen_model="gpt3",
                 **kwargs):
        if prompt_file:
            with open(prompt_file, "r") as input_fp:
                self.prompt = "".join(input_fp.readlines())
        else:
            self.prompt = None
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMClientGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)

        self.num_calls = 0
        self.regex_extract = regex_extract
        self.add_context = add_context

    def ask_question(self, input_question, context):
        question_prompt = self.prompt
        if context and self.add_context:
            # TODO Hack!! Needs a better fix
            m = re.match(" *PARA_([0-9]+) (.*)", input_question)
            if m:
                assert isinstance(context, list)
                context = context[int(m.group(1))]
                input_question = m.group(2)
            elif isinstance(context, list):
                context = "\n".join(context)
            if context:
                question_prompt += "\n\n" + context

        question_prompt += "\n\nQ: " + input_question + "\nA:"
        # print("<QA>: ... %s" % question_prompt[-500:])
        output_text_scores = self.generator.generate_text_sequence(question_prompt)
        self.num_calls += 1
        if len(output_text_scores) > 1:
            print("Can not handle more than one answer for QA model yet" +
                             "\n" + str(output_text_scores))
            output_text_scores = [output_text_scores[0]]
        # print(input_question)
        # print(output_text_scores)
        # only answer string
        answer_str = output_text_scores[0][0].strip()
        if self.regex_extract:
            m = re.match(self.regex_extract, answer_str)
            if m:
                answer_str = m.group(1).strip()
            else:
                # No match
                print("Did not find a match for input regex: {} in {}".format(self.regex_extract,
                                                                              answer_str))
                return "", []
        try:
            json_answer = json.loads(answer_str)
            return json_answer, []
        except ValueError:
            # Not a valid json ignore
            pass
        return answer_str, []
