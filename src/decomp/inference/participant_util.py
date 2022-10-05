import re

from decomp.inference.data_instances import QuestionAnsweringStep
from decomp.inference.model_search import ParticipantModel
from decomp.inference.utils import get_sequence_representation


class DumpChainsParticipant(ParticipantModel):

    def __init__(self, output_file, next_model="gen"):
        self.output_file = output_file
        self.next_model = next_model
        self.num_calls = 0

    def return_model_calls(self):
        return {"dumpchains": self.num_calls}

    def dump_chain(self, state):
        data = state.data
        origq = data["query"]
        qchain = data.get_current_qseq()
        achain = data.get_current_aseq()
        sequence = get_sequence_representation(origq=origq, question_seq=qchain, answer_seq=achain)
        ans = achain[-1]
        with open(self.output_file, 'a') as chains_fp:
            chains_fp.write(data["qid"] + "\t" + sequence + "\t" + ans + "\n")

    def query(self, state, debug=False):
        self.num_calls += 1
        if len(state.data["question_seq"]) > 0:
            self.dump_chain(state)
        new_state = state.copy()
        new_state.next = self.next_model
        return new_state


class AnswerExtractor(ParticipantModel):
    def __init__(self, regex, next_model="[EOQ]"):
        self.regex = re.compile(regex)
        self.next_model = next_model
        self.num_calls = 0

    def return_model_calls(self):
        return {"extract": self.num_calls}

    def query(self, state, debug=False):
        self.num_calls += 1

        new_state = state.copy()
        question = new_state.data.get_last_question()
        m = self.regex.match(question)
        if m:
            answer = m.group(1)
            if debug:
                print("EXT: " + answer)

            new_state.data.add_answer(QuestionAnsweringStep(
                answer=answer,
                score=0,
                participant=state.next
            ))
            # new_state.data["answer_seq"].append(answer)
            # new_state.data["para_seq"].append("")
            # new_state.data["command_seq"].append("qa")
            # new_state.data["model_seq"].append("extractor")
            # new_state.data["operation_seq"].append("")
            # new_state.data["subquestion_seq"].append(question)
            ## change output
            new_state.last_output = answer
            new_state.next = self.next_model
            return new_state
        else:
            # No match
            print("Answer Extractor did not find a match for input regex in {}".format(question))
            return []

