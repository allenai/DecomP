import json
from datetime import datetime

from dateutil.parser import parse

from decomp.models.llm_qa_model import LLMQAModel
from decomp.inference.data_instances import QuestionAnsweringStep, StructuredDataInstance
from decomp.inference.model_search import ParticipantModel
from decomp.inference.participant_qgen import QuestionGenParticipant


class LLMQAParticipantModel(ParticipantModel):
    def __init__(self, next_model=None, end_state="[EOQ]", allow_empty_answers=False, **kwargs):
        self.qa_model = LLMQAModel(**kwargs)
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0
        self.allow_empty_answers = allow_empty_answers

    def return_model_calls(self):
        return {"llm_qa": self.num_calls}

    def update_state(self, answer, state):
        if not self.allow_empty_answers and answer == "":
            return []
        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(
            answer=json.dumps(answer),
            score=0,
            participant=state.next
        ))
        new_state.next = self.next_model if self.next_model else self.end_state
        return new_state

    def query(self, state, debug=False):
        question = state.data.get_last_question()
        context = state.data["paras"] if "paras" in state.data else ""
        self.num_calls += 1
        answer, facts_used = self.qa_model.ask_question(input_question=question, context=context)
        return self.update_state(answer=answer, state=state)


class LLMQADecompParticipantModel(QuestionGenParticipant):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self, state, debug=False):
        # Is this being called to generate a question?
        if len(state.data.get_current_inference_seq()) == 0 or \
                isinstance(state.data.get_last_step(), QuestionAnsweringStep):
            # if there is no previous question or the last step was a QA Step
            new_states = super().query(state=state, debug=debug)
        else:
            # or answer a question
            new_state = state.copy()
            question = state.data.get_last_question()
            # take the last question and a decomposition level
            new_state.data.add_subdecomp(StructuredDataInstance(input_data={
                "qid": state.data["qid"],
                "query": question,
                "question": question
            }))
            # then generate the decomposition
            new_states = super().query(state=new_state, debug=debug)
        if not isinstance(new_states, list):
            new_states = [new_states]

        for new_state in new_states:
            # if [EOQ] was generated, i.e. the module is done answering this question
            if new_state.next == self.end_state and not new_state.data.at_root_level():
                last_answer = new_state.data.get_last_answer()
                new_state.data.popup_decomp_level()
                new_state.data.add_answer(QuestionAnsweringStep(
                    answer=last_answer,
                    score=0,
                    participant=state.next
                ))
        return new_states


def date_difference(date1: str, date2: str, units: str = "years"):
    default_date = datetime(3000, 1, 1)
    try:
        date1_datetime = parse(date1, default=default_date)
        date2_datetime = parse(date2, default=default_date)
    except Exception:
        # couldn't parse date
        return None
    # if one doesn't have month set, not usable
    if date1_datetime.year == default_date.year and date1_datetime.month == default_date.month:
        return None
    if date2_datetime.year == default_date.year and date2_datetime.month == default_date.month:
        return None

    if date1_datetime.year == default_date.year and date2_datetime.year != default_date.year:
        # one date is relative and other is not
        date1_datetime = date1_datetime.replace(year=date2_datetime.year)
    elif date2_datetime.year == default_date.year and date1_datetime.year != default_date.year:
        # one date is relative and other is not
        date2_datetime = date2_datetime.replace(year=date1_datetime.year)

    if units == "days":
        return (date1_datetime - date2_datetime).days
    if units == "months":
        return (date1_datetime.year - date2_datetime.year) * 12 + (
                date1_datetime.month - date2_datetime.month)
    if units == "years":
        # human annotations are often on just the year value
        return date1_datetime.year - date2_datetime.year
    print("Unknown unit:" + units)
    return None


def sort_without_duplicates(arr):
    last_val = None
    output_arr = []
    for (key, val) in sorted(arr, key=lambda x: x[1]):
        if val == last_val:
            continue
        else:
            output_arr.append((key, val))
            last_val = val
    return output_arr


def sorted_key(arr):
    return [x[0] for x in sort_without_duplicates(arr)]


def sorted_value(arr):
    return [x[1] for x in sort_without_duplicates(arr)]


class ExprEvalQAParticipantModel(ParticipantModel):
    def __init__(self, next_model=None, end_state="[EOQ]",
                 **kwargs):
        """
        :param kwargs: Unused. Only here for ease of JSON specification
        """
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0

    def return_model_calls(self):
        return {"expr_eval": self.num_calls}

    def query(self, state, debug=False):
        question = state.data.get_last_question()
        try:
            answer = eval(question)
            if isinstance(answer, float):
                answer = round(answer, 3)
            elif isinstance(answer, set):
                answer = list(answer)
        except Exception:
            # could not evaluate question
            answer = None
        self.num_calls += 1
        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(
            answer=json.dumps(answer),
            score=0,
            participant=state.next
        ))
        if answer is None:
            return []
        else:
            new_state.next = self.next_model if self.next_model else self.end_state
        return new_state
