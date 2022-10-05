from typing import Dict

from decomp.inference.dataset_readers import HotpotQAReader, DatasetReader, DropReader
from decomp.inference.participant_execution_routed import RoutedExecutionParticipant
from decomp.inference.participant_qa import ExprEvalQAParticipantModel, LLMQAParticipantModel, \
    LLMQADecompParticipantModel
from decomp.inference.participant_qgen import QuestionGenParticipant, RandomGenParticipant
from decomp.inference.participant_util import DumpChainsParticipant, AnswerExtractor

MODEL_NAME_CLASS = {
    "lmgen": QuestionGenParticipant,  # for backward-compatibility
    "randgen": RandomGenParticipant,
    "dump_chains": DumpChainsParticipant,
    "execute_router": RoutedExecutionParticipant,
    "answer_extractor": AnswerExtractor,
    "llmqa": LLMQAParticipantModel,
    "llmqadecomp": LLMQADecompParticipantModel,
    "expr_eval": ExprEvalQAParticipantModel
}

READER_NAME_CLASS: Dict[str, DatasetReader] = {
    "hotpot": HotpotQAReader,
    "drop": DropReader
}
