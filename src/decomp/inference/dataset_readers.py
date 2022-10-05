import json


class DatasetReader:

    def __init__(self, add_paras=False, add_gold_paras=False):
        self.add_paras = add_paras
        self.add_gold_paras = add_gold_paras

    def read_examples(self, file):
        return NotImplementedError("read_examples not implemented by " + self.__class__.__name__)


class HotpotQAReader(DatasetReader):

    def read_examples(self, file):
        with open(file, 'r') as input_fp:
            input_json = json.load(input_fp)

        for entry in input_json:
            output = {
                "qid": entry["_id"],
                "query": entry["question"],
                # metadata
                "answer": entry["answer"],
                "question": entry["question"],
                "type": entry.get("type", ""),
                "level": entry.get("level", "")
            }
            if self.add_paras:
                title_doc_map = self.get_paras(entry)
                if self.add_gold_paras:
                    output_paras = self.get_gold_paras(entry, title_doc_map)
                else:
                    output_paras = title_doc_map
                output["paras"] = [title + "||" + " ".join(sentences)
                                   for title, sentences in output_paras.items()]
            yield output

    def read_n_examples(self, file, n):
        with open(file, 'r') as input_fp:
            input_json = json.load(input_fp)
        read = 0
        for entry in input_json:
            output = {
                "qid": entry["_id"],
                "query": entry["question"],
                # metadata
                "answer": entry["answer"],
                "question": entry["question"],
                "type": entry.get("type", ""),
                "level": entry.get("level", "")
            }
            if self.add_paras:
                title_doc_map = self.get_paras(entry)
                if self.add_gold_paras:
                    output_paras = self.get_gold_paras(entry, title_doc_map)
                else:
                    output_paras = title_doc_map
                output["paras"] = [title + "||" + " ".join(sentences)
                                   for title, sentences in output_paras.items()]
            if n is None or read < n:
                yield output
                read += 1
            else:
                break


    def get_gold_paras(self, entry, title_doc_map):
        supporting_facts = entry["supporting_facts"]
        collected_facts = {}
        for (doc, idx) in supporting_facts:
            if doc not in collected_facts:
                collected_facts[doc] = title_doc_map[doc]
        return collected_facts

    def get_paras(self, entry):
        # collect title->doc map
        title_doc_map = {}
        for title, document in entry["context"]:
            if title in title_doc_map:
                # Don't raise exception. Expected behavior with 2WikiMulithopQA :(
                print("Two documents with same title: {} in {}".format(title, entry["_id"]))
                continue
            title_doc_map[title] = [doc.strip() for doc in document]
        return title_doc_map


def format_drop_answer(answer_json):
    if answer_json["number"]:
        return answer_json["number"]
    if len(answer_json["spans"]):
        return answer_json["spans"]
    # only date possible
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        print("Number, Span or Date not set in {}".format(answer_json))
        return None
    return date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"]


class DropReader(DatasetReader):

    def read_examples(self, file):
        with open(file, 'r') as input_fp:
            input_json = json.load(input_fp)

        for paraid, item in input_json.items():
            para = item["passage"]
            for qa_pair in item["qa_pairs"]:
                question = qa_pair["question"]
                qid = qa_pair["query_id"]
                answer = format_drop_answer(qa_pair["answer"])
                output = {
                    "qid": qid,
                    "query": question,
                    # metadata
                    "answer": answer,
                    "question": question
                }
                if self.add_paras:
                    output["paras"] = [para]
                yield output

    def read_n_examples(self, file, n):
        with open(file, 'r') as input_fp:
            input_json = json.load(input_fp)
        read = 0
        for paraid, item in input_json.items():
            para = item["passage"]
            for qa_pair in item["qa_pairs"]:
                question = qa_pair["question"]
                qid = qa_pair["query_id"]
                answer = format_drop_answer(qa_pair["answer"])
                output = {
                    "qid": qid,
                    "query": question,
                    # metadata
                    "answer": answer,
                    "question": question
                }
                if self.add_paras:
                    output["paras"] = [para]
                if n is None or read < n:
                    yield output
                    read += 1
                else:
                    break
