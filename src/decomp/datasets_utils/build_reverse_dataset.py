# Reverse the sequence "card, stamp, book, water, glasses".

import argparse
import logging
import json
import math
import os
import random
import string

log = logging.getLogger(__name__)

questions = {
    ("reversed", "words"): 'Reverse the sequence "{}".',
    ("reversed", "letters"): 'Reverse the sequence "{}".',
    ("reversed", "numbers"): 'Reverse the sequence "{}".',
    ("sorted", "words"): 'Sort alphabetically the sequence "{}".',
    ("sorted", "letters"): 'Sort alphabetically the sequence "{}".',
    ("sorted", "numbers"): 'Sort the sequence "{}".',
    ("sort_last", "words"): 'Sort alphabetically by last letter the sequence "{}".',
    ("sort_last", "numbers"): 'Sort by last digit the sequence "{}".',
}


def main():
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--elts", choices=["words", "letters", "numbers"])
    parser.add_argument("--listop", choices=["reversed", "sorted", "sort_last"])
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--train_size", type=int, default=10)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--word_list", default="configs/data/wordlist.txt")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--length", default=[4], nargs="+", type=int)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--delim", default=", ")

    args = parser.parse_args()

    with open(args.word_list, "r") as f:
        wordlist = list(map(str.strip, f))

    lst = (
        string.ascii_letters
        if args.elts == "letters"
        else wordlist
        if args.elts == "words"
        else list(range(1_000))
    )
    question = questions[args.listop, args.elts]
    if args.hard and args.elts == "words":
        sublists = get_hard_sublists(
            lst, args.length, args.num_examples, args.listop
        )
    else:
        try:
            sublists = get_sublists(lst, args.length, args.num_examples)
        except OverflowError:
            sublists = repeated_sample(lst, args.length, args.num_examples)
    func = eval(args.listop)
    qa_pairs = [listop(func, args.delim, question, sublist) for sublist in sublists]
    qa_pairs = [
        {
            "question": question,
            "answer": {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [answer],
            },
            "query_id": str(i),
            "validated_answers": [],
        }
        for i, (question, answer) in enumerate(qa_pairs)
    ]
    qa_pairs = random.sample(qa_pairs, len(qa_pairs))
    test = {
        "alg_qa": {
            "passage": "",
            "qa_pairs": qa_pairs[: -args.train_size],
        }
    }
    train = {
        "alg_qa": {
            "passage": "",
            "qa_pairs": qa_pairs[-args.train_size :],
        }
    }
    filename = "{}_{}_{}.json".format(
        str.join("_", map(str, args.length)),
        "hard" if args.hard else "normal",
        args.elts,
        args.num_examples,
    )
    if args.cot:
        cot_examples = map(get_cot, random.sample(train["alg_qa"]["qa_pairs"], 5))
        with open("cot.txt", "w") as f:
            f.write(str.join("\n\n", cot_examples))
    with open(os.path.join(args.out_dir, "test_" + filename), "w") as f:
        json.dump(test, f, indent=2)
    with open(os.path.join(args.out_dir, "train_" + filename), "w") as f:
        json.dump(train, f, indent=2)


def get_cot(qa_pair: dict) -> str:
    answer = qa_pair["answer"]["spans"][0]
    lines = [
        "QC: {}".format(qa_pair["question"]),
        'QS: The answer is "{}".'.format(answer),
        "A: {}".format(answer),
        "QS: [EOQ]",
    ]
    return str.join("\n", lines)


def listop(func, sep, question, sublist):
    sequence = sep.join(map(str, sublist))
    question = question.format(sequence)
    answer = sep.join(map(str, func(sublist)))
    return question, answer


def repeated_sample(lst, lengths: list, num_examples):
    samples = num_examples // len(lengths)
    log.info("sampling {} instances".format(samples * len(lengths)))
    return (random.sample(lst, l) for _ in range(samples) for l in lengths)


def get_hard_sublists(lst, lengths: list, num_examples, listop):
    if len(lengths) > 1:
        log.warning("--hard not implemented for multiple lengths. Only the first length will be used.")
    length = lengths[0]
    if listop == "reversed":
        return get_sublists(lst, [length], num_examples)
    elif listop == "sorted" or listop == "sort_last":
        pos = 0 if listop == "sorted" else -1
        output_list = []
        valid_list = lst.copy()
        while len(output_list) < num_examples:
            seed_word = random.choice(valid_list)
            wordlist = [x for x in valid_list if x[pos] == seed_word[pos]]
            if len(wordlist) > length:
                output_list.append(random.sample(wordlist, length))
            else:
                for x in wordlist:
                    valid_list.remove(x)
        return output_list
    else:
        raise ValueError("Hard examples not implemented for {}".format(listop))


def get_sublists(lst, lengths: list, num_examples):
    sublists = list()
    samples = num_examples // len(lengths)
    log.info("sampling {} instances".format(samples * len(lengths)))
    for l in lengths:
        permutation_count = math.perm(len(lst), l)
        permutation_idxs = random.sample(range(permutation_count), samples)
        sublists.extend(get_permutation(i, lst, l) for i in permutation_idxs)
    return sublists


def get_permutation(i, lst, length):
    permutation = list()
    for _ in range(length):
        i, idx = divmod(i, len(lst))
        permutation.append(lst[idx])
        # remove to prevent duplicates
        lst = lst[:idx] + lst[idx + 1 :]
    return permutation


def sort_last(sublist):
    return sorted(sublist, key=lambda x: str(list(reversed(x))))


if __name__ == "__main__":
    main()
