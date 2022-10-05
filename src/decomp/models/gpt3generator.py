import logging
import os

import openai
from diskcache import Cache

logger = logging.getLogger(__name__)

cache = Cache(os.path.expanduser("~/.cache/gpt3calls"))


@cache.memoize()
def cached_openai_call(  # kwargs doesn't work with caching.
        prompt, engine, temperature, max_tokens, top_p,
        frequency_penalty, presence_penalty, stop,
        n, best_of, logprobs,
):
    return openai.Completion.create(
        prompt=prompt, engine=engine, temperature=temperature, max_tokens=max_tokens,
        top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        stop=stop, n=n, best_of=best_of, logprobs=logprobs
    )


def openai_call(
        prompt, engine, temperature, max_tokens, top_p,
        frequency_penalty, presence_penalty, stop,
        n, best_of, logprobs,
):
    function = cached_openai_call if temperature == 0 else openai.Completion.create
    return function(
        prompt=prompt, engine=engine, temperature=temperature, max_tokens=max_tokens,
        top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        stop=stop, n=n, best_of=best_of, logprobs=logprobs
    )


class GPT3Generator:

    def __init__(self, engine="text-davinci-002", temperature=0, max_tokens=100,
                 top_p=1, frequency_penalty=0, presence_penalty=0, stop=["\n"],
                 n=1, best_of=1, logprobs=0):
        self.engine = engine
        self.logprobs = logprobs
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature

    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        # GPT3 can't handle trailing white-space
        prompt = prompt.rstrip()
        if self.best_of is None:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n=self.n,
                logprobs=self.logprobs,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop
            )
        else:
            response = openai_call(
                engine=self.engine,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n=self.n,
                best_of=self.best_of,
                logprobs=self.logprobs,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop
            )
        output_seq_score = []

        for index, choice in enumerate(response["choices"]):
            # print(choice)
            if "logprobs" in choice and "token_logprobs" in choice["logprobs"]:
                # get probs of the tokens used in text (i.e. till the stop token)
                probs = []
                # selected_toks = []
                for prob, tok in zip(choice["logprobs"]["token_logprobs"],
                                     choice["logprobs"]["tokens"]):
                    if tok not in self.stop and tok != "<|endoftext|>":
                        probs.append(prob)
                        # selected_toks.append(tok)
                    else:
                        # include the probability of the stop character too. This will also
                        # ensure that an empty string (i.e. first predicted character being a stop
                        # character) also has a reasonable probability measure
                        # selected_toks.append(tok)
                        probs.append(prob)
                        break
                # average the logits and negate to make them +ve scores where lower is better
                # set a high +ve score if no predictions
                # print(probs, selected_toks)
                score = -sum(probs) / len(probs) if len(probs) else 100.0
                output_seq_score.append((choice["text"], score))
            else:
                # no score, just use index
                output_seq_score.append((choice["text"], index))

        # Ensure sorted output
        return sorted(output_seq_score, key=lambda x: x[1])
