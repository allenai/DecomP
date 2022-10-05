import requests


class LLMClientGenerator:

    # Instructions to start the LLM Server are in the README here:
    # https://github.com/harshTrivedi/llm_server

    def __init__(self, model_name, host, port,
                 max_input=None, max_tokens=100,
                 min_length=1, do_sample=False, stop=["\n"],
                 temperature=1.0, top_k=50, top_p=1.0,
                 num_return_sequences=1, repetition_penalty=None,
                 length_penalty=None):

        valid_model_names = ["gpt-j-6B", "opt-66b", "gpt-neox-20b", "T0pp"]
        assert model_name in valid_model_names, \
            f"Model name {model_name} not in {valid_model_names}"

        self.model_name = model_name
        self.host = host
        self.port = port
        self.max_input = max_input
        self.max_length = max_tokens
        self.min_length = min_length
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop = stop
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        prompt = prompt.rstrip()

        params = {
            "prompt": prompt,
            "max_input": self.max_input,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
        }
        response = requests.get(self.host + ":" + str(self.port) + "/generate", params=params)

        if response.status_code != 200:
            raise Exception("LLM Generation request failed!")

        result = response.json()
        generated_texts = result.get("generated_texts", "")
        modified_texts = []
        for text in generated_texts:
            # remove the prompt
            if text.startswith(prompt):
                text = text[len(prompt):]
            print(text)
            if self.stop:
                for stop_str in self.stop:
                    if stop_str in text:
                        text = text[:text.index(stop_str)]
            print(text)
            modified_texts.append(text)
        generated_texts = modified_texts
        model_name = result.get("model_name",
                                "")  # To assure that response is from the right model.

        if model_name != self.model_name:
            raise Exception(
                f"Looks like incorrect LLM server is ON: {model_name} != {self.model_name}.")

        output_seq_score = [(text, 1 / (index + 1)) for index, text in enumerate(generated_texts)]

        # TODO: Deal with output-probabilities if needed.

        return sorted(output_seq_score, key=lambda x: x[1])
