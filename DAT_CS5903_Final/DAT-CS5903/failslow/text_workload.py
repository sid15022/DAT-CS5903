"""Natural-language prompts with precise work sizes for the measured server study."""

from functools import lru_cache

from .engines import MODEL, REVISION

TASKS = (
    (
        "The train departs at nine and reaches the last station at eleven. ",
        "How long is the journey?",
    ),
    (
        "A bakery makes twelve rolls each morning and sells eight before lunch. ",
        "How many rolls are left after lunch?",
    ),
    (
        "The observatory uses a telescope to study planets and stars. ",
        "Which instrument does the observatory use?",
    ),
    (
        "Prefill processes the input and decode generates one token at a time. ",
        "Name the two inference phases.",
    ),
)
PILOT_TASK = (
    "Paris is the capital of France. Lyon is another French city. ",
    "What is the capital of France? Answer in one short sentence.",
)


class TextWorkload:
    def __init__(self, tokenizer=None):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.tok = tokenizer

    @lru_cache(maxsize=8)
    def parts(self, task):
        context, question = task
        template = self.tok.apply_chat_template(
            [{"role": "user", "content": "DATIDENTIFIER DATCONTEXT\n" + question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix, remaining = template.split("DATIDENTIFIER")
        middle, suffix = remaining.split("DATCONTEXT")

        def encode(text):
            return self.tok.encode(text, add_special_tokens=False)

        return encode(prefix), encode(middle), encode(context), encode(suffix)

    def prompt(self, trial_seed, rid, length, pilot=False):
        index = 0 if pilot else int(rid[1:])
        task = PILOT_TASK if pilot else TASKS[(trial_seed + index) % len(TASKS)]
        prefix, middle, context, suffix = self.parts(task)
        key = self.tok.encode(f"Case {trial_seed}-{index}.", add_special_tokens=False)
        head = prefix + key + middle
        space = length - len(head) - len(suffix)
        if space < len(context):
            raise ValueError(
                f"Token budget {length} is too small for the workload template"
            )
        body = (context * (1 + space // len(context)))[:space]
        return head + body + suffix
