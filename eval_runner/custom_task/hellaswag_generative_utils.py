import re
import datasets


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        choices = [preprocess(ending) for ending in doc["endings"]]
        label = int(doc["label"])
        letters = ["A", "B", "C", "D"]
        return {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": choices,
            "answer": letters[label],
        }

    return dataset.map(_process)


def doc_to_text(doc):
    query = doc["query"]
    choices = doc["choices"]
    letters = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, choices)
    )
    return (
        f"Choose the most plausible continuation for the following scenario. "
        f"Reply with ONLY the letter (A, B, C, or D) of the best answer.\n\n"
        f"{query}\n\n{choices_text}\n\nThe correct answer is:"
    )
