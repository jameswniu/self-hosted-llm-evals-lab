"""Answer extraction is where this eval pipeline actually breaks.

A graded MMLU run in eval_runner/results/ scored 0.0021, far below the 0.25 you
get by guessing, because the harness filter harvested the first capital A-D out
of ordinary prose. These tests pin the behaviour that prevents the same class of
bug here: a letter is only an answer when the response says so.
"""
import pytest
import optimize_prompt

extract_answer = optimize_prompt.extract_answer
A, B, C, D = 0, 1, 2, 3


@pytest.mark.parametrize("response,expected", [
    ("A", A), ("C", C), ("  D  ", D),          # the whole response is the choice
    ("(B)", B), ("A.", A), ("D:", D),          # wrapped or punctuated
    ("B) because it scales", B),               # leading choice, then reasoning
    ("Answer: C", C), ("answer is D", D),      # explicitly stated
    ("I would say option B is best", B),
    ("The answer is (A).", A),
])
def test_extracts_a_stated_choice(response, expected):
    assert extract_answer(response) == expected


@pytest.mark.parametrize("response", [
    "A nice abstract algebra question!",   # the article "A", not choice A
    "A lot of people would disagree",
    "Based on the passage, the correct answer is:",   # truncated before the letter
    "The correct answer is:",
    "I am not sure about this one.",
    "",
    "   ",
])
def test_returns_none_when_no_choice_is_stated(response):
    """No answer must stay distinguishable from a wrong answer.

    Collapsing them into the same zero is what hides whether a run has a prompt
    problem, a stop-sequence problem, or a model problem.
    """
    assert extract_answer(response) is None


def test_a_stated_answer_beats_a_leading_article():
    """Regression: the original tier 1 accepted whitespace as a delimiter, so
    this string extracted A and silently overrode the C the model actually gave."""
    assert extract_answer("A lot of people think the answer is C") == C


def test_prose_beginning_with_an_article_is_not_an_answer():
    """Regression, taken verbatim from a logged harness sample."""
    assert extract_answer("A nice abstract algebra question!") is None
