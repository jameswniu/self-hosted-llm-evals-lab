"""Guard the claims on the landing page against the data they came from.

Every number in the README is supposed to be re-derivable from the results
files. These tests fail if someone edits a figure caption without the data, or
edits the data without regenerating the figures.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABLATION = ROOT / "ablation" / "results" / "ablation_results.json"


def _data():
    return json.loads(ABLATION.read_text())


def test_published_accuracies_match_the_results_file():
    """These five values are printed in the README table and drawn in the figure."""
    expected = {
        "template_baseline": 0.60,
        "template_instruction": 0.55,
        "template_cot": 0.35,
        "template_fewshot_cot": 0.55,
        "self_consistency_k5": 0.70,
    }
    data = _data()
    for key, acc in expected.items():
        assert key in data, f"{key} missing from ablation results"
        assert abs(data[key]["accuracy"] - acc) < 1e-9, f"{key} drifted"


def test_chain_of_thought_is_the_worst_strategy():
    """The headline claim of the repo, asserted rather than assumed."""
    data = _data()
    accs = {k: v["accuracy"] for k, v in data.items()}
    assert min(accs, key=accs.get) == "template_cot"


def test_self_consistency_is_the_only_strategy_above_baseline():
    data = _data()
    base = data["template_baseline"]["accuracy"]
    better = [k for k, v in data.items() if v["accuracy"] > base]
    assert better == ["self_consistency_k5"]


def test_every_figure_the_readme_references_exists():
    readme = (ROOT / "README.md").read_text()
    for name in ("accuracy_by_strategy", "confidence_routing",
                 "latency_throughput", "grading_failure"):
        path = ROOT / "docs" / "figures" / f"{name}.svg"
        assert path.exists(), f"{name}.svg is referenced but missing"
        assert f"docs/figures/{name}.svg" in readme


def test_assets_referenced_by_the_readme_exist():
    readme = (ROOT / "README.md").read_text()
    for name in ("hero", "architecture", "ablation-pipeline", "confidence-routing"):
        assert (ROOT / "assets" / f"{name}.svg").exists(), f"assets/{name}.svg missing"
        assert f"assets/{name}.svg" in readme
