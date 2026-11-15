from experiments.compare import onestep_table
from worldmodel.features import KEY_TARGETS


def _res(medae):
    return {"sets": {"adaptive": {"all": {"mlp": {t: {"mae": medae * 2, "medae": medae, "r2": 0.5, "n": 3}
                                                  for t in KEY_TARGETS}}}}}


def test_compare_table_rows():
    lines = onestep_table({"a": _res(1.0), "b": _res(0.5)}, "adaptive")
    rows = [l for l in lines if l.startswith("| ") and not l.startswith("| target")]
    assert len(rows) == len(KEY_TARGETS)
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        assert cells[1] == "2 / 1" and cells[2] == "**1 / 0.5**"     # best median is bolded
