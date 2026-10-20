import json

import pytest

from core_deepinfer.cli import main


@pytest.mark.parametrize("experiment", ["all", "replay", "budget", "cache", "speculative"])
def test_each_experiment_emits_machine_readable_results(experiment, capsys):
    assert main([experiment, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    if experiment != "all":
        assert experiment in result
    else:
        assert {"replay", "budget", "cache", "speculative"} <= result.keys()


def test_invalid_keep_is_an_explicit_cli_error(capsys):
    with pytest.raises(SystemExit) as error:
        main(["replay", "--keep", "1"])
    assert error.value.code == 2
    assert "keep must cover" in capsys.readouterr().err
