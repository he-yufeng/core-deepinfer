import json

import pytest

from core_deepinfer import Engine, EngineConfig, TinyModel
from core_deepinfer.cli import main
from core_deepinfer.demo import run_demo


def test_default_command_is_the_integrated_engine_demo_not_four_unconnected_experiments(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Greedy output matches: True" in output
    assert "cold_and_prefix_reuse" in output
    assert "warm" in output
    assert "random, untrained" in output
    assert "not a speedup" in output


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_demo_uses_real_engine_cache_expiry_and_matches_the_oracle(seed):
    demo = run_demo(seed=seed, include_trace=True)
    assert demo["same_tokens_as_greedy"]
    assert demo["committed_state_max_error"] < 1e-10
    assert [stage["restores"][0]["cache"] for stage in demo["stages"]] == ["cold", "hot", "warm", "cold"]
    assert demo["stages"][0]["results"][-1]["finish_reason"] == "cancelled"
    assert all(stage["events"] for stage in demo["stages"])
    assert "in-vocabulary" in demo["rejected_prompt_error"]


def test_demo_json_and_trace_are_machine_readable(capsys):
    assert main(["demo", "--trace", "--json"]) == 0
    demo = json.loads(capsys.readouterr().out)["demo"]
    events = [event for stage in demo["stages"] for event in stage["events"]]
    assert any(event["kind"] == "budget" for event in events)
    assert any(event.get("discarded_target", 0) > 0 for event in events)
    assert demo["same_tokens_as_greedy"]


def test_generate_supports_repeated_prompts_and_budgeted_execution(capsys):
    assert (
        main(
            [
                "generate",
                "--prompt",
                "1,2,3",
                "--prompt",
                "4,5,6",
                "--max-tokens",
                "4",
                "--batch-budget",
                "2",
                "--trace",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)["generation"]
    assert [len(row["tokens"]) for row in result["results"]] == [4, 4]
    assert result["exact_recovery"]
    assert all(event["total_tokens"] <= 2 for event in result["events"] if event["kind"] == "budget")


def test_json_batch_file_preserves_request_ids_and_per_request_limits(tmp_path, capsys):
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            [
                {"request_id": "short", "prompt": [1, 2, 3], "max_tokens": 1},
                {"request_id": "long", "prompt": [1, 2, 3, 4], "max_tokens": 5},
            ]
        )
    )
    assert main(["generate", "--input", str(path), "--greedy", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["generation"]
    assert [row["request_id"] for row in result["results"]] == ["short", "long"]
    assert [len(row["tokens"]) for row in result["results"]] == [1, 5]
    assert result["work"]["draft"]["incremental_steps"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--prompt", "1,,2"],
        ["--prompt", "1000"],
        ["--max-tokens", "-1"],
        ["--batch-budget", "0"],
        ["--draft-width", "0"],
        ["--recovery", "compact", "--keep", "1"],
    ],
)
def test_generate_invalid_input_is_an_explicit_cli_error(args, capsys):
    with pytest.raises(SystemExit) as error:
        main(["generate", *args])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("payload", [[], {}, [1], [{"prompt": [1], "unknown": True}], [{"max_tokens": 1}]])
def test_malformed_batch_files_do_not_produce_a_short_successful_result(payload, tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as error:
        main(["generate", "--input", str(path), "--json"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert "error:" in output.err


def test_public_library_exports_the_complete_loop():
    model = TinyModel()
    engine = Engine(model, config=EngineConfig(max_active=1))
    request = engine.submit([1, 2, 3], max_tokens=2)
    engine.run()
    assert engine.result(request).state.tokens == (1, 2, 3, *engine.result(request).tokens)
