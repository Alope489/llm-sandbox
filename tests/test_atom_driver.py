import subprocess

from src.multi.materials.atom_driver import AtomAgentDriver


def test_run_materials_sim_returns_output_and_latency(monkeypatch):
    driver = AtomAgentDriver()
    observed = {}

    def fake_run(cmd, capture_output, text, env, check):
        observed["cmd"] = cmd
        observed["capture_output"] = capture_output
        observed["text"] = text
        observed["check"] = check
        observed["env_has_openai"] = "OPENAI_API_KEY" in env
        observed["env_has_chroma"] = "CHROMA_OPENAI_API_KEY" in env
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    times = iter([100.0, 100.25])
    monkeypatch.setattr("src.multi.materials.atom_driver.time.time", lambda: next(times))
    monkeypatch.setattr("src.multi.materials.atom_driver.subprocess.run", fake_run)

    result = driver.run_materials_sim("Bulk Nickel")

    assert result["output"] == "ok"
    assert result["latency_seconds"] == 0.25
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert observed["check"] is False
    assert observed["env_has_openai"] is True
    assert observed["env_has_chroma"] is True
    assert observed["cmd"][1].endswith("AtomAgents.py")
    assert observed["cmd"][2:] == ["--prompt", "Bulk Nickel"]
