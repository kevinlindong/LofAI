"""Dependency-free blind pairwise listening report generation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import random
import shutil
from typing import Any

from .runner import _atomic_json


def _complete_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for case in manifest.get("cases", [])
        if case.get("status") == "complete" and case.get("wav_path")
    ]


def _variant(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "preset_id": case["preset"]["id"],
        "note_mode_id": case["note_mode"]["id"],
        "drum_mode_id": case["drum_mode"]["id"],
        "codebook_depth": case["codebook_depth"],
        "decoder_seed": case["seed"],
        "composition_seed": case["composition_seed"],
        "style_id": case["style"]["id"],
    }


def generate_pairwise_report(
    manifest_path: str | Path,
    *,
    seed: int,
    max_pairs: int = 80,
    hold_seed_constant: bool = True,
) -> tuple[Path, Path]:
    """Create blinded audio copies, a private key, and a standalone report."""

    source_path = Path(manifest_path).resolve()
    run_dir = source_path.parent
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    cases = _complete_cases(manifest)
    if len(cases) < 2:
        raise ValueError("a pairwise report needs at least two completed renders")

    if max_pairs < 0:
        raise ValueError("max_pairs must be >= 0 (0 means no limit)")
    rng = random.Random(seed)
    shuffled_cases = list(cases)
    rng.shuffle(shuffled_cases)
    blind_dir = run_dir / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    blind_for_case: dict[str, str] = {}
    key_cases: list[dict[str, Any]] = []
    for index, case in enumerate(shuffled_cases, start=1):
        blind_id = f"clip-{index:04d}"
        relative = Path("blind") / f"{blind_id}.wav"
        source_wav = run_dir / case["wav_path"]
        if not source_wav.is_file():
            raise FileNotFoundError(source_wav)
        shutil.copyfile(source_wav, run_dir / relative)
        blind_for_case[case["case_id"]] = relative.as_posix()
        key_cases.append({"blind_id": blind_id, "audio": relative.as_posix(), **_variant(case)})

    # Matrix reports normally hold source style and seed constant while note
    # mode, preset, and codebook depth vary. Multi-take reports opt into pairing
    # seeds while still holding style constant.
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for case in cases:
        seed_group = case["seed"] if hold_seed_constant else None
        groups.setdefault((case["style"]["id"], seed_group), []).append(case)
    candidates: list[
        tuple[tuple[str, int | None], dict[str, Any], dict[str, Any]]
    ] = []
    for group, values in groups.items():
        for left, right in itertools.combinations(values, 2):
            candidates.append((group, left, right))
    if not candidates:
        raise ValueError(
            "no comparable pairs: each style/seed group contains only one completed variant"
        )
    rng.shuffle(candidates)
    if max_pairs:
        candidates = candidates[:max_pairs]

    public_trials: list[dict[str, Any]] = []
    key_trials: list[dict[str, Any]] = []
    for index, (group, left, right) in enumerate(candidates, start=1):
        if rng.getrandbits(1):
            left, right = right, left
        trial_id = f"trial-{index:04d}"
        audio_a = blind_for_case[left["case_id"]]
        audio_b = blind_for_case[right["case_id"]]
        public_trials.append(
            {
                "trial_id": trial_id,
                "group": (
                    f"style {group[0]} / seed {group[1]}"
                    if group[1] is not None
                    else f"style {group[0]} / seeded takes"
                ),
                "audio_a": audio_a,
                "audio_b": audio_b,
            }
        )
        key_trials.append(
            {
                "trial_id": trial_id,
                "a_case_id": left["case_id"],
                "b_case_id": right["case_id"],
            }
        )

    key = {
        "schema_version": 1,
        "kind": "lofai_pairwise_blind_key",
        "report_seed": seed,
        "hold_seed_constant": hold_seed_constant,
        "source_manifest": source_path.name,
        "cases": key_cases,
        "trials": key_trials,
    }
    key_path = run_dir / "blind_key.json"
    _atomic_json(key_path, key)

    report_path = run_dir / "listening_report.html"
    payload = json.dumps(public_trials, separators=(",", ":")).replace("</", "<\\/")
    report_path.write_text(_report_html(payload, seed), encoding="utf-8")
    return report_path, key_path


def _report_html(trials_json: str, seed: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>lofAI blind listening comparison</title>
<style>
:root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ max-width: 760px; margin: 3rem auto; padding: 0 1.25rem; line-height: 1.5; }}
.card {{ border: 1px solid #8887; border-radius: 12px; padding: 1.25rem; }}
.players {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
audio {{ width: 100%; }}
button {{ padding: .7rem 1rem; margin: .35rem; cursor: pointer; }}
.choices {{ text-align: center; margin-top: 1rem; }}
textarea, input {{ box-sizing: border-box; width: 100%; padding: .6rem; margin: .3rem 0 1rem; }}
.muted {{ opacity: .72; }}
@media (max-width: 600px) {{ .players {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Blind listening comparison</h1>
<p>Judge which clip you would rather keep listening to. Match playback positions as closely as practical, use the same headphones/volume, and choose Tie when there is no meaningful preference.</p>
<label>Listener ID (use a nickname, not personal information)
  <input id="listener" autocomplete="off">
</label>
<div class="card" id="trial">
  <p><strong id="progress"></strong> <span class="muted" id="group"></span></p>
  <div class="players">
    <section><h2>A</h2><audio id="audio-a" controls preload="metadata"></audio></section>
    <section><h2>B</h2><audio id="audio-b" controls preload="metadata"></audio></section>
  </div>
  <label>Optional note
    <textarea id="note" rows="2" placeholder="What drove the preference?"></textarea>
  </label>
  <div class="choices">
    <button data-choice="a">Prefer A</button>
    <button data-choice="tie">Tie / no preference</button>
    <button data-choice="b">Prefer B</button>
  </div>
</div>
<div id="done" hidden>
  <h2>Complete</h2>
  <p>Download the response file and send it to the study owner. It contains only blinded trial IDs; the separate blind key decodes variants.</p>
  <button id="download">Download results JSON</button>
</div>
<p class="muted">Randomization seed: {seed}. The objective metrics in the render manifest are diagnostics only and are deliberately hidden here; they are not musical-quality verdicts.</p>
<script>
const trials = {trials_json};
const answers = [];
let index = 0;
const a = document.getElementById('audio-a');
const b = document.getElementById('audio-b');
function stopOther(event) {{ const other = event.target === a ? b : a; other.pause(); }}
a.addEventListener('play', stopOther); b.addEventListener('play', stopOther);
function show() {{
  if (index >= trials.length) {{
    document.getElementById('trial').hidden = true;
    document.getElementById('done').hidden = false;
    return;
  }}
  const trial = trials[index];
  document.getElementById('progress').textContent = `Comparison ${{index + 1}} of ${{trials.length}}`;
  document.getElementById('group').textContent = trial.group;
  a.src = trial.audio_a; b.src = trial.audio_b;
  document.getElementById('note').value = '';
}}
document.querySelectorAll('[data-choice]').forEach(button => button.addEventListener('click', () => {{
  const trial = trials[index];
  answers.push({{trial_id: trial.trial_id, choice: button.dataset.choice, note: document.getElementById('note').value.trim()}});
  a.pause(); b.pause(); index += 1; show();
}}));
document.getElementById('download').addEventListener('click', () => {{
  const result = {{schema_version: 1, kind: 'lofai_pairwise_responses', listener_id: document.getElementById('listener').value.trim(), completed_at_utc: new Date().toISOString(), answers}};
  const blob = new Blob([JSON.stringify(result, null, 2) + '\\n'], {{type: 'application/json'}});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = `lofai-listening-${{Date.now()}}.json`; link.click(); URL.revokeObjectURL(link.href);
}});
show();
</script>
</body>
</html>
"""


def decode_pairwise_results(
    key_path: str | Path, results_path: str | Path
) -> list[dict[str, Any]]:
    """Join blinded choices to case IDs for later statistical analysis."""

    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    responses = json.loads(Path(results_path).read_text(encoding="utf-8"))
    trials = {trial["trial_id"]: trial for trial in key["trials"]}
    decoded: list[dict[str, Any]] = []
    for answer in responses.get("answers", []):
        trial = trials.get(answer.get("trial_id"))
        if trial is None:
            raise ValueError(f"response has unknown trial_id {answer.get('trial_id')!r}")
        choice = answer.get("choice")
        if choice not in {"a", "b", "tie"}:
            raise ValueError(f"invalid choice {choice!r}")
        decoded.append(
            {
                "listener_id": responses.get("listener_id", ""),
                "trial_id": answer["trial_id"],
                "a_case_id": trial["a_case_id"],
                "b_case_id": trial["b_case_id"],
                "choice": choice,
                "winner_case_id": (
                    None if choice == "tie" else trial[f"{choice}_case_id"]
                ),
                "note": answer.get("note", ""),
            }
        )
    return decoded
