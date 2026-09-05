"""Command-line entry point for MRT2 evaluation renders and listening tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from .audio_metrics import METRICS_DISCLAIMER, analyze_audio
from .mrt_adapter import production_adapter_factory
from .report import decode_pairwise_results, generate_pairwise_report
from .runner import _safe_json, read_pcm16_wav, run_evaluation
from .shortlist import build_signal_shortlist
from .spec import FRAME_RATE, evaluation_spec_from_dict, load_evaluation_spec


DEFAULT_QUALITY_PROMPT = "dusty lo-fi hip hop beat, warm vinyl, mellow jazz guitar"


def _add_quality_render_args(parser: argparse.ArgumentParser, *, multi: bool) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-size", choices=("mrt2_base", "mrt2_small"), default="mrt2_base"
    )
    parser.add_argument(
        "--bits",
        choices=(0, 4, 8),
        type=int,
        default=8,
        help="weight quantization; 0 keeps full precision",
    )
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--warmup-bars", type=int)
    warmup.add_argument("--warmup-seconds", type=float)
    parser.add_argument("--station", default="dusty-beats")
    parser.add_argument(
        "--prompt",
        help="custom audible style tags; use with --station custom",
    )
    parser.add_argument("--mood", choices=("somber", "neutral", "lively"))
    parser.add_argument("--instrument", choices=("piano", "guitar", "brass"))
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--groove", type=float)
    parser.add_argument("--intensity", type=float)
    parser.add_argument(
        "--melody", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--drums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--note-mode", choices=("masked", "app_melody"), default="app_melody")
    parser.add_argument(
        "--drum-mode", choices=("planned", "masked", "off"), default="planned"
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--audio-reference", type=Path)
    reference.add_argument(
        "--text-only",
        action="store_true",
        help="disable any station WAV discovered through MRT_STYLE_REFERENCE_DIR",
    )
    parser.add_argument("--audio-style-blend", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cfg-musiccoca", type=float, default=1.6)
    parser.add_argument("--cfg-notes", type=float, default=2.4)
    parser.add_argument("--cfg-drums", type=float, default=4.0)
    parser.add_argument("--style-token-levels", type=int, choices=range(1, 13), default=6)
    parser.add_argument(
        "--sampling-mode", choices=("production", "raw"), default="production"
    )
    parser.add_argument(
        "--composition-seed",
        type=int,
        help="hold composition fixed while decoder seeds vary (multi-take)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    if multi:
        parser.add_argument("--takes", type=int, default=6)
        parser.add_argument("--seed-start", type=int, default=20260903)
        parser.add_argument("--shortlist-count", type=int, default=3)
        parser.add_argument("--report-seed", type=int, default=314159)
        parser.add_argument("--max-pairs", type=int, default=80)
    else:
        parser.add_argument("--seed", type=int, default=20260903)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.evaluation",
        description="Fixed-seed Magenta RealTime 2 render/listening evaluation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="validate and print a matrix without loading MRT2")
    plan.add_argument("--config", required=True, type=Path)

    render = commands.add_parser("render", help="render the configured matrix")
    render.add_argument("--config", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    render.add_argument("--resume", action="store_true")
    render.add_argument("--continue-on-error", action="store_true")
    render.add_argument(
        "--report",
        choices=("none", "pairwise"),
        default="pairwise",
        help="create a static blind report after rendering (default: pairwise)",
    )

    candidate = commands.add_parser(
        "candidate",
        help="render one quality-first 30-90s offline candidate at 12 codebooks",
    )
    _add_quality_render_args(candidate, multi=False)

    multi_take = commands.add_parser(
        "multi-take",
        help="render seeded 30-90s takes, triage signal risks, and build blind review",
    )
    _add_quality_render_args(multi_take, multi=True)

    analyze = commands.add_parser(
        "analyze", help="calculate the same descriptive metrics for PCM16 WAV files"
    )
    analyze.add_argument("wav", nargs="+", type=Path)
    analyze.add_argument("--output", type=Path)

    report = commands.add_parser(
        "report", help="rebuild a blind pairwise report from a render manifest"
    )
    report.add_argument("--manifest", required=True, type=Path)
    report.add_argument("--seed", type=int, default=20260903)
    report.add_argument("--max-pairs", type=int, default=80)
    report.add_argument(
        "--compare-across-seeds",
        action="store_true",
        help="pair different takes/seeds of a style (useful for multi-take review)",
    )

    shortlist = commands.add_parser(
        "shortlist", help="rebuild the non-authoritative signal-risk shortlist"
    )
    shortlist.add_argument("--manifest", required=True, type=Path)
    shortlist.add_argument("--count", type=int, default=3)
    shortlist.add_argument("--output", type=Path)

    decode = commands.add_parser(
        "decode", help="decode a downloaded blind-listening response to CSV"
    )
    decode.add_argument("--key", required=True, type=Path)
    decode.add_argument("--results", required=True, type=Path)
    decode.add_argument("--output", required=True, type=Path)
    return parser


def _plan(config: Path) -> int:
    spec = load_evaluation_spec(config)
    cases = spec.expand()
    seconds = sum(case.total_frames for case in cases) / FRAME_RATE
    result = {
        "case_count": len(cases),
        "generated_audio_seconds_including_warmup": seconds,
        "estimated_realtime_render_minutes": seconds / 60.0,
        "cases": [case.to_dict() for case in cases],
    }
    print(json.dumps(_safe_json(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


def _render(args: argparse.Namespace) -> int:
    spec = load_evaluation_spec(args.config)
    manifest = run_evaluation(
        spec,
        args.output,
        production_adapter_factory,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
    )
    complete = sum(case["status"] == "complete" for case in manifest["cases"])
    errors = sum(case["status"] == "error" for case in manifest["cases"])
    print(f"rendered {complete} clip(s); {errors} error(s)")
    print(f"manifest: {(args.output / 'manifest.json').resolve()}")
    if args.report == "pairwise" and complete >= 2:
        report, key = generate_pairwise_report(
            args.output / "manifest.json",
            seed=spec.report.seed,
            max_pairs=spec.report.max_pairs,
        )
        print(f"listening report: {report}")
        print(f"private blind key: {key}")
    return 1 if errors else 0


def _quality_spec(args: argparse.Namespace, *, multi: bool):
    if not 30.0 <= args.duration_seconds <= 90.0:
        raise ValueError("quality candidate duration must be between 30 and 90 seconds")
    if multi:
        if args.takes < 2:
            raise ValueError("multi-take requires at least two takes")
        if args.shortlist_count <= 0:
            raise ValueError("multi-take shortlist count must be positive")
        seeds = list(range(args.seed_start, args.seed_start + args.takes))
        report = {"seed": args.report_seed, "max_pairs": args.max_pairs}
    else:
        seeds = [args.seed]
        report = {"seed": 314159, "max_pairs": 0}
    style = {
        "id": "quality-style",
        "station": args.station,
        "melody": args.melody,
        "drums": args.drums,
    }
    for key in ("mood", "instrument", "bpm", "groove", "intensity"):
        value = getattr(args, key, None)
        if value is not None:
            style[key] = value
    if args.prompt is not None:
        style["prompt"] = args.prompt
    elif args.station == "custom":
        style["prompt"] = DEFAULT_QUALITY_PROMPT
    if args.audio_reference is not None:
        style["audio_reference"] = str(args.audio_reference)
    elif args.text_only:
        style["audio_reference"] = None

    warmup_seconds = getattr(args, "warmup_seconds", None)
    warmup_bars = getattr(args, "warmup_bars", None)
    warmup = (
        {"warmup_seconds": warmup_seconds}
        if warmup_seconds is not None
        else {"warmup_bars": 1 if warmup_bars is None else warmup_bars}
    )
    raw = {
        "schema_version": 1,
        "duration_seconds": args.duration_seconds,
        **warmup,
        "engine": {
            "model_size": args.model_size,
            "backend": "python",
            "bits": args.bits,
            "fast_sampler": True,
            "audio_style_blend": args.audio_style_blend,
        },
        "seeds": seeds,
        "composition_seed": args.composition_seed,
        "styles": [style],
        "presets": [
            {
                "id": "quality-first-starting-point",
                "temperature": args.temperature,
                "top_k": args.top_k,
                "cfg_musiccoca": args.cfg_musiccoca,
                "cfg_notes": args.cfg_notes,
                "cfg_drums": args.cfg_drums,
                "style_token_levels": args.style_token_levels,
                "sampling_mode": args.sampling_mode,
            }
        ],
        "note_modes": [
            {"id": args.note_mode, "kind": args.note_mode}
        ],
        "drum_modes": [{"id": args.drum_mode, "kind": args.drum_mode}],
        "codebook_depths": [12],
        "report": report,
    }
    return evaluation_spec_from_dict(raw)


def _quality_render(args: argparse.Namespace, *, multi: bool) -> int:
    spec = _quality_spec(args, multi=multi)
    manifest = run_evaluation(
        spec,
        args.output,
        production_adapter_factory,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
    )
    complete = sum(case["status"] == "complete" for case in manifest["cases"])
    errors = sum(case["status"] == "error" for case in manifest["cases"])
    print(f"rendered {complete} quality-first take(s) at 12 codebooks; {errors} error(s)")
    if multi:
        selection_kind = (
            "decoder-only variations with fixed composition"
            if spec.composition_seed is not None
            else "whole-take variations (decoder and composition both change)"
        )
        print(f"take mode: {selection_kind}")
    print(f"manifest: {(args.output / 'manifest.json').resolve()}")
    if multi and complete:
        shortlist = build_signal_shortlist(
            args.output / "manifest.json", count=args.shortlist_count
        )
        print(
            "non-authoritative signal shortlist: "
            f"{(args.output / 'signal_shortlist.json').resolve()} "
            f"({len(shortlist['shortlist'])} clip(s); blind listening still required)"
        )
    if multi and complete >= 2:
        report, key = generate_pairwise_report(
            args.output / "manifest.json",
            seed=spec.report.seed,
            max_pairs=spec.report.max_pairs,
            hold_seed_constant=False,
        )
        print(f"listening report: {report}")
        print(f"private blind key: {key}")
    return 1 if errors else 0


def _analyze(wavs: list[Path], output: Path | None) -> int:
    rows = []
    for path in wavs:
        samples, sample_rate = read_pcm16_wav(path)
        rows.append(
            {
                "wav": str(path),
                "metrics_disclaimer": METRICS_DISCLAIMER,
                "metrics": analyze_audio(samples, sample_rate).to_dict(),
            }
        )
    encoded = json.dumps(_safe_json(rows), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


def _report(args: argparse.Namespace) -> int:
    report, key = generate_pairwise_report(
        args.manifest,
        seed=args.seed,
        max_pairs=args.max_pairs,
        hold_seed_constant=not args.compare_across_seeds,
    )
    print(f"listening report: {report}")
    print(f"private blind key: {key}")
    return 0


def _shortlist(args: argparse.Namespace) -> int:
    result = build_signal_shortlist(
        args.manifest, count=args.count, output_path=args.output
    )
    target = args.output or args.manifest.parent / "signal_shortlist.json"
    print(f"wrote {len(result['shortlist'])} triage entries to {target.resolve()}")
    print(result["disclaimer"])
    return 0


def _decode(args: argparse.Namespace) -> int:
    rows = decode_pairwise_results(args.key, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "listener_id",
        "trial_id",
        "a_case_id",
        "b_case_id",
        "choice",
        "winner_case_id",
        "note",
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"decoded {len(rows)} response(s) to {args.output.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            return _plan(args.config)
        if args.command == "render":
            return _render(args)
        if args.command == "candidate":
            return _quality_render(args, multi=False)
        if args.command == "multi-take":
            return _quality_render(args, multi=True)
        if args.command == "analyze":
            return _analyze(args.wav, args.output)
        if args.command == "report":
            return _report(args)
        if args.command == "shortlist":
            return _shortlist(args)
        if args.command == "decode":
            return _decode(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
