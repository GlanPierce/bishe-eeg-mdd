from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def utc_ts_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def run_cmd(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified benchmark entry for EEG/MDD experiments.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("outputs/metrics/runs"))
    p.add_argument("--skip-nongnn", action="store_true")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--skip-multiband", action="store_true")
    return p.parse_args()


def extract_agg(path: Path) -> dict[str, object]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["aggregate"]


def to_std_record(
    run_id: str,
    name: str,
    category: str,
    source_file: Path,
    aggregate: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "eeg_mdd_benchmark_v1",
        "run_id": run_id,
        "name": name,
        "category": category,
        "source_file": str(source_file),
        "accuracy_mean": aggregate["accuracy"]["mean"],
        "accuracy_std": aggregate["accuracy"]["std"],
        "balanced_accuracy_mean": aggregate["balanced_accuracy"]["mean"],
        "balanced_accuracy_std": aggregate["balanced_accuracy"]["std"],
        "f1_mean": aggregate["f1"]["mean"],
        "f1_std": aggregate["f1"]["std"],
        "roc_auc_mean": aggregate["roc_auc"]["mean"],
        "roc_auc_std": aggregate["roc_auc"]["std"],
    }


def main() -> int:
    args = parse_args()
    run_id = args.run_id if args.run_id else f"benchmark_{utc_ts_compact()}_seed{args.seed}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    records: list[dict[str, object]] = []

    # Non-GNN
    if not args.skip_nongnn:
        out_nongnn = run_dir / "nongnn_cv10.json"
        run_cmd(
            [
                py,
                "src/train_nongnn_cv10.py",
                "--seed",
                str(args.seed),
                "--folds",
                str(args.folds),
                "--out-path",
                str(out_nongnn),
                "--experiment-name",
                "nongnn_cv10",
            ]
        )
        d = json.loads(out_nongnn.read_text(encoding="utf-8"))
        for model_name in ["logistic_regression", "random_forest"]:
            records.append(
                to_std_record(
                    run_id=run_id,
                    name=f"nongnn_{model_name}",
                    category="nongnn",
                    source_file=out_nongnn,
                    aggregate=d["models"][model_name]["aggregate"],
                )
            )

    # Single-graph GNN family
    if not args.skip_gnn:
        gnn_jobs = [
            ("gnn_signed_pcc", "data/processed/graphs_pcc_task/manifest.csv"),
            ("gnn_signed_pcc_plv_broad", "data/processed/graphs_pcc_plv_task/manifest.csv"),
            ("gnn_signed_pcc_plv_theta", "data/processed/graphs_pcc_plv_theta_task/manifest.csv"),
            ("gnn_signed_pcc_plv_alpha", "data/processed/graphs_pcc_plv_alpha_task/manifest.csv"),
            ("gnn_signed_pcc_plv_beta", "data/processed/graphs_pcc_plv_beta_task/manifest.csv"),
        ]
        for name, manifest in gnn_jobs:
            out_path = run_dir / f"{name}.json"
            run_cmd(
                [
                    py,
                    "src/train_gnn_cv10.py",
                    "--manifest",
                    manifest,
                    "--seed",
                    str(args.seed),
                    "--folds",
                    str(args.folds),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--weight-decay",
                    str(args.weight_decay),
                    "--hidden-dim",
                    str(args.hidden_dim),
                    "--dropout",
                    str(args.dropout),
                    "--out-path",
                    str(out_path),
                    "--experiment-name",
                    name,
                ]
            )
            records.append(
                to_std_record(
                    run_id=run_id,
                    name=name,
                    category="gnn_single_graph",
                    source_file=out_path,
                    aggregate=extract_agg(out_path),
                )
            )

    # Multiband fusion
    if not args.skip_multiband:
        mb_fixed = run_dir / "gnn_multiband_dynamic_weight_fixed_thr.json"
        run_cmd(
            [
                py,
                "src/train_gnn_multiband_cv10.py",
                "--seed",
                str(args.seed),
                "--folds",
                str(args.folds),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--hidden-dim",
                str(args.hidden_dim),
                "--dropout",
                str(args.dropout),
                "--threshold-strategy",
                "fixed",
                "--out-path",
                str(mb_fixed),
            ]
        )
        records.append(
            to_std_record(
                run_id=run_id,
                name="gnn_multiband_dynamic_weight_fixed_thr",
                category="gnn_multiband",
                source_file=mb_fixed,
                aggregate=extract_agg(mb_fixed),
            )
        )

        mb_dyn = run_dir / "gnn_multiband_dynamic_weight_dynamic_thr.json"
        run_cmd(
            [
                py,
                "src/train_gnn_multiband_cv10.py",
                "--seed",
                str(args.seed),
                "--folds",
                str(args.folds),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--hidden-dim",
                str(args.hidden_dim),
                "--dropout",
                str(args.dropout),
                "--threshold-strategy",
                "val_balacc",
                "--out-path",
                str(mb_dyn),
            ]
        )
        records.append(
            to_std_record(
                run_id=run_id,
                name="gnn_multiband_dynamic_weight_dynamic_thr",
                category="gnn_multiband",
                source_file=mb_dyn,
                aggregate=extract_agg(mb_dyn),
            )
        )

    records.sort(key=lambda x: x["name"])
    summary = {
        "schema_version": "eeg_mdd_benchmark_v1",
        "run_id": run_id,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "seed": args.seed,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "records": records,
    }
    summary_json = run_dir / "benchmark_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Benchmark Run: {run_id}",
        "",
        "| Name | Category | Acc | BalAcc | F1 | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in records:
        md_lines.append(
            "| {name} | {category} | {acc:.4f} ± {accs:.4f} | {bal:.4f} ± {bals:.4f} | {f1:.4f} ± {f1s:.4f} | {auc:.4f} ± {aucs:.4f} |".format(
                name=r["name"],
                category=r["category"],
                acc=r["accuracy_mean"],
                accs=r["accuracy_std"],
                bal=r["balanced_accuracy_mean"],
                bals=r["balanced_accuracy_std"],
                f1=r["f1_mean"],
                f1s=r["f1_std"],
                auc=r["roc_auc_mean"],
                aucs=r["roc_auc_std"],
            )
        )
    summary_md = run_dir / "benchmark_summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    index_path = args.output_root / "benchmark_runs_index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "summary_json": str(summary_json)}, ensure_ascii=False) + "\n")

    print("\nSaved unified outputs:")
    print("-", summary_json)
    print("-", summary_md)
    print("-", index_path)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
