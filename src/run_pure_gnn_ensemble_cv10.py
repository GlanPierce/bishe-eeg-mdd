from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_static_module():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import train_gnn_dualgraph_multiband_cv10 as mod

    return mod


def load_flatreadout_module():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import train_gnn_dualgraph_multiband_flatreadout_cv10 as mod

    return mod


def load_spatiotemporal_module():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    module_path = Path("src/__pycache__/train_gnn_dualgraph_multiband_spatiotemporal_cv10.cpython-311.pyc")
    loader = importlib.machinery.SourcelessFileLoader("spatiotemporal_ensemble_mod", str(module_path))
    spec = importlib.util.spec_from_loader("spatiotemporal_ensemble_mod", loader)
    if spec is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)

    orig_load = torch.load

    def load_allow_pickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = load_allow_pickle
    mod.torch.load = load_allow_pickle
    return mod


def load_flatwin_module():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import train_gnn_dualgraph_multiband_spatiotemporal_flatwin_cv10 as mod

    return mod


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(x.strip()) for x in str(seed_text).split(",") if x.strip()]
    if not seeds:
        raise ValueError("Seed list must not be empty.")
    return seeds


def parse_optional_int_list(text: str) -> list[int]:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    return [int(x) for x in values]


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    from feature_model_utils import calc_metrics as _calc_metrics

    return _calc_metrics(y_true, prob_pos, threshold=threshold)


def aggregate_fold_results(fold_results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    from feature_model_utils import aggregate_fold_results as _aggregate_fold_results

    return _aggregate_fold_results(fold_results)


def summarize_run_payloads(run_payloads: list[dict[str, object]]) -> dict[str, object]:
    from feature_model_utils import summarize_run_payloads as _summarize_run_payloads

    return _summarize_run_payloads(run_payloads)


def write_json(path: Path, payload: dict[str, object]) -> None:
    from feature_model_utils import write_json as _write_json

    _write_json(path, payload)


def simplex_grid_3(step: float) -> list[tuple[float, float, float]]:
    step = float(step)
    n = int(round(1.0 / step))
    combos: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append((i * step, j * step, k * step))
    return combos


def select_weights_threshold(
    y_val: np.ndarray,
    prob_flat_static: np.ndarray,
    prob_st_base: np.ndarray,
    prob_flatwin: np.ndarray,
    weight_step: float,
    thr_low: float,
    thr_high: float,
    thr_step: float,
    allow_flat_static: bool = True,
    allow_st_base: bool = True,
    allow_flatwin: bool = True,
) -> tuple[tuple[float, float, float], float, float]:
    weight_grid = simplex_grid_3(step=weight_step)
    thr_grid = np.arange(float(thr_low), float(thr_high) + (float(thr_step) * 0.5), float(thr_step), dtype=np.float64)
    best_weights = (1.0, 0.0, 0.0)
    best_thr = 0.5
    best_acc = -1.0
    for w_flat_static, w_st_base, w_flatwin in weight_grid:
        if (not allow_flat_static and w_flat_static > 1e-12) or (not allow_st_base and w_st_base > 1e-12) or (not allow_flatwin and w_flatwin > 1e-12):
            continue
        prob = (w_flat_static * prob_flat_static) + (w_st_base * prob_st_base) + (w_flatwin * prob_flatwin)
        for thr in thr_grid:
            pred = (prob >= float(thr)).astype(np.int64)
            acc = float(accuracy_score(y_val, pred))
            if (acc > best_acc) or (
                abs(acc - best_acc) <= 1e-12
                and (w_st_base + w_flatwin > best_weights[1] + best_weights[2] or (abs((w_st_base + w_flatwin) - (best_weights[1] + best_weights[2])) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)))
            ):
                best_acc = acc
                best_weights = (float(w_flat_static), float(w_st_base), float(w_flatwin))
                best_thr = float(thr)
    return best_weights, best_thr, best_acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure-GNN ensemble 10-fold CV")
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs-flat-static", type=int, default=200)
    p.add_argument("--epochs-st-base", type=int, default=200)
    p.add_argument("--epochs-flatwin", type=int, default=200)
    p.add_argument("--batch-size-flat-static", type=int, default=16)
    p.add_argument("--batch-size-st-base", type=int, default=16)
    p.add_argument("--batch-size-flatwin", type=int, default=16)
    p.add_argument("--lr-flat-static", type=float, default=1e-3)
    p.add_argument("--lr-st-base", type=float, default=1e-3)
    p.add_argument("--lr-flatwin", type=float, default=3e-4)
    p.add_argument("--weight-decay-flat-static", type=float, default=1e-4)
    p.add_argument("--weight-decay-st-base", type=float, default=1e-4)
    p.add_argument("--weight-decay-flatwin", type=float, default=5e-4)
    p.add_argument("--hidden-dim-flat-static", type=int, default=16)
    p.add_argument("--dropout-flat-static", type=float, default=0.2)
    p.add_argument("--hidden-dim-st-base", type=int, default=32)
    p.add_argument("--temporal-hidden-st-base", type=int, default=32)
    p.add_argument("--dropout-st-base", type=float, default=0.2)
    p.add_argument("--st-init-seeds", type=str, default="42")
    p.add_argument("--st-select-top-k", type=int, default=1)
    p.add_argument("--hidden-dim-flatwin", type=int, default=16)
    p.add_argument("--temporal-hidden-flatwin", type=int, default=32)
    p.add_argument("--dropout-flatwin", type=float, default=0.3)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--patience-flatwin", type=int, default=40)
    p.add_argument("--min-epochs-flatwin", type=int, default=40)
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--thr-low", type=float, default=0.3)
    p.add_argument("--thr-high", type=float, default=0.7)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--disable-flat-static", action="store_true")
    p.add_argument("--disable-st-base", action="store_true")
    p.add_argument("--disable-flatwin", action="store_true")
    p.add_argument("--refit-full-train", action="store_true")
    p.add_argument("--append-adj-row-flat-static", action="store_true")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w16p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--fold-ids", type=str, default="", help="Optional comma-separated outer fold ids to run, e.g. 5,7.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/pure_gnn_strongensemble_probe"))
    p.add_argument("--experiment-name", type=str, default="pure_gnn_strongensemble_cv10")
    return p.parse_args()


def load_or_build_spatiotemporal_items(spatio_mod, args: argparse.Namespace):
    meta_expect = {
        "split_csv": str(args.split_csv),
        "max_seconds": 120,
        "window_seconds": 16.0,
        "step_seconds": 8.0,
        "max_windows": 12,
        "pcc_quantile": 0.8,
        "min_edges": 30,
        "top_k_per_node": 4,
    }
    if args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and cache_obj.get("meta") == meta_expect and "items" in cache_obj:
            print(f"Loaded cache: {args.items_cache_path}")
            return cache_obj["items"]

    build_args = SimpleNamespace(
        split_csv=args.split_csv,
        max_seconds=120,
        window_seconds=16.0,
        step_seconds=8.0,
        max_windows=12,
        items_cache_path=args.items_cache_path,
        pcc_quantile=0.8,
        min_edges=30,
        top_k_per_node=4,
    )
    items = spatio_mod.build_subject_sequences(build_args)
    args.items_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta_expect, "items": items}, args.items_cache_path)
    return items


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def train_static_one(
    static_mod,
    train_rows: list[dict[str, object]],
    val_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    cfg: dict[str, float | int],
    device: torch.device,
    seed: int,
    patience: int,
    min_epochs: int,
) -> dict[str, object]:
    train_ds = static_mod.MultiBandDataset(train_rows)
    val_ds = static_mod.MultiBandDataset(val_rows)
    test_ds = static_mod.MultiBandDataset(test_rows)
    in_dim = int(train_ds[0]["pcc"].x.shape[1])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=static_mod.collate_multiband)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=static_mod.collate_multiband)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=static_mod.collate_multiband)

    model = static_mod.DualGraphMultiBandFusionModel(in_dim=in_dim, hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    best_state = None
    best_epoch = 1
    best_val_acc = -1.0
    stale = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        _ = static_mod.train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        y_val, p_val, _ = static_mod.predict_with_weights(model, val_loader, device)
        val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= int(min_epochs) and stale >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_val, p_val, _ = static_mod.predict_with_weights(model, val_loader, device)
    y_test, p_test, _ = static_mod.predict_with_weights(model, test_loader, device)
    result = {
        "y_val": y_val,
        "p_val": p_val,
        "y_test": y_test,
        "p_test": p_test,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
    }
    if bool(cfg.get("refit_full_train", False)):
        full_ds = static_mod.MultiBandDataset(train_rows + val_rows)
        full_loader = torch.utils.data.DataLoader(full_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=static_mod.collate_multiband)
        full_model = static_mod.DualGraphMultiBandFusionModel(in_dim=in_dim, hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"])).to(device)
        full_opt = torch.optim.Adam(full_model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        for _ in range(int(best_epoch)):
            _ = static_mod.train_one_epoch(model=full_model, loader=full_loader, optimizer=full_opt, device=device)
        _, p_test_refit, _ = static_mod.predict_with_weights(full_model, test_loader, device)
        result["p_test"] = p_test_refit
    return result


def train_flatreadout_one(
    flat_mod,
    train_rows: list[dict[str, object]],
    val_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    cfg: dict[str, float | int | bool],
    device: torch.device,
    seed: int,
    patience: int,
    min_epochs: int,
) -> dict[str, object]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    train_ds = flat_mod.MultiBandDataset(train_rows, append_adj_row=bool(cfg["append_adj_row"]))
    val_ds = flat_mod.MultiBandDataset(val_rows, append_adj_row=bool(cfg["append_adj_row"]))
    test_ds = flat_mod.MultiBandDataset(test_rows, append_adj_row=bool(cfg["append_adj_row"]))
    sample_graph = train_ds[0]["pcc"]

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=flat_mod.collate_multiband)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=flat_mod.collate_multiband)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=flat_mod.collate_multiband)

    model = flat_mod.FlatReadoutStaticGNN(
        in_dim=int(sample_graph.x.shape[1]),
        n_nodes=int(sample_graph.x.shape[0]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    best_state = None
    best_epoch = 1
    best_val_acc = -1.0
    stale = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        _ = flat_mod.train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        y_val, p_val, _ = flat_mod.predict(model, val_loader, device)
        val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= int(min_epochs) and stale >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_val, p_val, _ = flat_mod.predict(model, val_loader, device)
    y_test, p_test, _ = flat_mod.predict(model, test_loader, device)
    result = {
        "y_val": y_val,
        "p_val": p_val,
        "y_test": y_test,
        "p_test": p_test,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
    }
    if bool(cfg.get("refit_full_train", False)):
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        full_ds = flat_mod.MultiBandDataset(train_rows + val_rows, append_adj_row=bool(cfg["append_adj_row"]))
        full_loader = torch.utils.data.DataLoader(full_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=flat_mod.collate_multiband)
        full_model = flat_mod.FlatReadoutStaticGNN(
            in_dim=int(sample_graph.x.shape[1]),
            n_nodes=int(sample_graph.x.shape[0]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        full_opt = torch.optim.Adam(full_model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        for _ in range(int(best_epoch)):
            _ = flat_mod.train_one_epoch(model=full_model, loader=full_loader, optimizer=full_opt, device=device)
        _, p_test_refit, _ = flat_mod.predict(full_model, test_loader, device)
        result["p_test"] = p_test_refit
    return result


def train_spatiotemporal_one(
    spatio_mod,
    train_items: list[dict[str, object]],
    val_items: list[dict[str, object]],
    test_items: list[dict[str, object]],
    cfg: dict[str, float | int],
    device: torch.device,
    seed: int,
    patience: int,
    min_epochs: int,
) -> dict[str, object]:
    train_ds = spatio_mod.TemporalDualGraphDataset(train_items)
    val_ds = spatio_mod.TemporalDualGraphDataset(val_items)
    test_ds = spatio_mod.TemporalDualGraphDataset(test_items)
    in_dim = int(train_items[0]["windows"][0]["pcc"].x.shape[1])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=spatio_mod.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=spatio_mod.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=spatio_mod.collate_fn)
    init_seeds = list(cfg.get("init_seeds", [int(seed)]))
    select_top_k = max(1, min(int(cfg.get("select_top_k", 1)), len(init_seeds)))
    candidates: list[dict[str, object]] = []

    for init_seed in init_seeds:
        np.random.seed(int(init_seed))
        torch.manual_seed(int(init_seed))
        model = spatio_mod.DualGraphMultiBandSpatioTemporalModel(
            in_dim=in_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            temporal_hidden=int(cfg["temporal_hidden"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

        best_state = None
        best_epoch = 1
        best_val_acc = -1.0
        stale = 0
        for epoch in range(1, int(cfg["epochs"]) + 1):
            _ = spatio_mod.train_one_epoch(model, train_loader, optimizer, device)
            y_val, p_val, _, _ = spatio_mod.predict(model, val_loader, device)
            val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= int(min_epochs) and stale >= int(patience):
                break

        if best_state is None:
            continue
        model.load_state_dict(best_state)
        y_val, p_val, _, _ = spatio_mod.predict(model, val_loader, device)
        y_test, p_test, _, _ = spatio_mod.predict(model, test_loader, device)
        candidates.append(
            {
                "init_seed": int(init_seed),
                "best_epoch": int(best_epoch),
                "best_val_acc": float(best_val_acc),
                "y_val": y_val,
                "p_val": p_val,
                "y_test": y_test,
                "p_test": p_test,
            }
        )

    if not candidates:
        raise RuntimeError("No spatiotemporal candidates were trained.")

    candidates.sort(key=lambda x: (float(x["best_val_acc"]), -int(x["best_epoch"])), reverse=True)
    selected = candidates[:select_top_k]
    y_val_ref = selected[0]["y_val"]
    y_test_ref = selected[0]["y_test"]
    p_val = np.mean(np.stack([x["p_val"] for x in selected], axis=0), axis=0)
    p_test = np.mean(np.stack([x["p_test"] for x in selected], axis=0), axis=0)
    return {
        "y_val": y_val_ref,
        "p_val": p_val,
        "y_test": y_test_ref,
        "p_test": p_test,
        "best_epoch": int(selected[0]["best_epoch"]),
        "best_val_acc": float(selected[0]["best_val_acc"]),
        "selected_init_seeds": [int(x["init_seed"]) for x in selected],
        "selected_best_epochs": [int(x["best_epoch"]) for x in selected],
        "selected_best_val_accuracies": [float(x["best_val_acc"]) for x in selected],
    }


def train_flatwin_one(
    flatwin_mod,
    train_items: list[dict[str, object]],
    val_items: list[dict[str, object]],
    test_items: list[dict[str, object]],
    cfg: dict[str, float | int | bool],
    device: torch.device,
    seed: int,
    patience: int,
    min_epochs: int,
) -> dict[str, object]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    train_ds = flatwin_mod.AugmentedTemporalDataset(train_items)
    val_ds = flatwin_mod.AugmentedTemporalDataset(val_items)
    test_ds = flatwin_mod.AugmentedTemporalDataset(test_items)
    sample_graph = train_ds[0]["windows"][0]["pcc"]

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=flatwin_mod.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=flatwin_mod.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=flatwin_mod.collate_fn)

    model = flatwin_mod.FlatWindowSpatioTemporalGNN(
        in_dim=int(sample_graph.x.shape[1]),
        n_nodes=int(sample_graph.x.shape[0]),
        hidden_dim=int(cfg["hidden_dim"]),
        temporal_hidden=int(cfg["temporal_hidden"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    best_state = None
    best_epoch = 1
    best_val_acc = -1.0
    stale = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        _ = flatwin_mod.train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        y_val, p_val, _ = flatwin_mod.predict(model, val_loader, device)
        val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= int(min_epochs) and stale >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_val, p_val, _ = flatwin_mod.predict(model, val_loader, device)
    y_test, p_test, _ = flatwin_mod.predict(model, test_loader, device)
    result = {
        "y_val": y_val,
        "p_val": p_val,
        "y_test": y_test,
        "p_test": p_test,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
    }
    if bool(cfg.get("refit_full_train", False)):
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        full_ds = flatwin_mod.AugmentedTemporalDataset(train_items + val_items)
        full_loader = torch.utils.data.DataLoader(full_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=flatwin_mod.collate_fn)
        full_model = flatwin_mod.FlatWindowSpatioTemporalGNN(
            in_dim=int(sample_graph.x.shape[1]),
            n_nodes=int(sample_graph.x.shape[0]),
            hidden_dim=int(cfg["hidden_dim"]),
            temporal_hidden=int(cfg["temporal_hidden"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        full_opt = torch.optim.Adam(full_model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        for _ in range(int(best_epoch)):
            _ = flatwin_mod.train_one_epoch(model=full_model, loader=full_loader, optimizer=full_opt, device=device)
        _, p_test_refit, _ = flatwin_mod.predict(full_model, test_loader, device)
        result["p_test"] = p_test_refit
    return result


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    fold_filter = set(parse_optional_int_list(args.fold_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat_mod = load_flatreadout_module()
    spatio_mod = load_spatiotemporal_module()
    flatwin_mod = load_flatwin_module()

    static_master = flat_mod.build_master_table(args.pcc_manifest, args.theta_manifest, args.alpha_manifest, args.beta_manifest)
    y_static = static_master["label"].to_numpy(dtype=np.int64)
    subjects_static = static_master["subject_id"].to_numpy()

    spatio_items = load_or_build_spatiotemporal_items(spatio_mod, args)
    subjects_spatio = np.array([str(x["subject_id"]) for x in spatio_items])
    y_spatio = np.array([int(x["label"]) for x in spatio_items], dtype=np.int64)

    subject_to_static_index = {str(sid): idx for idx, sid in enumerate(subjects_static)}
    static_rows = static_master.to_dict(orient="records")
    if set(subjects_static.tolist()) != set(subjects_spatio.tolist()):
        raise ValueError("Static and spatiotemporal subject sets do not match.")

    model_cfgs = {
        "flat_static": {
            "epochs": int(args.epochs_flat_static),
            "batch_size": int(args.batch_size_flat_static),
            "lr": float(args.lr_flat_static),
            "weight_decay": float(args.weight_decay_flat_static),
            "hidden_dim": int(args.hidden_dim_flat_static),
            "dropout": float(args.dropout_flat_static),
            "append_adj_row": bool(args.append_adj_row_flat_static),
            "refit_full_train": bool(args.refit_full_train),
        },
        "st_base": {
            "epochs": int(args.epochs_st_base),
            "batch_size": int(args.batch_size_st_base),
            "lr": float(args.lr_st_base),
            "weight_decay": float(args.weight_decay_st_base),
            "hidden_dim": int(args.hidden_dim_st_base),
            "temporal_hidden": int(args.temporal_hidden_st_base),
            "dropout": float(args.dropout_st_base),
            "refit_full_train": bool(args.refit_full_train),
            "init_seeds": parse_seed_list(args.st_init_seeds),
            "select_top_k": int(args.st_select_top_k),
        },
        "flatwin": {
            "epochs": int(args.epochs_flatwin),
            "batch_size": int(args.batch_size_flatwin),
            "lr": float(args.lr_flatwin),
            "weight_decay": float(args.weight_decay_flatwin),
            "hidden_dim": int(args.hidden_dim_flatwin),
            "temporal_hidden": int(args.temporal_hidden_flatwin),
            "dropout": float(args.dropout_flatwin),
            "refit_full_train": bool(args.refit_full_train),
        },
    }
    if args.disable_flat_static and args.disable_st_base and args.disable_flatwin:
        raise ValueError("At least one ensemble member must remain enabled.")

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        fold_results: list[dict[str, object]] = []

        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(len(spatio_items)), y_spatio), start=1):
            if fold_filter and fold_no not in fold_filter:
                continue
            trainval_items = [spatio_items[i] for i in trainval_idx]
            test_items = [spatio_items[i] for i in test_idx]
            y_trainval = y_spatio[trainval_idx]
            tr_idx, val_idx = train_test_split(
                np.arange(len(trainval_items)),
                test_size=float(args.val_size),
                random_state=seed + fold_no,
                stratify=y_trainval,
            )
            train_items = [trainval_items[i] for i in tr_idx]
            val_items = [trainval_items[i] for i in val_idx]
            test_subject_ids = [str(x["subject_id"]) for x in test_items]
            train_subject_ids = [str(x["subject_id"]) for x in train_items]
            val_subject_ids = [str(x["subject_id"]) for x in val_items]

            static_train_rows = [static_rows[subject_to_static_index[sid]] for sid in train_subject_ids]
            static_val_rows = [static_rows[subject_to_static_index[sid]] for sid in val_subject_ids]
            static_test_rows = [static_rows[subject_to_static_index[sid]] for sid in test_subject_ids]

            res_flat_static = None
            if not args.disable_flat_static:
                res_flat_static = train_flatreadout_one(
                    flat_mod=flat_mod,
                    train_rows=static_train_rows,
                    val_rows=static_val_rows,
                    test_rows=static_test_rows,
                    cfg=model_cfgs["flat_static"],
                    device=device,
                    seed=(seed * 100) + (fold_no * 10) + 1,
                    patience=args.patience,
                    min_epochs=args.min_epochs,
                )
            res_st_base = None
            if not args.disable_st_base:
                res_st_base = train_spatiotemporal_one(
                    spatio_mod=spatio_mod,
                    train_items=train_items,
                    val_items=val_items,
                    test_items=test_items,
                    cfg=model_cfgs["st_base"],
                    device=device,
                    seed=(seed * 100) + (fold_no * 10) + 2,
                    patience=args.patience,
                    min_epochs=args.min_epochs,
                )
            res_flatwin = None
            if not args.disable_flatwin:
                res_flatwin = train_flatwin_one(
                    flatwin_mod=flatwin_mod,
                    train_items=train_items,
                    val_items=val_items,
                    test_items=test_items,
                    cfg=model_cfgs["flatwin"],
                    device=device,
                    seed=(seed * 100) + (fold_no * 10) + 3,
                    patience=args.patience_flatwin,
                    min_epochs=args.min_epochs_flatwin,
                )

            ref_member = res_flat_static or res_st_base or res_flatwin
            if ref_member is None:
                raise RuntimeError("No enabled members produced predictions.")
            y_val = ref_member["y_val"]
            y_test = ref_member["y_test"]
            weights_sel, thr_sel, val_acc = select_weights_threshold(
                y_val=y_val,
                prob_flat_static=np.zeros_like(y_val, dtype=np.float64) if res_flat_static is None else res_flat_static["p_val"],
                prob_st_base=np.zeros_like(y_val, dtype=np.float64) if res_st_base is None else res_st_base["p_val"],
                prob_flatwin=np.zeros_like(y_val, dtype=np.float64) if res_flatwin is None else res_flatwin["p_val"],
                weight_step=args.weight_step,
                thr_low=args.thr_low,
                thr_high=args.thr_high,
                thr_step=args.thr_step,
                allow_flat_static=not args.disable_flat_static,
                allow_st_base=not args.disable_st_base,
                allow_flatwin=not args.disable_flatwin,
            )
            prob_test = (
                (weights_sel[0] * (np.zeros_like(y_test, dtype=np.float64) if res_flat_static is None else res_flat_static["p_test"]))
                + (weights_sel[1] * (np.zeros_like(y_test, dtype=np.float64) if res_st_base is None else res_st_base["p_test"]))
                + (weights_sel[2] * (np.zeros_like(y_test, dtype=np.float64) if res_flatwin is None else res_flatwin["p_test"]))
            )
            metrics = calc_metrics(y_test, prob_test, threshold=thr_sel)
            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_items)),
                    "n_val": int(len(val_items)),
                    "n_test": int(len(test_items)),
                    "test_subject_ids": test_subject_ids,
                    "ensemble_weights": {
                        "flat_static": float(weights_sel[0]),
                        "st_base": float(weights_sel[1]),
                        "flatwin": float(weights_sel[2]),
                    },
                    "ensemble_threshold": float(thr_sel),
                    "ensemble_val_accuracy": float(val_acc),
                    "members": {
                        "flat_static": None
                        if res_flat_static is None
                        else {
                            "best_epoch": int(res_flat_static["best_epoch"]),
                            "best_val_accuracy": float(res_flat_static["best_val_acc"]),
                        },
                        "st_base": None
                        if res_st_base is None
                        else {
                            "best_epoch": int(res_st_base["best_epoch"]),
                            "best_val_accuracy": float(res_st_base["best_val_acc"]),
                            "selected_init_seeds": [int(x) for x in res_st_base.get("selected_init_seeds", [])],
                            "selected_best_epochs": [int(x) for x in res_st_base.get("selected_best_epochs", [])],
                            "selected_best_val_accuracies": [float(x) for x in res_st_base.get("selected_best_val_accuracies", [])],
                        },
                        "flatwin": None
                        if res_flatwin is None
                        else {
                            "best_epoch": int(res_flatwin["best_epoch"]),
                            "best_val_accuracy": float(res_flatwin["best_val_acc"]),
                        },
                    },
                    "test_metrics": metrics,
                }
            )
            print(
                f"seed={seed} fold={fold_no:02d} "
                f"acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
                f"w=[{weights_sel[0]:.2f},{weights_sel[1]:.2f},{weights_sel[2]:.2f}] thr={thr_sel:.2f}"
            )

        if not fold_results:
            raise RuntimeError(f"No folds were executed for seed={seed}; check --fold-ids.")

        payload = {
            "schema_version": "eeg_mdd_pure_gnn_ensemble_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "PureGNNEnsemble",
            "params": {**serialize_args(args), "seed": int(seed), "seeds": seeds},
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"pure_gnn_ensemble_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_pure_gnn_ensemble_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "PureGNNEnsemble",
        "params": serialize_args(args),
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
