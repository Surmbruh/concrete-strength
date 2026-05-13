from __future__ import annotations

import argparse
import json
from pathlib import Path

from .make_neat_to_bnn import run_make_neat_to_bnn
from .train_neat import run_train_neat


def _write_payload(payload: str, output_path: str | None) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialgen",
        description="Concrete mix design: inverse NEAT training, BNN fine-tuning, and forward GAN prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── Existing inverse pipeline ────────────────────────────────────
    neat_parser = subparsers.add_parser(
        "train_neat",
        help="Train the inverse NEAT network and save it under artifacts/train_neat",
    )
    neat_parser.add_argument("--config", required=True, help="Path to backward.json")
    neat_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    neat_parser.add_argument("--inverse-dir", default=None, help="Optional override for the train_neat artifacts folder")
    neat_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    bnn_parser = subparsers.add_parser(
        "make_neat_to_bnn",
        help="Convert trained NEAT network into a Bayesian NN and fine-tune on known data",
    )
    bnn_parser.add_argument("--config", required=True, help="Path to make_neat_to_bnn.json")
    bnn_parser.add_argument("--artifacts-dir", default="artifacts", help="Root directory for all stage artifacts")
    bnn_parser.add_argument("--inverse-dir", default=None, help="Optional override for the train_neat artifacts folder")
    bnn_parser.add_argument("--bnn-dir", default=None, help="Optional override for the make_neat_to_bnn artifacts folder")
    bnn_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    # ─── New forward pipeline ─────────────────────────────────────────
    surrogate_parser = subparsers.add_parser(
        "train_surrogate",
        help="Train forward GAN model: composition + time → strength prediction",
    )
    surrogate_parser.add_argument("--config", required=True, help="Path to forward.json config")
    surrogate_parser.add_argument("--data-dir", default="data", help="Directory containing CSV datasets")
    surrogate_parser.add_argument("--artifacts-dir", default="artifacts", help="Root output directory")
    surrogate_parser.add_argument("--output", default=None, help="Optional path for JSON summary")

    batch_parser = subparsers.add_parser(
        "batch-predict",
        help="Batch prediction of strength for multiple compositions",
    )
    batch_parser.add_argument("--config", required=True, help="Path to batch.json config")
    batch_parser.add_argument("--model-dir", required=True, help="Directory with trained model checkpoint")
    batch_parser.add_argument("--output", default=None, help="Path for JSON output with predictions")

    check_parser = subparsers.add_parser(
        "complex-check",
        help="Comprehensive validation: metrics + GOST compliance + uncertainty calibration",
    )
    check_parser.add_argument("--config", required=True, help="Path to complex_check.json config")
    check_parser.add_argument("--model-dir", required=True, help="Directory with trained model checkpoint")
    check_parser.add_argument("--data-dir", default="data", help="Directory containing CSV datasets")
    check_parser.add_argument("--output", default=None, help="Path for JSON validation report")

    return parser


# ─── Handlers ─────────────────────────────────────────────────────────

def _handle_train_neat(args) -> int:
    summary = run_train_neat(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        inverse_dir=args.inverse_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_make_neat_to_bnn(args) -> int:
    summary = run_make_neat_to_bnn(
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        inverse_dir=args.inverse_dir,
        bnn_dir=args.bnn_dir,
    )
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_train_surrogate(args) -> int:
    """Обучить прямую GAN-модель: состав → прочность."""
    from .transfer import TransferLearner, TransferConfig

    with open(args.config, encoding="utf-8") as f:
        config_payload = json.load(f)

    transfer_config = TransferConfig.from_dict(config_payload.get("transfer", {}))
    transfer_config.data_dir = args.data_dir

    learner = TransferLearner(config=transfer_config)
    learner.load_data()

    # Этап 1: Pre-training
    pretrain_summary = learner.pretrain(
        artifacts_dir=str(Path(args.artifacts_dir) / "pretrain"),
    )

    # Этап 2: Fine-tuning (GAN)
    finetune_history = learner.finetune(
        artifacts_dir=str(Path(args.artifacts_dir) / "finetune"),
    )

    # Этап 3: Evaluation
    evaluation = learner.evaluate()

    summary = {
        "pretrain": pretrain_summary,
        "finetune": {
            "best_epoch": finetune_history.best_epoch,
            "best_val_mae": finetune_history.best_val_mae,
        },
        "evaluation": evaluation.to_dict(),
    }
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_batch_predict(args) -> int:
    """Пакетное предсказание прочности."""
    from .generator import ConcreteGenerator, GeneratorConfig
    from .uncertainty import UncertaintyEstimator

    with open(args.config, encoding="utf-8") as f:
        config_payload = json.load(f)

    model_dir = Path(args.model_dir)

    # Загружаем генератор
    gen_config = GeneratorConfig.from_dict(config_payload.get("model", {}))
    generator = ConcreteGenerator(gen_config)

    gen_path = model_dir / "generator.pt"
    if gen_path.exists():
        generator.load_state_dict(
            torch.load(gen_path, map_location="cpu", weights_only=True),
        )

    estimator = UncertaintyEstimator(generator)

    # Загружаем входные данные
    import numpy as np
    compositions = np.array(config_payload.get("compositions", []), dtype=float)
    times = config_payload.get("times", [28])

    results = []
    for t in times:
        log_t = np.full((compositions.shape[0], 1), np.log(t))
        x = np.concatenate([compositions, log_t], axis=1)
        prediction = estimator.predict(x)
        results.append({
            "time_days": t,
            "predictions": prediction.to_dict(),
        })

    summary = {"predictions": results}
    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


def _handle_complex_check(args) -> int:
    """Комплексная валидация модели."""
    from .generator import ConcreteGenerator, GeneratorConfig
    from .discriminator import NeatBNNDiscriminator
    from .data_preparation import load_and_unify_datasets, stratified_split
    from .metrics import evaluate_model
    from .physics import load_gost_table

    with open(args.config, encoding="utf-8") as f:
        config_payload = json.load(f)

    model_dir = Path(args.model_dir)

    # Загружаем модель
    gen_config = GeneratorConfig.from_dict(config_payload.get("model", {}))
    generator = ConcreteGenerator(gen_config)
    gen_path = model_dir / "generator.pt"
    if gen_path.exists():
        generator.load_state_dict(
            torch.load(gen_path, map_location="cpu", weights_only=True),
        )

    # Загружаем данные
    import numpy as np
    dataset = load_and_unify_datasets(args.data_dir, include_synthetic=True)
    split = grouped_stratified_split(dataset)

    x_test = dataset.all_features[split["test"]]
    y_test = dataset.target.to_numpy()[split["test"]]
    age_test = dataset.age_days.to_numpy()[split["test"]]

    mu, sigma = generator.predict(x_test, mc_samples=30)
    evaluation = evaluate_model(
        y_true=y_test,
        y_pred=mu.ravel(),
        y_std=sigma.ravel(),
        age_days=age_test,
    )

    summary = {"evaluation": evaluation.to_dict()}

    # ГОСТ compliance
    gost_path = Path(args.data_dir) / "ГОСТы.csv"
    if gost_path.exists():
        gost = load_gost_table(gost_path)
        summary["gost_bounds"] = gost.to_dict()

    _write_payload(json.dumps(summary, ensure_ascii=False, indent=2), args.output)
    return 0


# ─── Main ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # Lazy import for batch-predict/complex-check torch usage
    global torch
    import torch  # noqa: F811

    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "train_neat": _handle_train_neat,
        "make_neat_to_bnn": _handle_make_neat_to_bnn,
        "train_surrogate": _handle_train_surrogate,
        "batch-predict": _handle_batch_predict,
        "complex-check": _handle_complex_check,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(args)
