"""Unified experiment runner for both local and Colab.

Usage:
    python run_experiment.py --mode supervised_grid
    python run_experiment.py --mode gan_tune --top_k 3
    python run_experiment.py --mode full_pipeline --epochs 500

Automatically detects GPU, logs to tracker, saves checkpoints.
"""
import argparse
import json
import itertools
import time
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from materialgen.gan_trainer import ConcreteGAN, GANConfig
from materialgen.physics import load_gost_table
from materialgen.metrics import evaluate_model
from materialgen.tracker import ExperimentTracker, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(data_dir="data", seed=42):
    """Load and prepare all data splits."""
    ds = load_and_unify_datasets(data_dir)
    split = stratified_split(ds, seed=seed)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    ages_all = ds.age_days.to_numpy()

    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))

    data = {}
    for key in ["train", "val", "test"]:
        idx = split[key]
        data[key] = {
            "x": feat_scaler.transform(x_all[idx]),
            "y": tgt_scaler.transform(y_all[idx].reshape(-1, 1)).ravel(),
            "ages": ages_all[idx],
        }
    data["input_dim"] = data["train"]["x"].shape[1]
    data["feat_scaler"] = feat_scaler
    data["tgt_scaler"] = tgt_scaler
    data["n_train"] = len(data["train"]["x"])

    gost = None
    gost_path = Path(data_dir) / "ГОСТы.csv"
    if gost_path.exists():
        try:
            gost = load_gost_table(gost_path)
        except Exception:
            pass
    data["gost"] = gost
    return data


def eval_model(gen, data, mc_samples=30):
    """Evaluate generator on test set, return dict of metrics."""
    x_test = data["test"]["x"]
    y_test = data["test"]["y"]
    ages = data["test"]["ages"]
    tgt_scaler = data["tgt_scaler"]

    mu_s, sigma_s = gen.predict(x_test, mc_samples=mc_samples)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    sigma = sigma_s.ravel() * tgt_scaler.scale[0]
    y_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    ev = evaluate_model(y_true=y_orig, y_pred=mu, y_std=sigma, age_days=ages)
    result = {
        "mae": ev.regression.mae,
        "rmse": ev.regression.rmse,
        "r2": ev.regression.r2,
        "mape": ev.regression.mape,
    }
    if ev.calibration:
        result["picp"] = ev.calibration.picp
        result["mpiw"] = ev.calibration.mpiw
    return result


# ====================================================================
# EXPERIMENT MODES
# ====================================================================

def run_supervised_grid(data, tracker, checkpoint_dir):
    """Grid search over supervised generator configs.

    Optimized: 18 configs (3 lr × 3 arch × 2 dropout), 200 epochs w/ early stopping.
    """
    log("=" * 60)
    log("SUPERVISED HYPERPARAMETER GRID SEARCH")
    log("=" * 60)

    search = {
        "lr": [1e-3, 5e-4, 2e-4],
        "hidden": [[256, 128, 64], [128, 64, 32], [256, 128, 64, 32]],
        "dropout": [0.1, 0.2],
    }
    batch_size = 64  # fixed — not worth tuning for this dataset size

    results = []
    total = 1
    for v in search.values():
        total *= len(v)
    log(f"Total configs: {total} (batch_size={batch_size} fixed)")
    log(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    for i, (lr, hidden, dropout) in enumerate(itertools.product(
        search["lr"], search["hidden"], search["dropout"]
    )):
        tag = f"sup_lr{lr}_h{'x'.join(map(str,hidden))}_d{dropout}"
        config = {
            "lr": lr, "hidden": hidden, "dropout": dropout,
            "batch_size": batch_size, "epochs": 200,
        }

        t0 = time.time()
        with tracker.run(tag, config=config, tags=["supervised", "grid"]) as run:
            gen_cfg = GeneratorConfig(
                input_dim=data["input_dim"], hidden_dims=hidden,
                epochs=200, batch_size=batch_size, learning_rate=lr,
                weight_decay=1e-4, dropout=dropout, seed=42,
            )
            gen = ConcreteGenerator(gen_cfg)
            hist = train_generator_supervised(
                gen, data["train"]["x"], data["train"]["y"],
                data["val"]["x"], data["val"]["y"], config=gen_cfg,
            )
            # Move to CPU for evaluation consistency
            gen = gen.cpu()
            metrics = eval_model(gen, data)
            metrics["epochs_run"] = hist["epochs_run"]
            metrics["val_loss"] = hist["best_val_loss"]
            run.log_metrics(metrics)

            # Save model checkpoint
            save_path = checkpoint_dir / f"{tag}.pt"
            torch.save(gen.state_dict(), save_path)
            run.log_artifact("model", str(save_path))

        dt = time.time() - t0
        results.append({"tag": tag, **config, **metrics})
        log(f"  [{i+1}/{total}] {tag}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f} ({dt:.0f}s, {metrics['epochs_run']}ep)")

    results.sort(key=lambda r: r["mae"])
    log("\nTOP 5:")
    for i, r in enumerate(results[:5]):
        log(f"  #{i+1}: MAE={r['mae']:.2f}, R2={r['r2']:.4f} | {r['tag']}")

    with open(checkpoint_dir / "supervised_grid.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_gan_tune(data, tracker, checkpoint_dir, top_k=3):
    """Tune GAN on top-K supervised configs."""
    log("=" * 60)
    log(f"GAN TUNING (top-{top_k} supervised configs)")
    log("=" * 60)

    # Load supervised results
    grid_file = checkpoint_dir / "supervised_grid.json"
    if not grid_file.exists():
        log("ERROR: Run supervised_grid first!")
        return []

    sup_results = json.load(open(grid_file))[:top_k]
    gan_lrs = [2e-4, 1e-4, 5e-5]
    results = []

    for sup in sup_results:
        # Re-train supervised base
        gen_cfg = GeneratorConfig(
            input_dim=data["input_dim"], hidden_dims=sup["hidden"],
            epochs=500, batch_size=sup["batch_size"], learning_rate=sup["lr"],
            weight_decay=1e-4, dropout=sup["dropout"], seed=42,
        )
        gen_base = ConcreteGenerator(gen_cfg)
        train_generator_supervised(
            gen_base, data["train"]["x"], data["train"]["y"],
            data["val"]["x"], data["val"]["y"], config=gen_cfg,
        )
        base_state = {k: v.clone() for k, v in gen_base.state_dict().items()}

        for gan_lr in gan_lrs:
            tag = f"gan_{sup['tag']}_glr{gan_lr}"
            config = {**sup, "gan_lr": gan_lr, "gan_epochs": 300,
                      "neat_pop": 50, "neat_gen": 5}

            with tracker.run(tag, config=config, tags=["gan", "tune"]) as run:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=sup["hidden"],
                    dropout=sup["dropout"], seed=42,
                ))
                gen.load_state_dict({k: v.clone() for k, v in base_state.items()})

                disc = NeatBNNDiscriminator(DiscriminatorConfig(
                    algorithm="bneatest", neat_generations=5, pop_size=50,
                    max_eval_samples=150, svi_epochs=20, mc_samples=5, seed=42,
                ))
                gan = ConcreteGAN(gen, disc, config=GANConfig(
                    total_epochs=300, phase1_end=60, phase2_end=150,
                    generator_lr=gan_lr, generator_weight_decay=1e-4,
                    batch_size=sup["batch_size"], val_interval=5,
                    early_stopping_patience=60, lambda_physics=0.0, seed=42,
                ))
                log(f"  [{tag}] NEAT evolving...")
                gan.prepare_discriminator(
                    data["train"]["x"], data["train"]["y"],
                    artifacts_dir=str(checkpoint_dir / tag),
                )
                log(f"  [{tag}] GAN training...")
                history = gan.train(
                    data["train"]["x"], data["train"]["y"],
                    data["val"]["x"], data["val"]["y"],
                )

                metrics = eval_model(gen, data)
                metrics["best_epoch"] = history.best_epoch
                run.log_metrics(metrics)

                save_path = checkpoint_dir / f"{tag}.pt"
                torch.save(gen.state_dict(), save_path)
                run.log_artifact("model", str(save_path))

            results.append({"tag": tag, **config, **metrics})
            log(f"  {tag}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")

    results.sort(key=lambda r: r["mae"])
    log("\nTOP 5 GAN:")
    for i, r in enumerate(results[:5]):
        log(f"  #{i+1}: MAE={r['mae']:.2f}, R2={r['r2']:.4f} | {r['tag']}")

    with open(checkpoint_dir / "gan_tune.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_full_pipeline(data, tracker, checkpoint_dir, epochs=500):
    """Train best config end-to-end with more epochs."""
    log("=" * 60)
    log(f"FULL PIPELINE (best config, {epochs} epochs)")
    log("=" * 60)

    # Use best known config
    config = {
        "lr": 1e-3, "hidden": [256, 128, 64], "dropout": 0.1,
        "batch_size": 64, "epochs": epochs,
        "gan_lr": 1e-4, "gan_epochs": epochs,
        "neat_pop": 80, "neat_gen": 8,
    }

    with tracker.run("full_best", config=config, tags=["full", "best"]) as run:
        # Supervised pre-train
        gen = ConcreteGenerator(GeneratorConfig(
            input_dim=data["input_dim"], hidden_dims=config["hidden"],
            epochs=config["epochs"], batch_size=config["batch_size"],
            learning_rate=config["lr"], weight_decay=1e-4,
            dropout=config["dropout"], seed=42,
        ))
        log("Supervised pre-training...")
        train_generator_supervised(
            gen, data["train"]["x"], data["train"]["y"],
            data["val"]["x"], data["val"]["y"], config=gen.config,
        )
        sup_metrics = eval_model(gen, data)
        log(f"Supervised: MAE={sup_metrics['mae']:.2f}, R2={sup_metrics['r2']:.4f}")

        # GAN fine-tune
        disc = NeatBNNDiscriminator(DiscriminatorConfig(
            algorithm="bneatest", neat_generations=config["neat_gen"],
            pop_size=config["neat_pop"], max_eval_samples=200,
            svi_epochs=30, mc_samples=10, seed=42,
        ))
        gan = ConcreteGAN(gen, disc, config=GANConfig(
            total_epochs=config["gan_epochs"],
            phase1_end=int(config["gan_epochs"] * 0.2),
            phase2_end=int(config["gan_epochs"] * 0.5),
            generator_lr=config["gan_lr"], generator_weight_decay=1e-4,
            batch_size=config["batch_size"], val_interval=5,
            early_stopping_patience=80, lambda_physics=0.0, seed=42,
        ))
        log("NEAT evolving...")
        gan.prepare_discriminator(
            data["train"]["x"], data["train"]["y"],
            artifacts_dir=str(checkpoint_dir / "full_best"),
        )
        log("GAN training...")
        history = gan.train(
            data["train"]["x"], data["train"]["y"],
            data["val"]["x"], data["val"]["y"],
        )

        metrics = eval_model(gen, data, mc_samples=50)
        metrics["sup_mae"] = sup_metrics["mae"]
        metrics["best_epoch"] = history.best_epoch
        run.log_metrics(metrics)

        save_path = checkpoint_dir / "best_model.pt"
        torch.save(gen.state_dict(), save_path)
        run.log_artifact("best_model", str(save_path))
        gan.save(str(checkpoint_dir / "full_best"))

    log(f"\nFINAL: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")
    return metrics


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Concrete Strength Experiments")
    parser.add_argument("--mode", required=True,
                        choices=["supervised_grid", "gan_tune", "full_pipeline",
                                 "smoke_test"],
                        help="Experiment mode")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--output_dir", default="experiments", help="Output directory")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K for GAN tuning")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for full pipeline")
    args = parser.parse_args()

    device = get_device()
    log(f"Device: {device}")

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(args.output_dir)

    log("Loading data...")
    data = load_data(args.data_dir)
    log(f"Data: train={data['n_train']}, input_dim={data['input_dim']}")

    if args.mode == "smoke_test":
        log("Smoke test: training 1 supervised model (10 epochs)")
        gen = ConcreteGenerator(GeneratorConfig(
            input_dim=data["input_dim"], hidden_dims=[64, 32],
            epochs=10, batch_size=64, learning_rate=1e-3, seed=42,
        ))
        train_generator_supervised(
            gen, data["train"]["x"], data["train"]["y"],
            data["val"]["x"], data["val"]["y"], config=gen.config,
        )
        m = eval_model(gen, data, mc_samples=5)
        log(f"Smoke test OK: MAE={m['mae']:.2f}")
        return

    elif args.mode == "supervised_grid":
        run_supervised_grid(data, tracker, checkpoint_dir)

    elif args.mode == "gan_tune":
        run_gan_tune(data, tracker, checkpoint_dir, top_k=args.top_k)

    elif args.mode == "full_pipeline":
        run_full_pipeline(data, tracker, checkpoint_dir, epochs=args.epochs)

    log("\nExperiment log:")
    log(tracker.summary_table())


if __name__ == "__main__":
    main()
