"""GAN retrain with physics-informed loss (lambda_physics > 0).

Reuses existing supervised checkpoints and NEAT discriminators.
Creates NEW checkpoints with '_phy' suffix (doesn't overwrite old ones).
After running, re-run run_stacking.py to include physics-informed models.

Usage:
    python run_physics_gan.py --output_dir /path/to/experiments
"""
import argparse
import json
import re
import time
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig
from materialgen.discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from materialgen.gan_trainer import ConcreteGAN, GANConfig
from materialgen.physics import load_gost_table
from materialgen.metrics import evaluate_model
from materialgen.tracker import ExperimentTracker, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(data_dir="data", seed=42):
    ds = load_and_unify_datasets(data_dir)
    split = grouped_stratified_split(ds, seed=seed)
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
            log(f"GOST table loaded: {len(gost.grades)} grades")
        except Exception as e:
            log(f"GOST load failed: {e}")
    data["gost"] = gost
    return data


def eval_model(gen, data, mc_samples=30):
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
        "mae": ev.regression.mae, "rmse": ev.regression.rmse,
        "r2": ev.regression.r2, "mape": ev.regression.mape,
    }
    if ev.calibration:
        result["picp"] = ev.calibration.picp
        result["mpiw"] = ev.calibration.mpiw
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--lambda_physics", type=float, default=0.3)
    args = parser.parse_args()

    device = get_device()
    log(f"Device: {device}")
    log(f"lambda_physics: {args.lambda_physics}")

    data = load_data(args.data_dir)
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    tracker = ExperimentTracker(args.output_dir)

    log(f"Data: train={data['n_train']}, input_dim={data['input_dim']}")

    # ── 1. Find top-3 supervised configs ─────────────────────────────
    grid_file = checkpoint_dir / "supervised_grid.json"
    if not grid_file.exists():
        log("ERROR: supervised_grid.json not found! Run supervised_grid first.")
        return

    sup_results = json.load(open(grid_file))[:3]
    log(f"Using top-{len(sup_results)} supervised configs")

    # ── 2. GAN sweep with physics ────────────────────────────────────
    log("=" * 60)
    log(f"GAN + PHYSICS (lambda={args.lambda_physics})")
    log("=" * 60)

    gan_lrs = [1e-4, 2e-4]  # Focus on the two best LRs from previous run
    results = []

    for sup in sup_results:
        sup_tag = sup["tag"]
        hidden = sup["hidden"]
        dropout = sup["dropout"]
        bs = sup.get("batch_size", 64)

        # Load supervised base checkpoint
        sup_ckpt = checkpoint_dir / f"{sup_tag}.pt"
        if not sup_ckpt.exists():
            log(f"  [{sup_tag}] SKIP: no supervised checkpoint")
            continue

        gen_base = ConcreteGenerator(GeneratorConfig(
            input_dim=data["input_dim"], hidden_dims=hidden,
            dropout=dropout, seed=42,
        ))
        gen_base.load_state_dict(
            torch.load(sup_ckpt, map_location="cpu", weights_only=True))
        base_state = {k: v.clone() for k, v in gen_base.state_dict().items()}
        log(f"  [{sup_tag}] Loaded supervised checkpoint")

        # Load or evolve NEAT discriminator
        disc_ckpt = checkpoint_dir / f"disc_{sup_tag}.pt"
        if disc_ckpt.exists():
            disc_template = NeatBNNDiscriminator(DiscriminatorConfig(
                algorithm="bneatest", neat_generations=5, pop_size=50,
                max_eval_samples=150, svi_epochs=20, mc_samples=5, seed=42,
            ))
            disc_template.load(disc_ckpt)
            log(f"  [{sup_tag}] NEAT loaded from cache")
        else:
            log(f"  [{sup_tag}] NEAT evolving...")
            disc_template = NeatBNNDiscriminator(DiscriminatorConfig(
                algorithm="bneatest", neat_generations=5, pop_size=50,
                max_eval_samples=150, svi_epochs=20, mc_samples=5, seed=42,
            ))
            gen_probe = ConcreteGenerator(GeneratorConfig(
                input_dim=data["input_dim"], hidden_dims=hidden,
                dropout=dropout, seed=42,
            ))
            gan_probe = ConcreteGAN(gen_probe, disc_template, config=GANConfig(
                total_epochs=1, seed=42,
            ))
            neat_dir = str(checkpoint_dir / f"neat_{sup_tag}")
            gan_probe.prepare_discriminator(
                data["train"]["x"], data["train"]["y"],
                artifacts_dir=neat_dir,
            )
            disc_template.save(disc_ckpt)
            log(f"  [{sup_tag}] NEAT done")

        # Sweep GAN LRs with physics
        for gan_lr in gan_lrs:
            tag = f"gan_{sup_tag}_glr{gan_lr}_phy"

            # Skip if already trained
            save_path = checkpoint_dir / f"{tag}.pt"
            if save_path.exists():
                log(f"    [{tag}] checkpoint exists, skipping")
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hidden,
                    dropout=dropout, seed=42,
                ))
                gen.load_state_dict(
                    torch.load(save_path, map_location="cpu", weights_only=True))
                metrics = eval_model(gen, data)
                results.append({"tag": tag, **metrics})
                log(f"    [{tag}] MAE={metrics['mae']:.2f} (cached)")
                continue

            config = {
                "lr": sup["lr"], "hidden": hidden, "dropout": dropout,
                "batch_size": bs, "gan_lr": gan_lr, "gan_epochs": 1000,
                "lambda_physics": args.lambda_physics,
            }

            t0 = time.time()
            with tracker.run(tag, config=config,
                             tags=["gan", "physics"]) as run:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hidden,
                    dropout=dropout, seed=42,
                ))
                gen.load_state_dict(
                    {k: v.clone() for k, v in base_state.items()})

                disc_fresh = NeatBNNDiscriminator(DiscriminatorConfig(
                    algorithm="bneatest", neat_generations=5, pop_size=50,
                    max_eval_samples=150, svi_epochs=20, mc_samples=5, seed=42,
                ))
                disc_fresh.load(disc_ckpt)

                gan = ConcreteGAN(gen, disc_fresh, config=GANConfig(
                    total_epochs=1000, phase1_end=100, phase2_end=250,
                    generator_lr=gan_lr, generator_weight_decay=1e-4,
                    batch_size=bs, val_interval=5,
                    early_stopping_patience=80,
                    lambda_physics=args.lambda_physics,
                    lambda_mono=1.0, lambda_abrams=0.5, lambda_gost=0.3,
                    seed=42,
                ), gost=data["gost"])

                log(f"    [{tag}] Training (1000ep, lr={gan_lr}, "
                    f"λ_phy={args.lambda_physics})...")
                history = gan.train(
                    data["train"]["x"], data["train"]["y"],
                    data["val"]["x"], data["val"]["y"],
                )

                gen = gen.cpu()
                metrics = eval_model(gen, data)
                metrics["best_epoch"] = history.best_epoch
                run.log_metrics(metrics)

                torch.save(gen.state_dict(), save_path)
                run.log_artifact("model", str(save_path))

            dt = time.time() - t0
            results.append({"tag": tag, **config, **metrics})
            log(f"    [{tag}] MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}, "
                f"best_ep={metrics['best_epoch']} ({dt:.0f}s)")

    # ── 3. Summary ───────────────────────────────────────────────────
    if results:
        results.sort(key=lambda r: r["mae"])
        log("\n" + "=" * 60)
        log("PHYSICS GAN RESULTS")
        log("=" * 60)
        for i, r in enumerate(results):
            log(f"  #{i+1}: MAE={r['mae']:.2f}, R2={r['r2']:.4f} | {r['tag']}")

        with open(checkpoint_dir / "physics_gan_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    log(f"\n✅ Done! Now re-run: python run_stacking.py --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
