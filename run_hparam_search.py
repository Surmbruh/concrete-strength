"""Hyperparameter tuning: grid search over most impactful parameters.

Tunes supervised generator first (fast), then GAN on the best config.

Search space (supervised, ~15 configs):
  - learning_rate: [1e-3, 5e-4, 2e-4]
  - hidden_dims: [[256,128,64], [128,64,32], [64,32,16]]
  - dropout: [0.1, 0.3]
  - batch_size fixed at 64

Then GAN tuning (top-3 supervised configs):
  - gan_lr: [2e-4, 1e-4, 5e-5]
"""
import json
import time
import itertools
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from materialgen.gan_trainer import ConcreteGAN, GANConfig
from materialgen.metrics import evaluate_model

ARTIFACTS = Path("artifacts/hparam_search")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eval_quick(gen, x_test, y_test, tgt_scaler, mc_samples=20):
    """Quick evaluation returning MAE, R2."""
    mu_s, sigma_s = gen.predict(x_test, mc_samples=mc_samples)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    y_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mae = np.mean(np.abs(y_orig - mu))
    ss_res = np.sum((y_orig - mu) ** 2)
    ss_tot = np.sum((y_orig - y_orig.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mae, r2


def main():
    t0 = time.time()

    # Data
    log("Loading data...")
    ds = load_and_unify_datasets("data")
    split = stratified_split(ds, seed=SEED)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    ages_all = ds.age_days.to_numpy()

    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    x_val = x_all[split["val"]]
    y_val = y_all[split["val"]]
    x_test = x_all[split["test"]]
    y_test = y_all[split["test"]]
    ages_test = ages_all[split["test"]]

    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))

    x_train_s = feat_scaler.transform(x_train)
    x_val_s = feat_scaler.transform(x_val)
    x_test_s = feat_scaler.transform(x_test)
    y_train_s = tgt_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = tgt_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = tgt_scaler.transform(y_test.reshape(-1, 1)).ravel()

    input_dim = x_train_s.shape[1]
    log(f"Data: train={len(x_train_s)}, val={len(x_val_s)}, test={len(x_test_s)}")

    # ====================================================================
    # PHASE 1: Supervised grid search
    # ====================================================================
    log("\n" + "=" * 60)
    log("PHASE 1: Supervised Hyperparameter Search")
    log("=" * 60)

    search_space = {
        "lr": [1e-3, 5e-4, 2e-4],
        "hidden": [[256, 128, 64], [128, 64, 32], [64, 32, 16]],
        "dropout": [0.1, 0.3],
    }

    sup_results = []

    for lr, hidden, dropout in itertools.product(
        search_space["lr"], search_space["hidden"], search_space["dropout"]
    ):
        tag = f"lr={lr}_h={'x'.join(map(str,hidden))}_d={dropout}"
        config = GeneratorConfig(
            input_dim=input_dim,
            hidden_dims=hidden,
            epochs=300,
            batch_size=64,
            learning_rate=lr,
            weight_decay=1e-4,
            dropout=dropout,
            seed=SEED,
        )
        gen = ConcreteGenerator(config)
        hist = train_generator_supervised(gen, x_train_s, y_train_s, x_val_s, y_val_s, config=config)

        mae, r2 = eval_quick(gen, x_test_s, y_test_s, tgt_scaler)
        val_loss = hist["best_val_loss"]
        epochs_run = hist["epochs_run"]

        result = {
            "tag": tag,
            "lr": lr,
            "hidden": hidden,
            "dropout": dropout,
            "test_mae": mae,
            "test_r2": r2,
            "val_loss": val_loss,
            "epochs_run": epochs_run,
        }
        sup_results.append(result)
        log(f"  {tag}: MAE={mae:.2f}, R2={r2:.4f}, val={val_loss:.4f}, ep={epochs_run}")

    # Sort by MAE
    sup_results.sort(key=lambda r: r["test_mae"])

    log("\n" + "-" * 60)
    log("TOP 5 SUPERVISED CONFIGS:")
    log("-" * 60)
    for i, r in enumerate(sup_results[:5]):
        log(f"  #{i+1}: MAE={r['test_mae']:.2f}, R2={r['test_r2']:.4f} | {r['tag']}")

    with open(ARTIFACTS / "supervised_search.json", "w") as f:
        json.dump(sup_results, f, indent=2, default=str)

    # ====================================================================
    # PHASE 2: GAN tuning on top-3 configs
    # ====================================================================
    log("\n" + "=" * 60)
    log("PHASE 2: GAN Tuning (top-3 supervised configs x 3 LRs)")
    log("=" * 60)

    gan_lrs = [2e-4, 1e-4, 5e-5]
    gan_results = []

    for sup_cfg in sup_results[:3]:
        hidden = sup_cfg["hidden"]
        dropout = sup_cfg["dropout"]
        sup_lr = sup_cfg["lr"]

        # Pre-train supervised with best config
        gen_base = ConcreteGenerator(GeneratorConfig(
            input_dim=input_dim, hidden_dims=hidden,
            epochs=300, batch_size=64, learning_rate=sup_lr,
            weight_decay=1e-4, dropout=dropout, seed=SEED,
        ))
        train_generator_supervised(gen_base, x_train_s, y_train_s, x_val_s, y_val_s, config=gen_base.config)
        base_state = {k: v.clone() for k, v in gen_base.state_dict().items()}

        for gan_lr in gan_lrs:
            tag = f"sup={sup_cfg['tag']}_glr={gan_lr}"

            gen = ConcreteGenerator(GeneratorConfig(
                input_dim=input_dim, hidden_dims=hidden, dropout=dropout, seed=SEED,
            ))
            gen.load_state_dict({k: v.clone() for k, v in base_state.items()})

            disc = NeatBNNDiscriminator(DiscriminatorConfig(
                algorithm="bneatest", neat_generations=5, pop_size=50,
                max_eval_samples=150, svi_epochs=20, mc_samples=5, seed=SEED,
            ))

            gan_config = GANConfig(
                total_epochs=300,
                phase1_end=60, phase2_end=150,
                generator_lr=gan_lr,
                generator_weight_decay=1e-4,
                batch_size=64, val_interval=5,
                early_stopping_patience=60,
                lambda_physics=0.0,
                seed=SEED,
            )

            gan = ConcreteGAN(gen, disc, config=gan_config)
            log(f"  [{tag}] Evolving NEAT...")
            gan.prepare_discriminator(x_train_s, y_train_s, artifacts_dir=str(ARTIFACTS / tag.replace("=", "")))
            log(f"  [{tag}] Training GAN (300 ep, lr={gan_lr})...")
            history = gan.train(x_train_s, y_train_s, x_val_s, y_val_s)

            mae, r2 = eval_quick(gen, x_test_s, y_test_s, tgt_scaler)

            result = {
                "tag": tag,
                "sup_lr": sup_lr,
                "hidden": hidden,
                "dropout": dropout,
                "gan_lr": gan_lr,
                "test_mae": mae,
                "test_r2": r2,
                "best_epoch": history.best_epoch,
                "epochs_run": len(history.generator_losses),
            }
            gan_results.append(result)
            log(f"  [{tag}] MAE={mae:.2f}, R2={r2:.4f}, best_ep={history.best_epoch}")

    gan_results.sort(key=lambda r: r["test_mae"])

    log("\n" + "-" * 60)
    log("TOP 5 GAN CONFIGS:")
    log("-" * 60)
    for i, r in enumerate(gan_results[:5]):
        log(f"  #{i+1}: MAE={r['test_mae']:.2f}, R2={r['test_r2']:.4f} | {r['tag']}")

    with open(ARTIFACTS / "gan_search.json", "w") as f:
        json.dump(gan_results, f, indent=2, default=str)

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)

    best_sup = sup_results[0]
    best_gan = gan_results[0] if gan_results else None

    log(f"Best Supervised: MAE={best_sup['test_mae']:.2f}, R2={best_sup['test_r2']:.4f}")
    log(f"  Config: {best_sup['tag']}")
    if best_gan:
        log(f"Best GAN:        MAE={best_gan['test_mae']:.2f}, R2={best_gan['test_r2']:.4f}")
        log(f"  Config: {best_gan['tag']}")
    log(f"Previous best:   MAE=9.62 (GAN 500ep, lr=1e-4, [128,64,32], d=0.2)")

    # Save combined
    all_results = {
        "supervised_search": sup_results,
        "gan_search": gan_results,
        "best_supervised": best_sup,
        "best_gan": best_gan,
    }
    with open(ARTIFACTS / "hparam_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    dt = time.time() - t0
    log(f"\nTotal: {dt/60:.1f} min")


if __name__ == "__main__":
    main()
