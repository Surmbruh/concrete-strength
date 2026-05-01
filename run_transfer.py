"""Transfer Learning: Pre-train on large data, fine-tune on small synthetic.

Strategy:
1. Pre-train supervised generator on Normal_Concrete_DB + boxcrete (2945 samples)
2. Fine-tune with GAN on synthetic_training_data (800 samples)
3. Compare with training from scratch on all data

Evaluates on the full combined test set.
"""
import json
import time
import copy
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

ARTIFACTS = Path("artifacts/transfer")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate(gen, x_test, y_test, ages_test, tgt_scaler, tag, mc_samples=50):
    """Evaluate and print."""
    mu_s, sigma_s = gen.predict(x_test, mc_samples=mc_samples)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    sigma = sigma_s.ravel() * tgt_scaler.scale[0]
    y_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    ev = evaluate_model(y_true=y_orig, y_pred=mu, y_std=sigma, age_days=ages_test)
    reg = ev.regression
    log(f"[{tag}] MAE={reg.mae:.2f}, RMSE={reg.rmse:.2f}, R2={reg.r2:.4f}")
    if ev.calibration:
        log(f"[{tag}] PICP={ev.calibration.picp:.2%}, MPIW={ev.calibration.mpiw:.2f}")
    return ev.to_dict()


def main():
    t0 = time.time()

    # ====================================================================
    # 1. LOAD ALL DATA
    # ====================================================================
    log("Loading data...")

    # Pre-training data: Normal_Concrete_DB + boxcrete
    ds_pretrain = load_and_unify_datasets(
        "data", include_normal=True, include_boxcrete=True, include_synthetic=False
    )
    # Fine-tuning data: synthetic only
    ds_finetune = load_and_unify_datasets(
        "data", include_normal=False, include_boxcrete=False, include_synthetic=True
    )
    # Full data (for combined test evaluation)
    ds_full = load_and_unify_datasets("data")
    split_full = stratified_split(ds_full, seed=SEED)

    log(f"Pre-train: {ds_pretrain.n_samples}, Fine-tune: {ds_finetune.n_samples}")
    log(f"Full: {ds_full.n_samples}, test={len(split_full['test'])}")

    # Scale on full training set for fair comparison
    x_full = ds_full.all_features
    y_full = ds_full.target.to_numpy()
    ages_full = ds_full.age_days.to_numpy()

    x_train_full = x_full[split_full["train"]]
    y_train_full = y_full[split_full["train"]]

    feat_scaler = StandardScaler.fit(x_train_full)
    tgt_scaler = StandardScaler.fit(y_train_full.reshape(-1, 1))

    # Full test set (scaled)
    x_test = feat_scaler.transform(x_full[split_full["test"]])
    y_test = tgt_scaler.transform(y_full[split_full["test"]].reshape(-1, 1)).ravel()
    ages_test = ages_full[split_full["test"]]

    # Pre-train splits (using pre-train data with same scaler)
    split_pt = stratified_split(ds_pretrain, seed=SEED)
    x_pt_all = feat_scaler.transform(ds_pretrain.all_features)
    y_pt_all = tgt_scaler.transform(ds_pretrain.target.to_numpy().reshape(-1, 1)).ravel()

    x_pt_train = x_pt_all[split_pt["train"]]
    y_pt_train = y_pt_all[split_pt["train"]]
    x_pt_val = x_pt_all[split_pt["val"]]
    y_pt_val = y_pt_all[split_pt["val"]]

    # Fine-tune splits
    split_ft = stratified_split(ds_finetune, seed=SEED)
    x_ft_all = feat_scaler.transform(ds_finetune.all_features)
    y_ft_all = tgt_scaler.transform(ds_finetune.target.to_numpy().reshape(-1, 1)).ravel()

    x_ft_train = x_ft_all[split_ft["train"]]
    y_ft_train = y_ft_all[split_ft["train"]]
    x_ft_val = x_ft_all[split_ft["val"]]
    y_ft_val = y_ft_all[split_ft["val"]]

    input_dim = x_pt_train.shape[1]
    log(f"Pre-train: train={len(x_pt_train)}, val={len(x_pt_val)}")
    log(f"Fine-tune: train={len(x_ft_train)}, val={len(x_ft_val)}")

    gost = None
    try:
        gost = load_gost_table(Path("data") / "ГОСТы.csv")
    except Exception:
        pass

    gen_config = GeneratorConfig(
        input_dim=input_dim, hidden_dims=[128, 64, 32],
        dropout=0.2, seed=SEED,
    )
    results = {}

    # ====================================================================
    # 2. BASELINE: Supervised on full data (for reference)
    # ====================================================================
    log("\n" + "=" * 60)
    log("BASELINE: Supervised on ALL data (300 epochs)")
    log("=" * 60)

    x_full_train = feat_scaler.transform(x_train_full)
    y_full_train = tgt_scaler.transform(y_train_full.reshape(-1, 1)).ravel()
    x_full_val = feat_scaler.transform(x_full[split_full["val"]])
    y_full_val = tgt_scaler.transform(y_full[split_full["val"]].reshape(-1, 1)).ravel()

    gen_baseline = ConcreteGenerator(GeneratorConfig(
        input_dim=input_dim, hidden_dims=[128, 64, 32],
        epochs=300, batch_size=64, learning_rate=5e-4,
        weight_decay=1e-4, dropout=0.2, seed=SEED,
    ))
    log("Training...")
    hist = train_generator_supervised(
        gen_baseline, x_full_train, y_full_train, x_full_val, y_full_val,
        config=gen_baseline.config,
    )
    log(f"Stopped at epoch {hist['epochs_run']}")
    results["baseline_full"] = evaluate(gen_baseline, x_test, y_test, ages_test, tgt_scaler, "FULL")

    # ====================================================================
    # 3. TRANSFER: Pre-train on large data -> Fine-tune on synthetic
    # ====================================================================
    log("\n" + "=" * 60)
    log("TRANSFER: Pre-train (2945) -> Fine-tune (800)")
    log("=" * 60)

    # Step 1: Pre-train on large data
    gen_transfer = ConcreteGenerator(GeneratorConfig(
        input_dim=input_dim, hidden_dims=[128, 64, 32],
        epochs=200, batch_size=64, learning_rate=5e-4,
        weight_decay=1e-4, dropout=0.2, seed=SEED,
    ))
    log("Pre-training on Normal_Concrete + boxcrete (200 epochs)...")
    hist_pt = train_generator_supervised(
        gen_transfer, x_pt_train, y_pt_train, x_pt_val, y_pt_val,
        config=gen_transfer.config,
    )
    log(f"Pre-train done at epoch {hist_pt['epochs_run']}")
    results["pretrain_only"] = evaluate(gen_transfer, x_test, y_test, ages_test, tgt_scaler, "PRE")

    pretrain_state = {k: v.clone() for k, v in gen_transfer.state_dict().items()}
    torch.save(pretrain_state, ARTIFACTS / "pretrained.pt")

    # Step 2a: Fine-tune supervised only (lower LR, freeze early layers)
    log("\nFine-tuning supervised (100 epochs, lr=1e-4, freeze 2 layers)...")
    gen_ft_sup = ConcreteGenerator(gen_config)
    gen_ft_sup.load_state_dict(copy.deepcopy(pretrain_state))

    # Freeze first 2 layers
    frozen = 0
    for child in gen_ft_sup.backbone.children():
        if frozen >= 2:
            break
        for param in child.parameters():
            param.requires_grad = False
        frozen += 1

    hist_ft = train_generator_supervised(
        gen_ft_sup, x_ft_train, y_ft_train, x_ft_val, y_ft_val,
        config=GeneratorConfig(
            input_dim=input_dim, epochs=100, batch_size=32,
            learning_rate=1e-4, weight_decay=1e-4, dropout=0.2, seed=SEED,
        ),
    )
    log(f"Fine-tune supervised done at epoch {hist_ft['epochs_run']}")
    results["transfer_supervised"] = evaluate(gen_ft_sup, x_test, y_test, ages_test, tgt_scaler, "TF-S")

    # Step 2b: Fine-tune with GAN
    log("\nFine-tuning with GAN (200 epochs)...")
    gen_ft_gan = ConcreteGenerator(gen_config)
    gen_ft_gan.load_state_dict(copy.deepcopy(pretrain_state))

    # Freeze first 2 layers for GAN too
    frozen = 0
    for child in gen_ft_gan.backbone.children():
        if frozen >= 2:
            break
        for param in child.parameters():
            param.requires_grad = False
        frozen += 1

    disc = NeatBNNDiscriminator(DiscriminatorConfig(
        algorithm="bneatest", neat_generations=5, pop_size=50,
        max_eval_samples=150, svi_epochs=20, mc_samples=5, seed=SEED,
    ))
    gan = ConcreteGAN(
        gen_ft_gan, disc,
        config=GANConfig(
            total_epochs=200, phase1_end=40, phase2_end=100,
            generator_lr=1e-4, generator_weight_decay=1e-4,
            batch_size=32, val_interval=5, early_stopping_patience=40,
            lambda_physics=0.0, seed=SEED,
        ),
    )
    log("Evolving NEAT discriminator...")
    gan.prepare_discriminator(x_ft_train, y_ft_train, artifacts_dir=str(ARTIFACTS / "gan_ft"))
    log("Training GAN...")
    hist_gan = gan.train(x_ft_train, y_ft_train, x_ft_val, y_ft_val)
    log(f"GAN fine-tune best epoch {hist_gan.best_epoch}")
    gan.save(str(ARTIFACTS / "transfer_gan"))
    results["transfer_gan"] = evaluate(gen_ft_gan, x_test, y_test, ages_test, tgt_scaler, "TF-G")

    # ====================================================================
    # 4. NO TRANSFER: Train from scratch on synthetic only
    # ====================================================================
    log("\n" + "=" * 60)
    log("NO TRANSFER: Supervised on synthetic only (300 epochs)")
    log("=" * 60)

    gen_scratch = ConcreteGenerator(GeneratorConfig(
        input_dim=input_dim, hidden_dims=[128, 64, 32],
        epochs=300, batch_size=32, learning_rate=5e-4,
        weight_decay=1e-4, dropout=0.2, seed=SEED,
    ))
    hist_sc = train_generator_supervised(
        gen_scratch, x_ft_train, y_ft_train, x_ft_val, y_ft_val,
        config=gen_scratch.config,
    )
    log(f"Scratch done at epoch {hist_sc['epochs_run']}")
    results["scratch_synthetic"] = evaluate(gen_scratch, x_test, y_test, ages_test, tgt_scaler, "SCR")

    # ====================================================================
    # 5. SUMMARY
    # ====================================================================
    log("\n" + "=" * 60)
    log("TRANSFER LEARNING COMPARISON")
    log("=" * 60)
    header = f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'PICP':>8}"
    log(header)
    log("-" * len(header))
    for name, r in results.items():
        reg = r["regression"]
        cal = r.get("calibration", {})
        picp = f"{cal['PICP']*100:.1f}%" if cal else "N/A"
        log(f"{name:<25} {reg['MAE']:8.2f} {reg['RMSE']:8.2f} {reg['R2']:8.4f} {picp:>8}")

    with open(ARTIFACTS / "transfer_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    dt = time.time() - t0
    log(f"\nTotal: {dt/60:.1f} min")
    log(f"Results: {ARTIFACTS / 'transfer_results.json'}")


if __name__ == "__main__":
    main()
