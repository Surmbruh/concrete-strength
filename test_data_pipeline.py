"""Full pipeline test: data → supervised baseline → physics → GAN training."""
import sys
import time
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from materialgen.gan_trainer import ConcreteGAN, GANConfig
from materialgen.physics import load_gost_table, monotonicity_loss
from materialgen.metrics import compute_regression_metrics, evaluate_model
from materialgen.uncertainty import UncertaintyEstimator

ARTIFACTS = Path("artifacts/full_test")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()

    # ──────────────────────────────────────────────────────────
    # 1. DATA
    # ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("1. DATA LOADING")
    print("=" * 60)

    ds = load_and_unify_datasets("data")
    split = grouped_stratified_split(ds, seed=42)
    print(f"Samples: {ds.n_samples}  |  Train: {len(split['train'])}  Val: {len(split['val'])}  Test: {len(split['test'])}")

    x_all = ds.all_features
    y_all = ds.target.to_numpy()

    x_train, y_train = x_all[split["train"]], y_all[split["train"]]
    x_val, y_val = x_all[split["val"]], y_all[split["val"]]
    x_test, y_test = x_all[split["test"]], y_all[split["test"]]

    # Scalers
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))

    x_train_s = feat_scaler.transform(x_train)
    x_val_s = feat_scaler.transform(x_val)
    x_test_s = feat_scaler.transform(x_test)
    y_train_s = tgt_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = tgt_scaler.transform(y_val.reshape(-1, 1)).ravel()

    # GOST
    gost = None
    gost_path = Path("data/ГОСТы.csv")
    if gost_path.exists():
        try:
            gost = load_gost_table(gost_path)
            print(f"GOST loaded: {len(gost.grades)} grades, R range: {gost.strength_bounds()}")
        except Exception as e:
            print(f"GOST load failed: {e}")

    # ──────────────────────────────────────────────────────────
    # 2. SUPERVISED BASELINE (stronger training)
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("2. SUPERVISED BASELINE (300 epochs)")
    print("=" * 60)

    config = GeneratorConfig(
        input_dim=x_train_s.shape[1],
        hidden_dims=[128, 64, 32],
        epochs=300,
        batch_size=64,
        learning_rate=5e-4,
        weight_decay=1e-4,
        dropout=0.2,
        seed=42,
    )
    gen = ConcreteGenerator(config)

    history = train_generator_supervised(
        gen, x_train_s, y_train_s, x_val_s, y_val_s, config=config,
    )
    print(f"Epochs: {history['epochs_run']}, Best val loss: {history['best_val_loss']:.4f}")

    # Test metrics (original scale)
    mu_s, sigma_s = gen.predict(x_test_s)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    m = compute_regression_metrics(y_test, mu)
    print(f"Baseline: MAE={m.mae:.2f}, RMSE={m.rmse:.2f}, R2={m.r2:.4f}")

    # Save baseline
    torch.save(gen.state_dict(), ARTIFACTS / "baseline_generator.pt")

    # ──────────────────────────────────────────────────────────
    # 3. PHYSICS LOSS TEST
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("3. PHYSICS LOSS CHECK")
    print("=" * 60)

    gen.train()
    x_comp_s = torch.as_tensor(x_train_s[:32, :-1], dtype=torch.float32)
    mono_loss = monotonicity_loss(gen, x_comp_s, times=[1.0, 3.0, 7.0, 28.0])
    print(f"Monotonicity loss (sample): {mono_loss.item():.4f}")

    # ──────────────────────────────────────────────────────────
    # 4. GAN TRAINING (short run for validation)
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("4. GAN TRAINING (50 epochs, smoke test)")
    print("=" * 60)

    # Fresh generator
    gen2 = ConcreteGenerator(config)
    # Load baseline weights as starting point
    gen2.load_state_dict(torch.load(ARTIFACTS / "baseline_generator.pt", weights_only=True))

    disc_config = DiscriminatorConfig(
        algorithm="bneatest",
        neat_generations=3,
        pop_size=30,
        svi_epochs=10,
        mc_samples=5,
        seed=42,
    )
    disc = NeatBNNDiscriminator(disc_config)

    gan_config = GANConfig(
        total_epochs=50,
        phase1_end=15,
        phase2_end=30,
        generator_lr=2e-4,
        batch_size=64,
        val_interval=10,
        early_stopping_patience=20,
        lambda_physics=0.3,
        seed=42,
    )

    gan = ConcreteGAN(gen2, disc, config=gan_config, gost=gost)

    print("Preparing discriminator (NEAT evolution)...")
    try:
        prep_result = gan.prepare_discriminator(
            x_train_s, y_train_s,
            artifacts_dir=str(ARTIFACTS / "disc_neat"),
        )
        print("Discriminator prepared OK")
        has_disc = True
    except Exception as e:
        print(f"Discriminator preparation failed (expected in some envs): {e}")
        has_disc = False

    if has_disc:
        print("Training GAN...")
        gan_history = gan.train(x_train_s, y_train_s, x_val_s, y_val_s)
        print(f"GAN epochs: {len(gan_history.generator_losses)}")
        print(f"Best val MAE: {gan_history.best_val_mae:.4f}")

        # Test GAN model
        mu2_s, sigma2_s = gen2.predict(x_test_s)
        mu2 = tgt_scaler.inverse_transform(mu2_s).ravel()
        m2 = compute_regression_metrics(y_test, mu2)
        print(f"GAN model:  MAE={m2.mae:.2f}, RMSE={m2.rmse:.2f}, R2={m2.r2:.4f}")

        gan.save(str(ARTIFACTS / "gan_model"))

    # ──────────────────────────────────────────────────────────
    # 5. UNCERTAINTY ESTIMATION
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("5. UNCERTAINTY ESTIMATION")
    print("=" * 60)

    estimator = UncertaintyEstimator(gen, disc if has_disc else None)
    result = estimator.predict(x_test_s[:20], mc_samples=20)
    print(f"Predictions shape: {result.strength_mean.shape}")
    print(f"Mean uncertainty: {result.strength_std.mean():.4f}")
    print(f"Aleatoric:  {result.aleatoric_std.mean():.4f}")
    print(f"Epistemic:  {result.epistemic_std.mean():.4f}")
    print(f"CI coverage: [{result.ci_lower.mean():.2f}, {result.ci_upper.mean():.2f}]")
    if result.discriminator_score is not None:
        print(f"Disc score: {result.discriminator_score.mean():.4f} +/- {result.discriminator_std.mean():.4f}")

    # ──────────────────────────────────────────────────────────
    # 6. FULL EVALUATION
    # ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("6. FULL EVALUATION")
    print("=" * 60)

    ages_test = ds.age_days.to_numpy()[split["test"]]
    eval_result = evaluate_model(y_test, mu, y_std=sigma_s.ravel(), age_days=ages_test)
    print(f"Overall: MAE={eval_result.regression.mae:.2f}, R2={eval_result.regression.r2:.4f}")
    if eval_result.calibration:
        print(f"PICP={eval_result.calibration.picp:.2%}, MPIW={eval_result.calibration.mpiw:.2f}")
    for t, tm in sorted(eval_result.per_time.items()):
        print(f"  t={t:3d}d: MAE={tm.mae:.2f}, R2={tm.r2:.4f}, n={tm.n_samples}")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f}s")
    print("ALL DONE.")


if __name__ == "__main__":
    main()
