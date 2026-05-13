"""Fast GAN pipeline test (small pop_size, few generations)."""
import time
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from materialgen.gan_trainer import ConcreteGAN, GANConfig
from materialgen.metrics import compute_regression_metrics
from materialgen.uncertainty import UncertaintyEstimator

ARTIFACTS = Path("artifacts/fast_test")


def main():
    t0 = time.time()

    # 1. Load & split data
    ds = load_and_unify_datasets("data")
    split = grouped_stratified_split(ds, seed=42)
    x_all, y_all = ds.all_features, ds.target.to_numpy()
    x_train, y_train = x_all[split["train"]], y_all[split["train"]]
    x_val, y_val = x_all[split["val"]], y_all[split["val"]]
    x_test, y_test = x_all[split["test"]], y_all[split["test"]]

    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
    x_train_s = feat_scaler.transform(x_train)
    x_val_s = feat_scaler.transform(x_val)
    x_test_s = feat_scaler.transform(x_test)
    y_train_s = tgt_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = tgt_scaler.transform(y_val.reshape(-1, 1)).ravel()

    print(f"Data: train={len(x_train_s)}, val={len(x_val_s)}, test={len(x_test_s)}")

    # 2. Quick supervised baseline (50 epochs)
    config = GeneratorConfig(
        input_dim=x_train_s.shape[1], hidden_dims=[128, 64, 32],
        epochs=50, batch_size=64, learning_rate=5e-4, seed=42,
    )
    gen = ConcreteGenerator(config)
    history = train_generator_supervised(gen, x_train_s, y_train_s, x_val_s, y_val_s, config=config)
    print(f"Baseline: {history['epochs_run']} epochs, val_loss={history['best_val_loss']:.4f}")

    mu_s, _ = gen.predict(x_test_s)
    mu = tgt_scaler.inverse_transform(mu_s).ravel()
    m = compute_regression_metrics(y_test, mu)
    print(f"Baseline test: MAE={m.mae:.2f}, R2={m.r2:.4f}")

    # 3. GAN with very small NEAT (pop=30, 2 gen, 100 eval samples)
    print("\n--- GAN Pipeline ---")
    disc_config = DiscriminatorConfig(
        algorithm="bneatest",
        neat_generations=2,
        pop_size=30,
        max_eval_samples=100,
        svi_epochs=10,
        mc_samples=5,
        seed=42,
    )
    disc = NeatBNNDiscriminator(disc_config)

    gan_config = GANConfig(
        total_epochs=30,
        phase1_end=10,
        phase2_end=20,
        generator_lr=2e-4,
        batch_size=64,
        val_interval=10,
        early_stopping_patience=15,
        lambda_physics=0.2,
        seed=42,
    )
    gan = ConcreteGAN(gen, disc, config=gan_config)

    print("Evolving discriminator topology...")
    try:
        gan.prepare_discriminator(x_train_s, y_train_s, artifacts_dir=str(ARTIFACTS))
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return

    print("Training GAN...")
    gan_history = gan.train(x_train_s, y_train_s, x_val_s, y_val_s)
    print(f"GAN: {len(gan_history.generator_losses)} epochs, best_val_mae={gan_history.best_val_mae:.4f}")

    mu_s2, _ = gen.predict(x_test_s)
    mu2 = tgt_scaler.inverse_transform(mu_s2).ravel()
    m2 = compute_regression_metrics(y_test, mu2)
    print(f"GAN test: MAE={m2.mae:.2f}, R2={m2.r2:.4f}")

    # 4. Uncertainty
    est = UncertaintyEstimator(gen, disc)
    unc = est.predict(x_test_s[:10], mc_samples=10)
    print(f"\nUncertainty (10 samples): mean_std={unc.strength_std.mean():.4f}")
    if unc.discriminator_score is not None:
        print(f"Disc score: {unc.discriminator_score.mean():.4f}")

    gan.save(str(ARTIFACTS / "model"))
    dt = time.time() - t0
    print(f"\nTotal: {dt:.0f}s")


if __name__ == "__main__":
    main()
