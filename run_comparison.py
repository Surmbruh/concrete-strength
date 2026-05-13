"""Comparison experiment: Supervised Baseline vs GAN vs GAN+Physics.

Trains three models and generates a detailed comparison report.
Uses StandardScaler for both features and targets.
"""
import json
import time
import numpy as np
import torch
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, grouped_stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.discriminator import NeatBNNDiscriminator, DiscriminatorConfig
from materialgen.gan_trainer import ConcreteGAN, GANConfig
from materialgen.physics import load_gost_table
from materialgen.metrics import compute_regression_metrics, evaluate_model
from materialgen.uncertainty import UncertaintyEstimator

ARTIFACTS = Path("artifacts/comparison")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ====================================================================
# Config
# ====================================================================
SUPERVISED_EPOCHS = 300
GAN_EPOCHS = 200
NEAT_POP = 50
NEAT_GEN = 5
NEAT_EVAL_SAMPLES = 150
BATCH_SIZE = 64
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def train_supervised_model(
    x_train, y_train, x_val, y_val, input_dim, tag="baseline"
):
    """Train a pure supervised generator."""
    config = GeneratorConfig(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        epochs=SUPERVISED_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=5e-4,
        weight_decay=1e-4,
        dropout=0.2,
        seed=SEED,
    )
    gen = ConcreteGenerator(config)
    log(f"[{tag}] Training supervised ({SUPERVISED_EPOCHS} epochs)...")
    history = train_generator_supervised(gen, x_train, y_train, x_val, y_val, config=config)
    log(f"[{tag}] Stopped at epoch {history['epochs_run']}, val_loss={history['best_val_loss']:.4f}")
    return gen, history


def train_gan_model(
    gen, x_train, y_train, x_val, y_val,
    use_physics: bool, gost, tag="gan"
):
    """Fine-tune a generator with GAN training."""
    disc_config = DiscriminatorConfig(
        algorithm="bneatest",
        neat_generations=NEAT_GEN,
        pop_size=NEAT_POP,
        max_eval_samples=NEAT_EVAL_SAMPLES,
        svi_epochs=20,
        mc_samples=5,
        seed=SEED,
    )
    disc = NeatBNNDiscriminator(disc_config)

    gan_config = GANConfig(
        total_epochs=GAN_EPOCHS,
        phase1_end=int(GAN_EPOCHS * 0.2),
        phase2_end=int(GAN_EPOCHS * 0.5),
        generator_lr=2e-4,
        generator_weight_decay=1e-4,
        batch_size=BATCH_SIZE,
        val_interval=10,
        early_stopping_patience=40,
        lambda_physics=0.5 if use_physics else 0.0,
        lambda_mono=1.0,
        lambda_abrams=0.5,
        lambda_gost=0.3,
        seed=SEED,
    )

    gan = ConcreteGAN(gen, disc, config=gan_config, gost=gost if use_physics else None)

    log(f"[{tag}] Evolving discriminator (pop={NEAT_POP}, gen={NEAT_GEN})...")
    gan.prepare_discriminator(x_train, y_train, artifacts_dir=str(ARTIFACTS / tag))

    log(f"[{tag}] Training GAN ({GAN_EPOCHS} epochs, physics={'ON' if use_physics else 'OFF'})...")
    history = gan.train(x_train, y_train, x_val, y_val)
    log(f"[{tag}] Best epoch {history.best_epoch}, val_mae={history.best_val_mae:.4f}")

    return gen, disc, history, gan


def evaluate_and_report(gen, disc, x_test, y_test, ages_test,
                        tgt_scaler, tag, mc_samples=30):
    """Evaluate a model and return results dict."""
    # Predict in scaled space
    mu_s, sigma_s = gen.predict(x_test, mc_samples=mc_samples)

    # Convert back to original scale
    mu_orig = tgt_scaler.inverse_transform(mu_s).ravel()
    sigma_orig = sigma_s.ravel() * tgt_scaler.scale[0]  # scale std by target scale

    y_test_orig = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Full evaluation
    evaluation = evaluate_model(
        y_true=y_test_orig,
        y_pred=mu_orig,
        y_std=sigma_orig,
        age_days=ages_test,
    )

    # Print summary
    reg = evaluation.regression
    log(f"[{tag}] TEST: MAE={reg.mae:.2f} MPa, RMSE={reg.rmse:.2f}, "
        f"MAPE={reg.mape:.1f}%, R2={reg.r2:.4f} (n={reg.n_samples})")

    if evaluation.calibration:
        cal = evaluation.calibration
        log(f"[{tag}] CALIBRATION: PICP={cal.picp:.2%}, MPIW={cal.mpiw:.2f}, "
            f"sharpness={cal.sharpness:.3f}")

    for t, m in sorted(evaluation.per_time.items()):
        log(f"[{tag}]   t={t:3d}d: MAE={m.mae:.2f}, R2={m.r2:.4f}, n={m.n_samples}")

    return evaluation.to_dict()


def main():
    t0 = time.time()

    # ==============================================================
    # 1. DATA
    # ==============================================================
    log("Loading data...")
    ds = load_and_unify_datasets("data")
    split = grouped_stratified_split(ds, seed=SEED)
    log(f"Data: {ds.n_samples} samples, train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}")

    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    ages_all = ds.age_days.to_numpy()

    x_train, y_train = x_all[split["train"]], y_all[split["train"]]
    x_val, y_val = x_all[split["val"]], y_all[split["val"]]
    x_test, y_test = x_all[split["test"]], y_all[split["test"]]
    ages_test = ages_all[split["test"]]

    # Scale
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))

    x_train_s = feat_scaler.transform(x_train)
    x_val_s = feat_scaler.transform(x_val)
    x_test_s = feat_scaler.transform(x_test)
    y_train_s = tgt_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = tgt_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = tgt_scaler.transform(y_test.reshape(-1, 1)).ravel()

    input_dim = x_train_s.shape[1]

    # GOST
    gost = None
    gost_path = Path("data/ГОСТы.csv")
    if gost_path.exists():
        try:
            gost = load_gost_table(gost_path)
            log(f"GOST: {len(gost.grades)} grades loaded")
        except Exception as e:
            log(f"GOST load failed: {e}")

    results = {}

    # ==============================================================
    # 2. MODEL A: Supervised Baseline
    # ==============================================================
    log("")
    log("=" * 60)
    log("MODEL A: SUPERVISED BASELINE")
    log("=" * 60)

    gen_a, hist_a = train_supervised_model(
        x_train_s, y_train_s, x_val_s, y_val_s, input_dim, tag="A"
    )
    torch.save(gen_a.state_dict(), ARTIFACTS / "model_a_supervised.pt")
    results["A_supervised"] = evaluate_and_report(
        gen_a, None, x_test_s, y_test_s, ages_test, tgt_scaler, "A"
    )
    results["A_supervised"]["epochs_run"] = hist_a["epochs_run"]

    # ==============================================================
    # 3. MODEL B: GAN (no physics)
    # ==============================================================
    log("")
    log("=" * 60)
    log("MODEL B: GAN (no physics)")
    log("=" * 60)

    # Start from supervised baseline
    gen_b, _ = train_supervised_model(
        x_train_s, y_train_s, x_val_s, y_val_s, input_dim, tag="B"
    )
    gen_b, disc_b, hist_b, gan_b = train_gan_model(
        gen_b, x_train_s, y_train_s, x_val_s, y_val_s,
        use_physics=False, gost=None, tag="B"
    )
    gan_b.save(str(ARTIFACTS / "model_b_gan"))
    results["B_gan"] = evaluate_and_report(
        gen_b, disc_b, x_test_s, y_test_s, ages_test, tgt_scaler, "B"
    )
    results["B_gan"]["best_epoch"] = hist_b.best_epoch

    # ==============================================================
    # 4. MODEL C: GAN + Physics
    # ==============================================================
    log("")
    log("=" * 60)
    log("MODEL C: GAN + PHYSICS")
    log("=" * 60)

    gen_c, _ = train_supervised_model(
        x_train_s, y_train_s, x_val_s, y_val_s, input_dim, tag="C"
    )
    gen_c, disc_c, hist_c, gan_c = train_gan_model(
        gen_c, x_train_s, y_train_s, x_val_s, y_val_s,
        use_physics=True, gost=gost, tag="C"
    )
    gan_c.save(str(ARTIFACTS / "model_c_gan_physics"))
    results["C_gan_physics"] = evaluate_and_report(
        gen_c, disc_c, x_test_s, y_test_s, ages_test, tgt_scaler, "C"
    )
    results["C_gan_physics"]["best_epoch"] = hist_c.best_epoch

    # ==============================================================
    # 5. COMPARISON TABLE
    # ==============================================================
    log("")
    log("=" * 60)
    log("COMPARISON SUMMARY")
    log("=" * 60)

    header = f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R2':>8} {'PICP':>8}"
    log(header)
    log("-" * len(header))
    for name, r in results.items():
        reg = r["regression"]
        cal = r.get("calibration", {})
        picp_str = f"{cal['PICP']:.2%}" if cal else "N/A"
        log(f"{name:<25} {reg['MAE']:8.2f} {reg['RMSE']:8.2f} "
            f"{reg['MAPE']:7.1f}% {reg['R2']:8.4f} {picp_str:>8}")

    # Save results
    results_path = ARTIFACTS / "comparison_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    dt = time.time() - t0
    log(f"\nTotal time: {dt/60:.1f} min")
    log(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
