"""Long GAN training: 500 epochs with physics, larger NEAT, more patience.

Starts from a supervised baseline (300 epochs), then fine-tunes with GAN.
Compares two variants: GAN-only and GAN+Physics.
Saves training curves for visualization.
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
from materialgen.metrics import evaluate_model

ARTIFACTS = Path("artifacts/long_gan")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ====================================================================
# Config - tuned for longer training
# ====================================================================
SUPERVISED_EPOCHS = 300
GAN_EPOCHS = 500
NEAT_POP = 80
NEAT_GEN = 8
NEAT_EVAL_SAMPLES = 200
BATCH_SIZE = 64
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_supervised(x_train, y_train, x_val, y_val, input_dim):
    """Train supervised baseline."""
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
    log(f"Supervised pre-training ({SUPERVISED_EPOCHS} epochs)...")
    history = train_generator_supervised(gen, x_train, y_train, x_val, y_val, config=config)
    log(f"  Stopped at epoch {history['epochs_run']}, val_loss={history['best_val_loss']:.4f}")
    return gen, history


def train_long_gan(gen, x_train, y_train, x_val, y_val,
                   use_physics, gost, tag):
    """Train GAN for 500 epochs."""
    disc_config = DiscriminatorConfig(
        algorithm="bneatest",
        neat_generations=NEAT_GEN,
        pop_size=NEAT_POP,
        max_eval_samples=NEAT_EVAL_SAMPLES,
        svi_epochs=30,
        mc_samples=10,
        seed=SEED,
    )
    disc = NeatBNNDiscriminator(disc_config)

    gan_config = GANConfig(
        total_epochs=GAN_EPOCHS,
        # Progressive schedule: 100 supervised -> 150 transition -> 250 adversarial
        phase1_end=100,
        phase2_end=250,
        generator_lr=1e-4,          # lower LR for stability
        generator_weight_decay=1e-4,
        batch_size=BATCH_SIZE,
        val_interval=5,             # validate more often
        early_stopping_patience=80, # more patience for 500 epochs
        n_disc_steps=1,
        disc_svi_steps=3,
        lambda_physics=0.5 if use_physics else 0.0,
        lambda_mono=1.0,
        lambda_abrams=0.5,
        lambda_gost=0.3,
        label_smoothing=0.1,
        noise_std=0.05,
        seed=SEED,
    )

    gan = ConcreteGAN(gen, disc, config=gan_config, gost=gost if use_physics else None)

    log(f"[{tag}] Evolving NEAT discriminator (pop={NEAT_POP}, gen={NEAT_GEN})...")
    gan.prepare_discriminator(x_train, y_train, artifacts_dir=str(ARTIFACTS / tag))
    log(f"[{tag}] NEAT evolution complete")

    log(f"[{tag}] GAN training ({GAN_EPOCHS} epochs, physics={'ON' if use_physics else 'OFF'})...")
    history = gan.train(x_train, y_train, x_val, y_val)
    log(f"[{tag}] Best epoch {history.best_epoch}, val_mae={history.best_val_mae:.4f}")
    log(f"[{tag}] Epochs run: {len(history.generator_losses)}")

    return gen, disc, history, gan


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
    for t, m in sorted(ev.per_time.items()):
        if m.n_samples >= 5:
            log(f"[{tag}]   t={t:3d}d: MAE={m.mae:.2f}, R2={m.r2:.4f} (n={m.n_samples})")
    return ev.to_dict()


def main():
    t0 = time.time()

    # 1. DATA
    log("Loading data...")
    ds = load_and_unify_datasets("data")
    split = grouped_stratified_split(ds, seed=SEED)
    x_all, y_all = ds.all_features, ds.target.to_numpy()
    ages_all = ds.age_days.to_numpy()

    x_train, y_train = x_all[split["train"]], y_all[split["train"]]
    x_val, y_val = x_all[split["val"]], y_all[split["val"]]
    x_test, y_test = x_all[split["test"]], y_all[split["test"]]
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

    gost = None
    try:
        gost = load_gost_table(Path("data") / "ГОСТы.csv")
        log(f"GOST: {len(gost.grades)} grades")
    except Exception:
        pass

    log(f"Data: train={len(x_train_s)}, val={len(x_val_s)}, test={len(x_test_s)}")
    results = {}

    # 2. SUPERVISED BASELINE (reuse same weights for both GAN variants)
    log("\n" + "=" * 60)
    log("SUPERVISED BASELINE (300 epochs)")
    log("=" * 60)
    gen_base, hist_base = make_supervised(x_train_s, y_train_s, x_val_s, y_val_s, input_dim)
    base_state = {k: v.clone() for k, v in gen_base.state_dict().items()}
    torch.save(base_state, ARTIFACTS / "supervised_baseline.pt")
    results["supervised"] = evaluate(gen_base, x_test_s, y_test_s, ages_test, tgt_scaler, "SUP")

    # 3. GAN (no physics) - 500 epochs
    log("\n" + "=" * 60)
    log("GAN (no physics) - 500 epochs")
    log("=" * 60)
    gen_b = ConcreteGenerator(GeneratorConfig(input_dim=input_dim, hidden_dims=[128, 64, 32],
                                              dropout=0.2, seed=SEED))
    gen_b.load_state_dict(base_state)
    gen_b, disc_b, hist_b, gan_b = train_long_gan(
        gen_b, x_train_s, y_train_s, x_val_s, y_val_s,
        use_physics=False, gost=None, tag="gan_nophys"
    )
    gan_b.save(str(ARTIFACTS / "gan_nophys"))
    results["gan_nophys"] = evaluate(gen_b, x_test_s, y_test_s, ages_test, tgt_scaler, "GAN")
    results["gan_nophys"]["best_epoch"] = hist_b.best_epoch
    results["gan_nophys"]["epochs_run"] = len(hist_b.generator_losses)

    # 4. GAN + Physics - 500 epochs
    log("\n" + "=" * 60)
    log("GAN + PHYSICS - 500 epochs")
    log("=" * 60)
    gen_c = ConcreteGenerator(GeneratorConfig(input_dim=input_dim, hidden_dims=[128, 64, 32],
                                              dropout=0.2, seed=SEED))
    gen_c.load_state_dict(base_state)
    gen_c, disc_c, hist_c, gan_c = train_long_gan(
        gen_c, x_train_s, y_train_s, x_val_s, y_val_s,
        use_physics=True, gost=gost, tag="gan_phys"
    )
    gan_c.save(str(ARTIFACTS / "gan_phys"))
    results["gan_phys"] = evaluate(gen_c, x_test_s, y_test_s, ages_test, tgt_scaler, "GAN+P")
    results["gan_phys"]["best_epoch"] = hist_c.best_epoch
    results["gan_phys"]["epochs_run"] = len(hist_c.generator_losses)

    # 5. SAVE TRAINING CURVES (for later visualization)
    curves = {
        "gan_nophys": {
            "g_loss": hist_b.generator_losses,
            "d_loss": hist_b.discriminator_losses,
            "sup_loss": hist_b.supervised_losses,
            "adv_loss": hist_b.adversarial_losses,
            "phys_loss": hist_b.physics_losses,
            "val_mae": hist_b.val_mae,
            "val_r2": hist_b.val_r2,
            "lambda_mse": hist_b.lambda_mse_history,
            "lambda_adv": hist_b.lambda_adv_history,
        },
        "gan_phys": {
            "g_loss": hist_c.generator_losses,
            "d_loss": hist_c.discriminator_losses,
            "sup_loss": hist_c.supervised_losses,
            "adv_loss": hist_c.adversarial_losses,
            "phys_loss": hist_c.physics_losses,
            "val_mae": hist_c.val_mae,
            "val_r2": hist_c.val_r2,
            "lambda_mse": hist_c.lambda_mse_history,
            "lambda_adv": hist_c.lambda_adv_history,
        },
    }
    with open(ARTIFACTS / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2, default=str)

    # 6. SUMMARY
    log("\n" + "=" * 60)
    log("LONG GAN COMPARISON")
    log("=" * 60)
    header = f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'PICP':>8} {'Epochs':>8}"
    log(header)
    log("-" * len(header))
    for name, r in results.items():
        reg = r["regression"]
        cal = r.get("calibration", {})
        picp = f"{cal['PICP']:.2%}" if cal else "N/A"
        ep = r.get("epochs_run", "-")
        log(f"{name:<20} {reg['MAE']:8.2f} {reg['RMSE']:8.2f} "
            f"{reg['R2']:8.4f} {picp:>8} {ep:>8}")

    with open(ARTIFACTS / "long_gan_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    dt = time.time() - t0
    log(f"\nTotal: {dt/60:.1f} min")
    log(f"Results: {ARTIFACTS / 'long_gan_results.json'}")


if __name__ == "__main__":
    main()
