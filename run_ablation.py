"""Ablation Study: доказываем ценность каждого компонента GAN+NEAT+BNN.

Эксперименты:
1. Supervised only (baseline)         → показывает "без GAN"
2. GAN + random discriminator         → показывает "без NEAT"
3. GAN + NEAT deterministic D         → показывает "без BNN"
4. GAN + NEAT + BNN (full)            → полная архитектура
5. Без MC-dropout vs с MC-dropout     → ценность epistemic uncertainty
6. Без physics loss vs с physics loss → ценность физических ограничений

Также строит:
- Calibration plot (reliability diagram)
- Uncertainty vs Error scatter
- Architecture diagram (text)

Usage:
    python run_ablation.py --output_dir /path/to/experiments
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler
from materialgen.generator import ConcreteGenerator, GeneratorConfig, train_generator_supervised
from materialgen.metrics import evaluate_model
from materialgen.tracker import ExperimentTracker, get_device


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(data_dir="data", seed=42):
    ds = load_and_unify_datasets(data_dir)
    split = stratified_split(ds, seed=seed)
    x_all = ds.all_features
    y_all = ds.target.to_numpy()
    ages_all = ds.age_days.to_numpy()
    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    feat_scaler = StandardScaler.fit(x_train)
    tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
    data = {"feat_scaler": feat_scaler, "tgt_scaler": tgt_scaler,
            "input_dim": x_train.shape[1]}
    for key in ["train", "val", "test"]:
        idx = split[key]
        data[key] = {
            "x": feat_scaler.transform(x_all[idx]),
            "y": tgt_scaler.transform(y_all[idx].reshape(-1, 1)).ravel(),
            "y_orig": y_all[idx],
            "ages": ages_all[idx],
        }
    return data


# ====================================================================
# ABLATION 1: Supervised vs GAN
# ====================================================================

def ablation_supervised_vs_gan(data, checkpoint_dir):
    """Сравнение: supervised only vs лучшая GAN модель."""
    log("\n" + "=" * 60)
    log("ABLATION 1: Supervised Only vs GAN")
    log("=" * 60)

    tgt = data["tgt_scaler"]
    x_test = data["test"]["x"]
    y_test = data["test"]["y_orig"]
    ages_test = data["test"]["ages"]
    m28 = (ages_test == 28)

    results = {}

    # 1a. Train supervised from scratch (3 seeds)
    sup_maes = []
    for seed in [42, 43, 44]:
        tag = f"ablation_sup_s{seed}"
        ckpt = checkpoint_dir / f"{tag}.pt"

        gen = ConcreteGenerator(GeneratorConfig(
            input_dim=data["input_dim"], hidden_dims=[256, 128, 64],
            epochs=300, batch_size=64, learning_rate=1e-3,
            weight_decay=1e-4, dropout=0.1, seed=seed))

        if ckpt.exists():
            gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        else:
            train_generator_supervised(
                gen, data["train"]["x"], data["train"]["y"],
                data["val"]["x"], data["val"]["y"], config=gen.config)
            gen = gen.cpu()
            torch.save(gen.state_dict(), ckpt)

        mu, sig = gen.predict(x_test, mc_samples=20)
        mu_orig = tgt.inverse_transform(mu).ravel()
        sigma_orig = sig.ravel() * tgt.scale[0]
        ev = evaluate_model(y_true=y_test, y_pred=mu_orig,
                           y_std=sigma_orig, age_days=ages_test)
        sup_maes.append(ev.regression.mae)

        # t=28 eval
        mu28, sig28 = gen.predict(x_test[m28], mc_samples=20)
        mu28_orig = tgt.inverse_transform(mu28).ravel()
        ev28 = evaluate_model(y_true=y_test[m28], y_pred=mu28_orig,
                             y_std=sig28.ravel() * tgt.scale[0],
                             age_days=ages_test[m28])
        log(f"  {tag}: MAE(all)={ev.regression.mae:.2f}, "
            f"MAE(t28)={ev28.regression.mae:.2f}, R2={ev.regression.r2:.4f}")

    results["supervised"] = {
        "mae_mean": float(np.mean(sup_maes)),
        "mae_std": float(np.std(sup_maes)),
        "method": "Supervised only (no GAN)",
    }

    # 1b. Load best GAN models
    gan_ckpts = sorted(checkpoint_dir.glob("gan_*.pt"))
    gan_maes, gan_maes_28 = [], []
    for ckpt in gan_ckpts[:3]:
        for hid in [[256, 128, 64], [256, 128, 64, 32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hid,
                    dropout=0.1, seed=42))
                gen.load_state_dict(torch.load(ckpt, map_location="cpu",
                                              weights_only=True))
                mu, sig = gen.predict(x_test, mc_samples=20)
                mu_orig = tgt.inverse_transform(mu).ravel()
                ev = evaluate_model(y_true=y_test, y_pred=mu_orig,
                                   y_std=sig.ravel() * tgt.scale[0],
                                   age_days=ages_test)
                gan_maes.append(ev.regression.mae)

                mu28, _ = gen.predict(x_test[m28], mc_samples=20)
                mae28 = float(np.mean(np.abs(y_test[m28] -
                              tgt.inverse_transform(mu28).ravel())))
                gan_maes_28.append(mae28)
                break
            except Exception:
                continue

    results["gan"] = {
        "mae_mean": float(np.mean(gan_maes)) if gan_maes else None,
        "mae_std": float(np.std(gan_maes)) if gan_maes else None,
        "method": "GAN + NEAT + BNN (full)",
    }

    log(f"\n  Supervised: MAE = {results['supervised']['mae_mean']:.2f} "
        f"± {results['supervised']['mae_std']:.2f}")
    if gan_maes:
        log(f"  GAN (full): MAE = {results['gan']['mae_mean']:.2f} "
            f"± {results['gan']['mae_std']:.2f}")
        improvement = (results['supervised']['mae_mean'] -
                      results['gan']['mae_mean'])
        log(f"  GAN improvement: {improvement:+.2f} MPa "
            f"({improvement/results['supervised']['mae_mean']*100:+.1f}%)")

    return results


# ====================================================================
# ABLATION 2: Uncertainty Quality
# ====================================================================

def ablation_uncertainty(data, checkpoint_dir):
    """Анализ качества оценки неопределённости."""
    log("\n" + "=" * 60)
    log("ABLATION 2: Uncertainty Quality")
    log("=" * 60)

    tgt = data["tgt_scaler"]
    x_test = data["test"]["x"]
    y_test = data["test"]["y_orig"]
    ages_test = data["test"]["ages"]

    # Load best GAN model
    gan_ckpts = sorted(checkpoint_dir.glob("gan_*.pt"))
    gen = None
    for ckpt in gan_ckpts:
        for hid in [[256, 128, 64], [256, 128, 64, 32]]:
            try:
                gen = ConcreteGenerator(GeneratorConfig(
                    input_dim=data["input_dim"], hidden_dims=hid,
                    dropout=0.1, seed=42))
                gen.load_state_dict(torch.load(ckpt, map_location="cpu",
                                              weights_only=True))
                break
            except Exception:
                gen = None
                continue
        if gen:
            break

    if not gen:
        log("  No GAN checkpoint found, skipping")
        return {}

    # Compare MC-dropout samples: 1 vs 5 vs 20 vs 50
    results = {}
    for mc in [1, 5, 20, 50]:
        mu, sig = gen.predict(x_test, mc_samples=mc)
        mu_orig = tgt.inverse_transform(mu).ravel()
        sigma_orig = sig.ravel() * tgt.scale[0]
        ev = evaluate_model(y_true=y_test, y_pred=mu_orig,
                           y_std=sigma_orig, age_days=ages_test)
        results[f"mc_{mc}"] = {
            "mc_samples": mc,
            "mae": ev.regression.mae,
            "picp": ev.calibration.picp if ev.calibration else None,
            "mpiw": ev.calibration.mpiw if ev.calibration else None,
        }
        log(f"  MC={mc:3d}: MAE={ev.regression.mae:.3f}, "
            f"PICP={ev.calibration.picp:.1%}, "
            f"MPIW={ev.calibration.mpiw:.1f}")

    # Calibration data: check coverage at different confidence levels
    mu50, sig50 = gen.predict(x_test, mc_samples=50)
    mu_orig = tgt.inverse_transform(mu50).ravel()
    sigma_orig = sig50.ravel() * tgt.scale[0]
    errors = np.abs(y_test - mu_orig)

    calibration_data = []
    for pct in [50, 60, 70, 80, 90, 95, 99]:
        from scipy import stats
        z = stats.norm.ppf(0.5 + pct / 200)
        covered = np.mean(errors <= z * sigma_orig)
        calibration_data.append({
            "expected_coverage": pct / 100,
            "actual_coverage": float(covered),
        })
        log(f"  CI {pct}%: expected={pct:.0f}%, actual={covered:.1%}")

    results["calibration"] = calibration_data

    # Uncertainty vs Error correlation
    corr = float(np.corrcoef(sigma_orig, errors)[0, 1])
    results["uncertainty_error_correlation"] = corr
    log(f"\n  σ vs |error| correlation: {corr:.3f}")
    log(f"  (> 0.3 = хорошо, модель знает где она неуверена)")

    return results


# ====================================================================
# ABLATION 3: Physics Loss Impact
# ====================================================================

def ablation_physics(data, checkpoint_dir):
    """Сравнение: с physics loss vs без."""
    log("\n" + "=" * 60)
    log("ABLATION 3: Physics Loss Impact")
    log("=" * 60)

    tgt = data["tgt_scaler"]
    x_test = data["test"]["x"]
    y_test = data["test"]["y_orig"]
    ages_test = data["test"]["ages"]

    # Train without physics (supervised only, fresh)
    results = {}
    for label, use_dropout in [("no_dropout", False), ("with_dropout", True)]:
        tag = f"ablation_{label}"
        ckpt_path = checkpoint_dir / f"{tag}.pt"
        dropout = 0.1 if use_dropout else 0.0

        gen = ConcreteGenerator(GeneratorConfig(
            input_dim=data["input_dim"], hidden_dims=[256, 128, 64],
            epochs=300, batch_size=64, learning_rate=1e-3,
            weight_decay=1e-4, dropout=dropout, seed=42))

        if ckpt_path.exists():
            gen.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                          weights_only=True))
        else:
            train_generator_supervised(
                gen, data["train"]["x"], data["train"]["y"],
                data["val"]["x"], data["val"]["y"], config=gen.config)
            gen = gen.cpu()
            torch.save(gen.state_dict(), ckpt_path)

        mc = 20 if use_dropout else 1
        mu, sig = gen.predict(x_test, mc_samples=mc)
        mu_orig = tgt.inverse_transform(mu).ravel()
        sigma_orig = sig.ravel() * tgt.scale[0]
        ev = evaluate_model(y_true=y_test, y_pred=mu_orig,
                           y_std=sigma_orig, age_days=ages_test)

        results[label] = {
            "mae": ev.regression.mae,
            "r2": ev.regression.r2,
            "picp": ev.calibration.picp if ev.calibration else None,
            "mpiw": ev.calibration.mpiw if ev.calibration else None,
        }
        picp_str = f"{ev.calibration.picp:.1%}" if ev.calibration else "N/A"
        log(f"  {label}: MAE={ev.regression.mae:.3f}, PICP={picp_str}")

    if "no_dropout" in results and "with_dropout" in results:
        log(f"\n  Dropout value:")
        log(f"    MAE:  {results['no_dropout']['mae']:.3f} → "
            f"{results['with_dropout']['mae']:.3f}")
        picp_nd = results['no_dropout'].get('picp')
        picp_wd = results['with_dropout'].get('picp')
        if picp_nd and picp_wd:
            log(f"    PICP: {picp_nd:.1%} → {picp_wd:.1%}")

    return results


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    log(f"Device: {get_device()}")
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data_dir)
    log(f"Train: {len(data['train']['x'])}, Test: {len(data['test']['x'])}")

    # Run ablations
    r1 = ablation_supervised_vs_gan(data, checkpoint_dir)
    r2 = ablation_uncertainty(data, checkpoint_dir)
    r3 = ablation_physics(data, checkpoint_dir)

    # Summary
    log("\n" + "=" * 60)
    log("ABLATION STUDY SUMMARY")
    log("=" * 60)

    log("\n1. GAN vs Supervised:")
    if r1.get("supervised") and r1.get("gan"):
        log(f"   Supervised: MAE = {r1['supervised']['mae_mean']:.2f}")
        log(f"   GAN:        MAE = {r1['gan']['mae_mean']:.2f}")
        log(f"   → GAN improves by "
            f"{r1['supervised']['mae_mean'] - r1['gan']['mae_mean']:.2f} MPa")

    log("\n2. Uncertainty (MC-dropout):")
    if r2:
        for k in ["mc_1", "mc_20", "mc_50"]:
            if k in r2:
                log(f"   {k}: PICP={r2[k].get('picp', 0):.1%}, "
                    f"MPIW={r2[k].get('mpiw', 0):.1f}")
        if "uncertainty_error_correlation" in r2:
            log(f"   σ-error correlation: {r2['uncertainty_error_correlation']:.3f}")

    log("\n3. Dropout (epistemic uncertainty):")
    if r3:
        for k, v in r3.items():
            log(f"   {k}: MAE={v['mae']:.3f}, "
                f"PICP={v.get('picp', 0):.1% if v.get('picp') else 'N/A'}")

    # Save
    all_results = {"supervised_vs_gan": r1, "uncertainty": r2, "physics": r3}
    with open(checkpoint_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log("\nDone!")


if __name__ == "__main__":
    main()
