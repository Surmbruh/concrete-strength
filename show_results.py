import json
r = json.load(open("artifacts/long_gan/long_gan_results.json"))
for name in ["supervised", "gan_nophys", "gan_phys"]:
    reg = r[name]["regression"]
    cal = r[name].get("calibration", {})
    picp = f'{cal["PICP"]*100:.1f}%' if cal else "N/A"
    print(f'{name:20s} MAE={reg["MAE"]:.2f} R2={reg["R2"]:.4f} PICP={picp}')
    pt = r[name].get("per_time", {})
    for t in sorted(pt.keys(), key=int):
        m = pt[t]
        if m["n_samples"] >= 5:
            print(f'  t={int(t):4d}d: MAE={m["MAE"]:.2f} R2={m["R2"]:.4f} n={m["n_samples"]}')
    print()
