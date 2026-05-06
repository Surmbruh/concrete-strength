# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Прогнозирование прочности бетона с помощью GAN + NEAT + BNN
# 
# **Архитектура**: ConcreteGAN — генератор (FC с residual connections) + 
# дискриминатор (NEAT-эволюционированная топология + Bayesian Neural Network)
#
# **Лучший результат (t=28 дней)**: 
# - Single model MAE = 5.57 MPa, R² = 0.74
# - Stacking ensemble MAE = 4.76 MPa, R² = 0.82
#
# **Бонусные задачи**: Transfer Learning (EWC), Few-Shot, Time Prediction, Multi-Property

# %% [markdown]
# ## 1. Setup

# %%
import os, sys, subprocess, json, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

REPO_NAME = "concrete-strength"
GITHUB_USERNAME = "Surmbruh"
REPO_DIR = f"/content/{REPO_NAME}"
EXPERIMENTS_DIR = "/content/drive/MyDrive/concrete_project/experiments"
CHECKPOINT_DIR = os.path.join(EXPERIMENTS_DIR, "checkpoints")

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone / pull repo
if os.path.exists(REPO_DIR):
    print("Repo exists, pulling latest...")
    os.chdir(REPO_DIR)
    subprocess.run(["git", "pull"], check=True)
else:
    print("Cloning repo...")
    result = subprocess.run(
        ["git", "clone", f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git", REPO_DIR],
        capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(REPO_DIR):
        raise RuntimeError(
            f"git clone failed! Make sure the repo is PUBLIC.\n"
            f"URL: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}\n"
            f"stderr: {result.stderr}")
    os.chdir(REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "-q"], check=True)
print(f"Working dir: {os.getcwd()}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# %% [markdown]
# ## 2. Данные
# 
# Три источника, объединённых в единую схему:
# | Источник | Записей | Особенности |
# |----------|:-------:|-------------|
# | Normal_Concrete_DB | ~1030 | BFS, зола-унос, slump |
# | Boxcrete | ~1200 | Временные ряды (1,3,5,28 дней) |
# | Synthetic | ~1500 | strength_1/3/7/28 |
# 
# **Итого**: 3745 записей, 10 признаков (7 composition + 3 derived)

# %%
from materialgen.data_preparation import load_and_unify_datasets, stratified_split
from materialgen.scaler import StandardScaler

ds = load_and_unify_datasets("data")
split = stratified_split(ds, seed=42)

x_all = ds.all_features
y_all = ds.target.to_numpy()
ages_all = ds.age_days.to_numpy()

x_train = x_all[split["train"]]
y_train = y_all[split["train"]]
feat_scaler = StandardScaler.fit(x_train)
tgt_scaler = StandardScaler.fit(y_train.reshape(-1, 1))

print(f"Всего: {len(ds.features)} записей")
print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")
print(f"Признаки: {ds.composition_columns + ds.derived_columns}")
print(f"\nИсточники: {dict(ds.source.value_counts())}")
print(f"Возрасты: {sorted(ds.age_days.unique())}")

# %%
# Распределение прочности и возрастов
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(y_all, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Прочность (МПа)')
axes[0].set_ylabel('Количество')
axes[0].set_title('Распределение прочности')
axes[0].axvline(y_all.mean(), color='red', linestyle='--', label=f'Mean={y_all.mean():.1f}')
axes[0].legend()

age_counts = pd.Series(ages_all).value_counts().sort_index()
axes[1].bar(range(len(age_counts)), age_counts.values, color='#4CAF50', edgecolor='white')
axes[1].set_xticks(range(len(age_counts)))
axes[1].set_xticklabels(age_counts.index.astype(int), rotation=45)
axes[1].set_xlabel('Возраст (дни)')
axes[1].set_ylabel('Количество')
axes[1].set_title('Распределение по возрастам')

for i, src in enumerate(ds.source.unique()):
    mask = ds.source == src
    axes[2].hist(y_all[mask.to_numpy()], bins=30, alpha=0.6, label=src)
axes[2].set_xlabel('Прочность (МПа)')
axes[2].set_title('Прочность по источникам')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENTS_DIR, "fig_data_overview.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 3. Архитектура
# 
# ### 3.1 Generator (ConcreteGenerator)
# ```
# Input(10) → ResBlock(256) → ResBlock(128) → ResBlock(64) → [μ, σ]
# ```
# - Residual connections для стабильности градиентов
# - BatchNorm + Dropout для регуляризации  
# - Два выхода: μ (prediction) и σ (aleatoric uncertainty)
# - Loss: Gaussian NLL = 0.5 * [log(σ²) + (y-μ)²/σ²]
#
# ### 3.2 Discriminator (NEAT + BNN)
# - **NEAT**: эволюционирует топологию нейросети (~50 поколений, 50 популяция)
# - **BNN**: инициализирует веса как распределения q(w)≈N(μ_w, σ_w) через SVI (Pyro)
# - Epistemic uncertainty дискриминатора стабилизирует GAN training
#
# ### 3.3 GAN Training (3 фазы)
# | Эпохи | Фаза | Loss генератора |
# |-------|------|-----------------|
# | 0–100 | Warmup | Только MSE |
# | 100–250 | Transition | MSE + нарастающий Adversarial |
# | 250–500 | Full GAN | MSE + полный Adversarial |

# %%
from materialgen.generator import ConcreteGenerator, GeneratorConfig

gen = ConcreteGenerator(GeneratorConfig(
    input_dim=10, hidden_dims=[256, 128, 64], dropout=0.1, seed=42))

total_params = sum(p.numel() for p in gen.parameters())
trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
print(f"Архитектура генератора:")
print(f"  Параметры: {total_params:,} ({trainable_params:,} trainable)")
print(f"\nСтруктура:")
print(gen)

# %% [markdown]
# ## 4. Результаты основной задачи
# 
# ### 4.1 Supervised Grid Search (57 конфигов)

# %%
# Загрузка результатов из JSON
sup_path = os.path.join(CHECKPOINT_DIR, "supervised_grid.json")
if os.path.exists(sup_path):
    with open(sup_path) as f:
        sup_results = json.load(f)
    
    sup_df = pd.DataFrame(sup_results)
    sup_df = sup_df.sort_values("mae")
    
    print("TOP-10 Supervised конфигов:")
    print(sup_df[["tag", "mae", "r2", "picp"]].head(10).to_string(index=False))
    
    best_sup = sup_df.iloc[0]
    print(f"\nBest Supervised: MAE={best_sup['mae']:.2f}, R²={best_sup['r2']:.4f}")
else:
    print("supervised_grid.json not found, using known results")
    best_sup = {"mae": 9.60, "r2": 0.499}

# %% [markdown]
# ### 4.2 GAN Fine-Tuning (9 конфигов)

# %%
gan_path = os.path.join(CHECKPOINT_DIR, "gan_tune.json")
if os.path.exists(gan_path):
    with open(gan_path) as f:
        gan_results = json.load(f)
    
    gan_df = pd.DataFrame(gan_results)
    gan_df = gan_df.sort_values("mae")
    print("GAN Tune Results:")
    cols = [c for c in ["tag","mae","r2","picp","best_epoch"] if c in gan_df.columns]
    print(gan_df[cols].to_string(index=False))
    best_gan = gan_df.iloc[0]
    print(f"\nBest GAN: MAE={best_gan['mae']:.2f}, R²={best_gan['r2']:.4f}")

# %% [markdown]
# ### 4.3 Stacking Ensemble

# %%
stack_path = os.path.join(CHECKPOINT_DIR, "robust_stacking_results.json")
if os.path.exists(stack_path):
    with open(stack_path) as f:
        stack_results = json.load(f)
    
    print("Robust K-Fold Stacking Results:")
    print(f"{'Method':<25} {'OOF MAE':>10} {'Test MAE':>10} {'Test R²':>10}")
    print("-" * 55)
    for name, r in sorted(stack_results.items(), 
                          key=lambda x: x[1].get("test_mae", x[1].get("mae", 99))):
        oof = f"{r['oof_mae']:.3f}" if "oof_mae" in r else "N/A"
        tm = r.get("test_mae", r.get("mae", 0))
        tr = r.get("test_r2", r.get("r2", 0))
        print(f"{name:<25} {oof:>10} {tm:10.3f} {tr:10.4f}")

# %% [markdown]
# ### 4.4 t=28 дней (основной критерий оценки)

# %%
t28_path = os.path.join(CHECKPOINT_DIR, "final_t28_results.json")
if os.path.exists(t28_path):
    with open(t28_path) as f:
        t28_results = json.load(f)
    
    # Individual models at t=28
    if "individual_t28" in t28_results:
        ind = pd.DataFrame(t28_results["individual_t28"]).sort_values("mae")
        print("Индивидуальные модели (t=28):")
        print(ind[["name","mae","r2"]].head(5).to_string(index=False))
    
    # Stacking at t=28
    if "stacking_t28" in t28_results:
        print("\nСтекинг (t=28):")
        print(f"{'Method':<20} {'MAE(t28)':>10} {'R²(t28)':>10}")
        print("-" * 40)
        for n, r in sorted(t28_results["stacking_t28"].items(),
                           key=lambda x: x[1]["test_mae_t28"]):
            print(f"{n:<20} {r['test_mae_t28']:10.3f} {r['test_r2_t28']:10.4f}")

# %%
# Сводная визуализация прогресса
stages = ['Supervised\nbest', 'GAN\nbest', 'Ensemble\ntop-3', 'Stacking\nRidge', 'Stacking\nGBR']
mae_all = [9.60, 9.08, 9.05, 7.02, 5.03]
mae_t28 = [None, 5.57, None, 6.63, 4.76]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(stages))
bars1 = ax.bar(x - 0.2, mae_all, 0.35, label='MAE (все возрасты)', color='#2196F3', alpha=0.8)
bars2_vals = [v if v else 0 for v in mae_t28]
bars2 = ax.bar(x + 0.2, bars2_vals, 0.35, label='MAE (t=28)', color='#FF9800', alpha=0.8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
for i, bar in enumerate(bars2):
    if mae_t28[i]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('MAE (МПа)')
ax.set_title('Прогресс улучшения MAE по этапам')
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.legend()
ax.set_ylim(0, 12)
ax.axhline(y=9.0, color='red', linestyle='--', alpha=0.5, label='Target 9.0')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EXPERIMENTS_DIR, "fig_progress.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 5. Бонусные задачи
# 
# ### 5.1 Transfer Learning (EWC + Replay Buffer)
# 
# Задача: перенос модели на узкую лабораторную выборку (206 образцов, цемент 250-400 кг/м³)

# %%
tr_path = os.path.join(CHECKPOINT_DIR, "transfer_results.json")
if os.path.exists(tr_path):
    with open(tr_path) as f:
        transfer = json.load(f)
    
    print("Transfer Learning Results (narrow lab subset, n=206):")
    print(f"{'Method':<30} {'MAE':>8} {'R²':>8} {'PICP':>8}")
    print("-" * 54)
    for r in transfer:
        picp = f"{r['picp']:.1%}" if r.get('picp') else "N/A"
        print(f"{r['method']:<30} {r['mae']:8.2f} {r['r2']:8.4f} {picp:>8}")
    
    # Visualization
    methods = [r['method'] for r in transfer]
    maes = [r['mae'] for r in transfer]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#4CAF50' if m == 'no_adapt' else '#F44336' if m == 'naive_ft' 
              else '#2196F3' for m in methods]
    bars = ax.barh(methods, maes, color=colors, alpha=0.8)
    ax.set_xlabel('MAE (МПа)')
    ax.set_title('Transfer Learning: адаптация к узкой выборке')
    ax.invert_yaxis()
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{mae:.2f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "fig_transfer.png"), dpi=150)
    plt.show()
    
    print("\nВывод: предобученная модель (no_adapt) уже хорошо обобщает на узкую выборку.")
    print("Fine-tuning на 144 сэмплах УХУДШАЕТ результат — катастрофическое забывание.")

# %% [markdown]
# ### 5.2 Few-Shot Evaluation

# %%
fs_path = os.path.join(CHECKPOINT_DIR, "fewshot_results.json")
if os.path.exists(fs_path):
    with open(fs_path) as f:
        fewshot = json.load(f)
    
    ns = [r['n_samples'] for r in fewshot]
    mae_m = [r['mae_mean'] for r in fewshot]
    mae_s = [r['mae_std'] for r in fewshot]
    r2_m = [r['r2_mean'] for r in fewshot]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.errorbar(ns, mae_m, yerr=mae_s, marker='o', capsize=5, color='#2196F3', linewidth=2)
    ax1.set_xlabel('Размер обучающей выборки')
    ax1.set_ylabel('MAE (МПа)')
    ax1.set_title('Few-Shot: MAE vs размер данных')
    ax1.set_xscale('log')
    ax1.grid(alpha=0.3)
    
    ax2.plot(ns, r2_m, marker='s', color='#4CAF50', linewidth=2)
    ax2.set_xlabel('Размер обучающей выборки')
    ax2.set_ylabel('R²')
    ax2.set_title('Few-Shot: R² vs размер данных')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "fig_fewshot.png"), dpi=150)
    plt.show()
    
    print("Few-Shot Summary:")
    for r in fewshot:
        print(f"  n={r['n_samples']:<6}: MAE={r['mae_mean']:.2f}±{r['mae_std']:.2f}, "
              f"R²={r['r2_mean']:.4f}±{r['r2_std']:.4f}")

# %% [markdown]
# ### 5.3 Прочность во времени

# %%
tp_path = os.path.join(CHECKPOINT_DIR, "time_prediction_results.json")
if os.path.exists(tp_path):
    with open(tp_path) as f:
        time_res = json.load(f)
    
    if "per_age" in time_res:
        pa = time_res["per_age"]
        ages_d = [r['age_days'] for r in pa]
        mae_a = [r['mae'] for r in pa]
        n_a = [r['n'] for r in pa]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(ages_d)), mae_a, color='#FF9800', alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(ages_d)))
        ax.set_xticklabels([f"{a}d\n(n={n})" for a, n in zip(ages_d, n_a)])
        ax.set_ylabel('MAE (МПа)')
        ax.set_title('MAE по возрастным группам')
        for bar, mae in zip(bars, mae_a):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{mae:.1f}', ha='center', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(EXPERIMENTS_DIR, "fig_per_age.png"), dpi=150)
        plt.show()
    
    if "time_curves" in time_res:
        print("\nПримеры временных кривых (предсказание f(t)):")
        for c in time_res["time_curves"][:3]:
            preds = ", ".join(f"t={p['t']}→{p['pred']:.1f}" for p in c['predictions'])
            print(f"  True: t={c['true_age']}d→{c['true_strength']:.1f}MPa")
            print(f"  Pred: {preds}")
    
    if "monotonicity_rate" in time_res and time_res["monotonicity_rate"]:
        print(f"\nМонотонность: {time_res['monotonicity_rate']:.1%}")
        print("(доля составов, где предсказанная прочность монотонно растёт с возрастом)")

# %% [markdown]
# ## 6. Заключение
# 
# ### Ключевые результаты
# 
# | Метрика | Single GAN | Stacking GBR |
# |---------|:----------:|:------------:|
# | **MAE (t=28)** | **5.57** | **4.76** |
# | R² (t=28) | 0.74 | 0.82 |
# | PICP | 96%+ | 98%+ |
# 
# ### Что сработало
# 1. **GAN > Supervised**: adversarial training улучшил MAE с 9.60 → 9.08 (+5.4%)
# 2. **Stacking**: комбинация 24 моделей (9 GAN + 15 enhanced features) → MAE=4.76
# 3. **K-fold CV**: OOF MAE ≈ Test MAE — результаты воспроизводимы
# 4. **Enhanced features**: interaction terms (cement×log_age, w/c×log_age) добавили diversity
# 
# ### Что не сработало
# 1. **Transfer EWC**: предобученная модель уже обобщает лучше, чем fine-tune
# 2. **t=28-only training**: меньше данных → хуже результат (6.56 vs 5.57)
# 3. **Multi-property slump**: слишком мало данных (873/3745), R²=-0.14
# 4. **Монотонность**: только 34% — модель плохо улавливает физику набора прочности

# %%
print("=" * 60)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"\n{'Задача':<35} {'Результат':>15}")
print("-" * 50)
print(f"{'Основная (t=28, single GAN)':<35} {'MAE=5.57':>15}")
print(f"{'Основная (t=28, stacking)':<35} {'MAE=4.76':>15}")
print(f"{'Transfer (no adaptation)':<35} {'MAE=6.32':>15}")
print(f"{'Few-shot (n=500)':<35} {'MAE=10.94':>15}")
print(f"{'Few-shot (n=2000)':<35} {'MAE=10.29':>15}")
print(f"{'Time (t=28d)':<35} {'MAE=5.69':>15}")
print(f"{'Multi-property (strength)':<35} {'MAE=9.76':>15}")
print(f"{'PICP':<35} {'98%+':>15}")
