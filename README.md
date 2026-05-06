# Прогнозирование прочности бетона: GAN + NEAT + BNN

## Архитектура

```
ConcreteGAN
├── Generator: FC [256→128→64] + Residual + BatchNorm + Dropout
│   └── Выход: (μ, σ) — предсказание + aleatoric uncertainty
├── Discriminator: NEAT-evolved topology + BNN (Pyro SVI)
│   └── Epistemic uncertainty стабилизирует GAN training
└── Training: 3-phase (Warmup → Transition → Full GAN)
```

## Результаты (t=28 дней — основной критерий)

| Метод | MAE | R² | PICP |
|-------|:---:|:--:|:----:|
| Supervised baseline | 9.60 | 0.50 | 98% |
| GAN (single best) | **5.57** | **0.74** | 97% |
| Stacking GBR (24 модели) | **4.76** | **0.82** | 98% |

## Бонусные задачи

- **Transfer Learning**: модель обобщает без адаптации (MAE=6.32)
- **Few-Shot**: MAE=12.84 при n=50, MAE=10.29 при n=2000
- **Time Prediction**: MAE=5.69 при t=28d, монотонность 34%
- **Multi-Property**: Strength + Slump prediction

## Файлы

- `materialgen/` — основной пакет (generator, discriminator, metrics, tracker)
- `run_experiment.py` — supervised grid + GAN tune + full pipeline
- `run_robust_stacking.py` — K-fold CV стекинг
- `run_final_t28.py` — оптимизация под t=28
- `run_bonus_*.py` — бонусные задачи
- `report_notebook.py` — отчётный notebook для Colab

## Запуск

```bash
pip install -e .
python run_experiment.py --mode supervised_grid --output_dir experiments
python run_experiment.py --mode gan_tune --output_dir experiments
python run_robust_stacking.py --output_dir experiments
python run_final_t28.py --output_dir experiments
```
