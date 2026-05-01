"""Physics-informed loss functions и ограничения для бетонных смесей.

Реализует три вида физических ограничений:
1. Монотонность прочности по времени
2. Закон Абрамса (антикорреляция w/c и прочности)
3. ГОСТ-bounds — нормативные ограничения на прочность

Каждая функция возвращает скалярный штраф (torch.Tensor), который
добавляется к loss генератора и/или дискриминатора.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# ГОСТ-ограничения
# =============================================================================

@dataclass
class GostGrade:
    """Одна марка бетона по ГОСТ 26633."""

    mark: str          # М50, М100, ...
    class_b: str       # B3.5, B7.5, ...
    r_min: float       # МПа, нижняя граница прочности на 28 сут.
    r_max: float       # МПа, верхняя граница прочности на 28 сут.
    r_typical: float   # МПа, типичное значение (среднее)


@dataclass
class GostTable:
    """Справочник марок бетона из ГОСТ 26633."""

    grades: list[GostGrade]

    def find_grade_for_strength(self, strength_28d: float) -> GostGrade | None:
        """Найти подходящую марку для заданной прочности на 28 сут."""
        for grade in self.grades:
            if grade.r_min <= strength_28d <= grade.r_max:
                return grade
        return None

    def strength_bounds(self) -> tuple[float, float]:
        """Глобальные min/max прочности по всем маркам."""
        all_min = min(g.r_min for g in self.grades)
        all_max = max(g.r_max for g in self.grades)
        return all_min, all_max

    def to_dict(self) -> dict[str, Any]:
        return {
            "grades": [
                {
                    "mark": g.mark,
                    "class_b": g.class_b,
                    "r_min": g.r_min,
                    "r_max": g.r_max,
                    "r_typical": g.r_typical,
                }
                for g in self.grades
            ]
        }


def load_gost_table(csv_path: str | Path) -> GostTable:
    """Парсинг файла ГОСТы.csv в структурированную таблицу.

    Файл имеет нестандартный формат (заголовки с переносами строк,
    кодировка cp1251/utf-8). Эта функция обрабатывает все крайние
    случаи и возвращает чистую таблицу марок.

    Parameters
    ----------
    csv_path : путь к файлу ГОСТы.csv
    """
    import pandas as pd

    path = Path(csv_path)

    # Попытка чтения в нескольких кодировках
    raw = None
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if raw is None:
        raise ValueError(f"Cannot decode GOST file: {csv_path}")

    lines = raw.strip().splitlines()

    # Ищем строки данных: начинаются с «М» + число
    grades: list[GostGrade] = []
    for line in lines:
        parts = line.split(";")
        if len(parts) < 5:
            continue
        mark_candidate = parts[0].strip().strip('"')
        if not mark_candidate.startswith("М"):
            continue
        try:
            class_b = parts[1].strip().strip('"')
            r_min = float(parts[2].replace(",", "."))
            r_max = float(parts[3].replace(",", "."))
            r_typical = float(parts[4].replace(",", "."))
            grades.append(GostGrade(
                mark=mark_candidate,
                class_b=class_b,
                r_min=r_min,
                r_max=r_max,
                r_typical=r_typical,
            ))
        except (ValueError, IndexError):
            continue

    if not grades:
        raise ValueError(f"No valid GOST grades found in {csv_path}")

    return GostTable(grades=grades)


# =============================================================================
# Физические loss-функции (PyTorch)
# =============================================================================

def monotonicity_loss(
    generator: torch.nn.Module,
    x_composition: torch.Tensor,
    times: list[float] | None = None,
) -> torch.Tensor:
    """Штраф за нарушение монотонности прочности по времени.

    Для каждой пары последовательных моментов (t_i, t_{i+1}) штрафуем,
    если strength(t_{i+1}) < strength(t_i).

    Parameters
    ----------
    generator : модель-генератор с интерфейсом forward(x) → (mu, sigma)
    x_composition : тензор составов [batch, n_components] БЕЗ log_age
    times : список моментов времени (дней); по умолчанию [1, 3, 7, 28]

    Returns
    -------
    Скалярный штраф ≥ 0 (0 если монотонность соблюдена)
    """
    if times is None:
        times = [1.0, 3.0, 7.0, 28.0]

    predictions: list[torch.Tensor] = []
    for t in times:
        log_t = torch.full(
            (x_composition.shape[0], 1),
            float(np.log(t)),
            dtype=x_composition.dtype,
            device=x_composition.device,
        )
        x_with_time = torch.cat([x_composition, log_t], dim=1)
        mu, _sigma = generator(x_with_time)
        predictions.append(mu)

    penalty = torch.tensor(0.0, device=x_composition.device)
    for i in range(len(predictions) - 1):
        # Штраф если прочность убывает: relu(prev - next)
        violation = F.relu(predictions[i] - predictions[i + 1])
        penalty = penalty + violation.mean()

    return penalty


def abrams_loss(
    generator: torch.nn.Module,
    x_with_time: torch.Tensor,
    wc_index: int,
) -> torch.Tensor:
    """Штраф за нарушение закона Абрамса.

    Закон Абрамса: ∂strength/∂(w/c) < 0.
    Штрафуем, если градиент прочности по w/c ratio положительный.

    Parameters
    ----------
    generator : модель-генератор
    x_with_time : входной тензор [batch, input_dim] С log_age
    wc_index : индекс столбца w/c ratio в входном тензоре

    Returns
    -------
    Скалярный штраф ≥ 0
    """
    x = x_with_time.detach().clone().requires_grad_(True)
    mu, _sigma = generator(x)

    # Gradient прочности по входным признакам
    grad_outputs = torch.ones_like(mu)
    grads = torch.autograd.grad(
        outputs=mu,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Градиент по w/c должен быть отрицательным → штрафуем положительный
    grad_wc = grads[:, wc_index]
    penalty = F.relu(grad_wc).mean()
    return penalty


def gost_compliance_loss(
    y_pred_28d: torch.Tensor,
    gost: GostTable,
) -> torch.Tensor:
    """Штраф за выход предсказаний за глобальные ГОСТ-границы.

    Мягкий штраф: если предсказанная прочность на 28 сут. выходит за
    пределы диапазона [global_min, global_max] из ГОСТ-таблицы.

    Parameters
    ----------
    y_pred_28d : предсказания прочности на 28 дней [batch]
    gost : таблица ГОСТ-ограничений

    Returns
    -------
    Скалярный штраф ≥ 0
    """
    r_min, r_max = gost.strength_bounds()
    lower_violation = F.relu(r_min - y_pred_28d)
    upper_violation = F.relu(y_pred_28d - r_max)
    return (lower_violation + upper_violation).mean()


def combined_physics_loss(
    generator: torch.nn.Module,
    x_composition: torch.Tensor,
    x_with_time: torch.Tensor,
    wc_index: int,
    gost: GostTable | None = None,
    y_pred_28d: torch.Tensor | None = None,
    *,
    lambda_mono: float = 1.0,
    lambda_abrams: float = 0.5,
    lambda_gost: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Комбинированный физический штраф.

    Суммирует монотонность + Абрамс + ГОСТ с настраиваемыми весами.

    Returns
    -------
    (total_penalty, detail_dict) — скалярный штраф и разбивка по компонентам
    """
    details: dict[str, float] = {}

    mono = monotonicity_loss(generator, x_composition)
    details["monotonicity"] = float(mono.item())

    abrams = abrams_loss(generator, x_with_time, wc_index)
    details["abrams"] = float(abrams.item())

    total = lambda_mono * mono + lambda_abrams * abrams

    if gost is not None and y_pred_28d is not None:
        gost_loss = gost_compliance_loss(y_pred_28d, gost)
        details["gost"] = float(gost_loss.item())
        total = total + lambda_gost * gost_loss

    details["total"] = float(total.item())
    return total, details
