#!/usr/bin/env python3
"""Summarize kNN-versus-similarity disagreement cases without causal claims."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_SPLITS = ("chemical_random", "scaffold", "temporal", "species")
CATEGORIES = (
    "both_rescued",
    "knn_only_rescued",
    "similarity_only_rescued",
    "both_omitted",
)
SUPPORT_COLUMNS = (
    "knn",
    "similarity",
    "d_chem",
    "d_species",
    "d_context",
    "d_mech",
    "context_missing_fraction",
    "bioactivity_missing_fraction",
    "model_std",
    "n_endpoints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _panel_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["dataset", "split", "seed"], dropna=False):
        dataset, split, seed = keys
        baseline = group.loc[group["false_negative"].astype(bool)]
        denominator = len(baseline)
        for category in CATEGORIES:
            selected = baseline.loc[baseline["disagreement_category"].eq(category)]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "seed": int(seed),
                    "disagreement_category": category,
                    "baseline_false_negative_chemicals": denominator,
                    "category_chemicals": int(len(selected)),
                    "category_fraction": (
                        float(len(selected) / denominator) if denominator else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _support_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame.disagreement_category.isin(CATEGORIES)].copy()
    group_columns = ["dataset", "split", "disagreement_category"]
    aggregation: dict[str, tuple[str, str]] = {"n_rows": ("chemical_id", "size")}
    for column in SUPPORT_COLUMNS:
        aggregation[f"mean_{column}"] = (column, "mean")
        aggregation[f"median_{column}"] = (column, "median")
    return frame.groupby(group_columns, as_index=False, dropna=False).agg(**aggregation)


def _source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    external = frame.loc[
        frame.dataset.eq("external")
        & frame.false_negative.astype(bool)
    ].copy()
    if external.empty:
        return pd.DataFrame()
    return (
        external.groupby(["source_group", "disagreement_category"], as_index=False)
        .agg(
            rows=("chemical_id", "size"),
            unique_chemical_seed_rows=("chemical_id", "nunique"),
        )
    )


def _write_report(
    output: Path,
    panel_rates: pd.DataFrame,
    support: pd.DataFrame,
    source: pd.DataFrame,
) -> None:
    lines = [
        "# 阶段3：kNN与相似性AD的分歧案例",
        "",
        "本分析固定预测结果、校准得到的终点阈值、25%审查比例和预测低关注优先队列。",
        "类别定义在化学品层面：`both_rescued`表示两种方法都审查该基线假阴性，",
        "`knn_only_rescued`和`similarity_only_rescued`表示仅对应方法审查，",
        "`both_omitted`表示两者都留在低优先级队列。统计单位仍包含不同seed/split面板，",
        "因此百分比是描述性证据，不能被解释为独立外部验证次数或因果机制。",
        "",
        "## 1. 基线假阴性中的分歧比例",
        "",
        "下表为每个seed/split先计算比例，再汇总其均值和标准差。",
        "",
        "| 数据集/偏移 | 类别 | 面板数 | 平均化学品数 | 平均占基线假阴性比例 | SD |\n"
        "|---|---|---:|---:|---:|---:|",
    ]
    selected = panel_rates.loc[
        panel_rates.split.isin(PRIMARY_SPLITS) | panel_rates.dataset.eq("external")
    ].copy()
    grouped = (
        selected.groupby(["dataset", "split", "disagreement_category"], as_index=False)
        .agg(
            n_panels=("seed", "nunique"),
            mean_chemicals=("category_chemicals", "mean"),
            mean_fraction=("category_fraction", "mean"),
            sd_fraction=("category_fraction", "std"),
        )
    )
    for row in grouped.itertuples(index=False):
        name = f"{row.dataset}/{row.split}"
        lines.append(
            f"| {name} | {row.disagreement_category} | {int(row.n_panels)} | "
            f"{row.mean_chemicals:.1f} | {row.mean_fraction * 100:.1f}% | "
            f"{row.sd_fraction * 100:.1f}% |"
        )

    lines += [
        "",
        "## 2. 支持信息的描述性对照",
        "",
        "这些均值用于提出后续分层假设，不代表支持信息对救回结果的因果作用。外部记录没有与内部特征块完全相同的距离字段，因此外部只比较可用的排序分数和模型标准差。",
        "",
        "关键对照：在内部物种偏移中，仅kNN找回的基线假阴性平均物种距离为 "
        f"{float(support.loc[(support.dataset == 'internal') & (support.split == 'species') & (support.disagreement_category == 'knn_only_rescued'), 'mean_d_species'].iloc[0]):.3f}，"
        "但这只能说明该类对象在当前距离定义下与训练支持不同，不能说明物种距离是其失效原因。",
        "",
        "## 3. 外部来源分层",
        "",
        "外部分层仅显示新增日本环境省来源与ECHA/PMRA来源中的类别组成；小来源的事件数很少，不据此作优劣判断。",
        "",
        "| 来源 | 类别 | 化学品-重复行数 | 唯一化学品-重复行数 |\n"
        "|---|---|---:|---:|",
    ]
    if source.empty:
        lines.append("| 无可用记录 | | | |")
    else:
        for row in source.itertuples(index=False):
            lines.append(
                f"| {row.source_group} | {row.disagreement_category} | "
                f"{int(row.rows)} | {int(row.unique_chemical_seed_rows)} |"
            )
    lines += [
        "",
        "## 4. 解释边界与下一步",
        "",
        "当前结果支持的最小结论是：不同偏移下，kNN与相似性AD会审查不同的低关注假阴性对象；物种和时间偏移中，仅kNN找回的比例较高，外部新增来源也观察到同类分歧。",
        "这支持把可靠性信号与预期偏移和审查目标匹配，而不支持把kNN解释为普遍优于AD，或把距离差异解释为毒理机制。下一步应使用阶段4的配对不确定性区间，并在需要时对来源、终点和缺失程度进行有限的预先设定分层。",
    ]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_rates = _panel_rates(frame)
    support = _support_summary(frame)
    source = _source_summary(frame)
    panel_rates.to_csv(args.output_dir / "stage3_panel_category_rates.csv", index=False)
    support.to_csv(args.output_dir / "stage3_support_summary.csv", index=False)
    source.to_csv(args.output_dir / "stage3_external_source_summary.csv", index=False)
    _write_report(args.output_dir / "阶段3_分歧案例与环境解释.md", panel_rates, support, source)
    print(panel_rates.to_string(index=False))


if __name__ == "__main__":
    main()
