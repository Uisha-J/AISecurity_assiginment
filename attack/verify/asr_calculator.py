"""Attack Success Rate (ASR) computation."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_asr(scores_df, group_by=None):
    if group_by is None:
        group_by = ["duration", "threshold"]
    asr_df = (
        scores_df.groupby(group_by)
        .agg(total=("verified", "count"), successful=("verified", "sum"),
             mean_score=("similarity_score", "mean"), std_score=("similarity_score", "std"))
        .reset_index()
    )
    asr_df["asr"] = asr_df["successful"] / asr_df["total"]
    return asr_df


def compute_asr_summary(scores_df):
    summary = {}
    for t in scores_df["threshold"].unique():
        summary[f"asr_threshold_{t:.2f}"] = scores_df[scores_df["threshold"] == t]["verified"].mean()
    for d in scores_df["duration"].unique():
        sub = scores_df[scores_df["duration"] == d]
        summary[f"asr_duration_{d}s"] = sub["verified"].mean()
        summary[f"mean_score_duration_{d}s"] = sub["similarity_score"].mean()
    spk = scores_df.groupby("speaker_id")["verified"].mean().describe()
    summary["asr_per_speaker_mean"] = spk["mean"]
    summary["asr_per_speaker_std"] = spk["std"]
    summary["similarity_mean"] = scores_df["similarity_score"].mean()
    summary["similarity_std"] = scores_df["similarity_score"].std()
    return summary


def save_asr_report(scores_df, output_path):
    asr = compute_asr(scores_df, ["duration", "threshold"])
    asr.to_csv(output_path, index=False)
    asr.pivot(index="duration", columns="threshold", values="asr").to_csv(
        output_path.replace(".csv", "_pivot.csv"))
    logger.info(f"ASR report saved to: {output_path}")
