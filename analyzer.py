"""Core analysis logic for the Maintenance Log Analyzer.

Expected CSV columns:
    date            - date of the breakdown (YYYY-MM-DD)
    machine_id      - e.g. CNC-01, PUMP-03
    failure_type    - e.g. Bearing failure, Overheating
    downtime_hours  - hours the machine was down
    repair_cost     - cost of the repair (any currency)
Optional:
    technician      - who did the repair
"""

import pandas as pd

REQUIRED_COLUMNS = ["date", "machine_id", "failure_type", "downtime_hours", "repair_cost"]


def load_log(file) -> pd.DataFrame:
    """Read and validate a maintenance log CSV."""
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["downtime_hours"] = pd.to_numeric(df["downtime_hours"], errors="coerce")
    df["repair_cost"] = pd.to_numeric(df["repair_cost"], errors="coerce")
    df = df.dropna(subset=["date", "machine_id", "failure_type", "downtime_hours"])
    df["repair_cost"] = df["repair_cost"].fillna(0)
    return df.sort_values("date").reset_index(drop=True)


def summary_metrics(df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard."""
    return {
        "total_failures": int(len(df)),
        "total_downtime_hours": float(df["downtime_hours"].sum()),
        "total_repair_cost": float(df["repair_cost"].sum()),
        "avg_downtime_per_failure": float(df["downtime_hours"].mean()),
        "machines": int(df["machine_id"].nunique()),
    }


def failures_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Which failure types happen most and cost the most downtime."""
    out = (
        df.groupby("failure_type")
        .agg(
            count=("failure_type", "size"),
            downtime_hours=("downtime_hours", "sum"),
            repair_cost=("repair_cost", "sum"),
        )
        .sort_values("downtime_hours", ascending=False)
    )
    out["share_of_downtime_%"] = (out["downtime_hours"] / out["downtime_hours"].sum() * 100).round(1)
    return out.reset_index()


def machine_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank machines by downtime, with MTBF (mean time between failures, in days)."""
    rows = []
    for machine, g in df.groupby("machine_id"):
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna()
        rows.append(
            {
                "machine_id": machine,
                "failures": len(g),
                "downtime_hours": g["downtime_hours"].sum(),
                "repair_cost": g["repair_cost"].sum(),
                "mtbf_days": round(gaps.mean(), 1) if len(gaps) else None,
                "avg_repair_hours": round(g["downtime_hours"].mean(), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("downtime_hours", ascending=False).reset_index(drop=True)


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Failures and downtime per month."""
    m = df.copy()
    m["month"] = m["date"].dt.to_period("M").astype(str)
    return (
        m.groupby("month")
        .agg(failures=("month", "size"), downtime_hours=("downtime_hours", "sum"))
        .reset_index()
    )


def repeat_failures(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    """Machine + failure-type pairs that keep recurring: the best candidates for a root-cause fix."""
    out = (
        df.groupby(["machine_id", "failure_type"])
        .agg(
            occurrences=("failure_type", "size"),
            downtime_hours=("downtime_hours", "sum"),
            repair_cost=("repair_cost", "sum"),
        )
        .reset_index()
    )
    out = out[out["occurrences"] >= min_count]
    return out.sort_values(["occurrences", "downtime_hours"], ascending=False).reset_index(drop=True)


def rule_based_recommendations(df: pd.DataFrame) -> list[str]:
    """Plain-Python recommendations that work with no API key."""
    recs = []
    types = failures_by_type(df)
    top = types.iloc[0]
    recs.append(
        f"'{top['failure_type']}' causes {top['share_of_downtime_%']}% of all downtime "
        f"({top['downtime_hours']:.0f} h). Start the root-cause analysis here."
    )

    machines = machine_ranking(df)
    worst = machines.iloc[0]
    recs.append(
        f"{worst['machine_id']} is the worst performer: {int(worst['failures'])} failures and "
        f"{worst['downtime_hours']:.0f} h of downtime. Review its maintenance schedule and condition."
    )

    repeats = repeat_failures(df)
    if len(repeats):
        r = repeats.iloc[0]
        recs.append(
            f"{r['machine_id']} had '{r['failure_type']}' {int(r['occurrences'])} times. "
            "A repeat failure usually means the repair treats the symptom, not the cause. "
            "Consider a 5-Why or fishbone analysis."
        )

    mtbf = machines.dropna(subset=["mtbf_days"])
    if len(mtbf):
        low = mtbf.sort_values("mtbf_days").iloc[0]
        recs.append(
            f"{low['machine_id']} has the lowest MTBF ({low['mtbf_days']} days). "
            "Shorten its preventive-maintenance interval until the failure rate drops."
        )
    return recs


def build_llm_prompt(df: pd.DataFrame) -> str:
    """Prompt for the optional AI summary: sends only aggregated tables, not raw logs."""
    m = summary_metrics(df)
    return (
        "You are a maintenance reliability engineer. Based on the summary below, write a short "
        "report: (1) the 3 biggest problems, (2) likely root causes for each, (3) specific "
        "preventive actions, ordered by expected impact. Be concrete and concise.\n\n"
        f"Headline metrics: {m}\n\n"
        f"Failures by type:\n{failures_by_type(df).to_string(index=False)}\n\n"
        f"Machine ranking:\n{machine_ranking(df).to_string(index=False)}\n\n"
        f"Repeat failures:\n{repeat_failures(df).to_string(index=False)}\n"
    )
