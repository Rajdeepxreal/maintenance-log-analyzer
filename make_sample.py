"""Generates sample_data.csv, a realistic fake maintenance log for demo purposes."""
import random
import pandas as pd

random.seed(42)
machines = {
    "CNC-01": ["Spindle bearing failure", "Coolant leak", "Tool breakage"],
    "CNC-02": ["Tool breakage", "Servo fault", "Coolant leak"],
    "PUMP-01": ["Seal leakage", "Bearing failure", "Cavitation"],
    "PUMP-02": ["Seal leakage", "Motor overheating"],
    "CONV-01": ["Belt slippage", "Motor overheating", "Roller jam"],
    "PRESS-01": ["Hydraulic leak", "Valve failure", "Sensor fault"],
}
# Weighted so there are clear hot spots to discover
weights = {
    ("CNC-01", "Spindle bearing failure"): 9,
    ("PUMP-01", "Seal leakage"): 8,
    ("PRESS-01", "Hydraulic leak"): 6,
}
base = {"Spindle bearing failure": (14, 9000), "Seal leakage": (6, 2500), "Hydraulic leak": (9, 4000),
        "Servo fault": (8, 5000), "Motor overheating": (7, 3500), "Tool breakage": (3, 1200),
        "Coolant leak": (4, 1500), "Bearing failure": (10, 4500), "Cavitation": (8, 3000),
        "Belt slippage": (3, 800), "Roller jam": (2, 600), "Valve failure": (7, 3200), "Sensor fault": (2, 700)}
techs = ["Ravi", "Anita", "Suresh", "Meena"]

rows = []
for m, fails in machines.items():
    for f in fails:
        n = weights.get((m, f), random.randint(2, 4))
        for _ in range(n):
            d = pd.Timestamp("2025-11-01") + pd.Timedelta(days=random.randint(0, 360))
            hrs, cost = base[f]
            rows.append({
                "date": d.strftime("%Y-%m-%d"), "machine_id": m, "failure_type": f,
                "downtime_hours": round(max(0.5, random.gauss(hrs, hrs * 0.3)), 1),
                "repair_cost": int(max(100, random.gauss(cost, cost * 0.25))),
                "technician": random.choice(techs),
            })
pd.DataFrame(rows).sort_values("date").to_csv("sample_data.csv", index=False)
print(len(rows), "rows written")
