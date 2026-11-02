"""Maintenance Log Analyzer - Streamlit app.

Run with:  streamlit run app.py
"""

import streamlit as st

import hashlib

import ai_report as ai
import analyzer as an

# Blank template users can download, fill in, and upload
TEMPLATE_CSV = (
    "date,machine_id,failure_type,downtime_hours,repair_cost,technician\n"
    "2026-03-14,CNC-01,Spindle bearing failure,6.5,4500,Ravi\n"
    "2026-03-20,PUMP-02,Seal leakage,3.0,1200,Anita\n"
)

AUTO = "Auto (recommended)"


def load_models(provider, api_key, base_url):
    """Ask the provider which models this key can use. Cached per key for this session."""
    cache = st.session_state.setdefault("model_cache", {})
    sig = (provider, base_url, hashlib.sha256(api_key.encode()).hexdigest())
    if sig not in cache:
        try:
            cache[sig] = (ai.list_models(provider, api_key, base_url), None)
        except Exception as e:  # noqa: BLE001
            cache[sig] = ([], str(e))
    return cache[sig]


st.set_page_config(page_title="Maintenance Log Analyzer", page_icon="🔧", layout="wide")

st.title("🔧 Maintenance Log Analyzer")
st.caption(
    "Upload a machine breakdown log to find what fails most, which machines cost the most "
    "downtime, and where to act first."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload maintenance log (CSV)", type="csv")
    use_sample = st.checkbox("Use sample data", value=uploaded is None)

    with st.expander("What format should my CSV be?"):
        st.markdown(
            "One row per breakdown. The first row must contain these column names:\n\n"
            "| Column | Example |\n"
            "|---|---|\n"
            "| `date` | 2026-03-14 (YYYY-MM-DD) |\n"
            "| `machine_id` | CNC-01 |\n"
            "| `failure_type` | Bearing failure |\n"
            "| `downtime_hours` | 6.5 (number only) |\n"
            "| `repair_cost` | 4500 (number only) |\n"
            "| `technician` | Ravi *(optional)* |\n\n"
            "Tip: export from Excel using **Save As > CSV**."
        )
        st.download_button(
            "Download blank template",
            TEMPLATE_CSV,
            file_name="maintenance_log_template.csv",
            mime="text/csv",
        )

    st.header("Optional: AI report")
    st.caption("Everything else works without a key. Use any provider you already have.")
    provider = st.selectbox("AI provider", list(ai.PROVIDERS))
    api_key = st.text_input(
        "API key",
        type="password",
        help="Leave empty to use the built-in rule-based recommendations.",
    )
    base_url = None
    if provider == "OpenAI or compatible":
        base_url = st.text_input(
            "Base URL",
            value=ai.PROVIDERS[provider]["base_url"],
            help="Change this for OpenAI-compatible services, e.g. https://api.groq.com/openai/v1",
        )

    model = ai.PROVIDERS[provider]["model"]  # used only if the live model list is unavailable
    if api_key:
        available, load_error = load_models(provider, api_key, base_url)
        if available:
            choice = st.selectbox(
                "Model",
                [AUTO] + available,
                key=f"model_choice_{provider}",
                help="Auto picks a recent, fast model from the list your key can use.",
            )
            model = ai.pick_default(provider, available) if choice == AUTO else choice
            if choice == AUTO:
                st.caption(f"Auto picked: {model}")
        else:
            st.caption(f"Could not load the model list ({load_error}). Type a model name instead.")
            model = st.text_input("Model name", value=model, key=f"model_{provider}")
            if st.button("Reload model list"):
                st.session_state.pop("model_cache", None)
                st.rerun()

source = uploaded if uploaded is not None else ("sample_data.csv" if use_sample else None)

if source is None:
    st.info("Upload a CSV or tick 'Use sample data' to begin.")
    st.stop()

try:
    df = an.load_log(source)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the file: {e}")
    st.stop()

if df.empty:
    st.warning("No valid rows found in the file.")
    st.stop()

# Optional filter
machines = sorted(df["machine_id"].unique())
selected = st.sidebar.multiselect("Filter machines", machines, default=machines)
df = df[df["machine_id"].isin(selected)]
if df.empty:
    st.warning("Select at least one machine.")
    st.stop()

# Headline metrics
m = an.summary_metrics(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total failures", m["total_failures"])
c2.metric("Total downtime (h)", f"{m['total_downtime_hours']:.0f}")
c3.metric("Repair cost", f"{m['total_repair_cost']:,.0f}")
c4.metric("Avg downtime / failure (h)", f"{m['avg_downtime_per_failure']:.1f}")

st.subheader("Where to act first")
for rec in an.rule_based_recommendations(df):
    st.markdown(f"- {rec}")

left, right = st.columns(2)
with left:
    st.subheader("Downtime by failure type")
    by_type = an.failures_by_type(df)
    st.bar_chart(by_type.set_index("failure_type")["downtime_hours"])
    st.dataframe(by_type, width="stretch", hide_index=True)

with right:
    st.subheader("Downtime by machine")
    ranking = an.machine_ranking(df)
    st.bar_chart(ranking.set_index("machine_id")["downtime_hours"])
    st.dataframe(ranking, width="stretch", hide_index=True)

st.subheader("Monthly trend")
trend = an.monthly_trend(df).set_index("month")
st.line_chart(trend[["failures", "downtime_hours"]])

st.subheader("Repeat failures (root-cause candidates)")
repeats = an.repeat_failures(df)
if len(repeats):
    st.dataframe(repeats, width="stretch", hide_index=True)
else:
    st.write("No machine/failure combination repeats 3 or more times.")

# Optional AI report
st.subheader("AI reliability report")
prompt = an.build_llm_prompt(df)
data_signature = hashlib.sha256(prompt.encode()).hexdigest()

if not api_key:
    st.caption("Optional: choose a provider and add your API key in the sidebar to get an AI-written report.")
elif st.button("Generate AI report"):
    try:
        with st.spinner("Analysing..."):
            text = ai.generate_report(provider, api_key, model, prompt, base_url)
        # Keep the report in session state so it survives the rerun a download click triggers
        st.session_state["ai_result"] = {
            "text": text, "sig": data_signature, "provider": provider, "model": model,
        }
    except Exception as e:  # noqa: BLE001
        st.error(f"AI report failed: {e}")

result = st.session_state.get("ai_result")
if result and result["sig"] == data_signature:
    st.markdown(result["text"])
    report_md = ai.build_markdown(m, result["text"], result["provider"], result["model"])
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button(
        "Download report (.md)",
        report_md,
        file_name="maintenance_ai_report.md",
        mime="text/markdown",
    )
    report_html = ai.to_html(report_md)
    if report_html:
        d2.download_button(
            "Download report (.html)",
            report_html,
            file_name="maintenance_ai_report.html",
            mime="text/html",
            help="Open in a browser, then use Print > Save as PDF.",
        )
elif result:
    st.info("The data or machine filter changed since the last report. Click 'Generate AI report' for an updated one.")
