"""Founder check-in desk for SaaS accounts.

Run: streamlit run app.py
Optional: set OPENAI_API_KEY to generate LLM-written drafts.
"""

import csv
import io
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Founder Check-ins", page_icon="✦", layout="wide")

REQUIRED_COLUMNS = {
    "Name",
    "Plan",
    "MRR",
    "Last Login Days Ago",
    "Payment Status",
}


def inject_tailwind() -> None:
    """Use Tailwind's CDN plus a small Streamlit-specific design layer."""
    st.markdown(
        """
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
          .stApp { background: #f8fafc; color: #0f172a; }
          #MainMenu, footer, header { visibility: hidden; }
          .block-container { max-width: 1280px; padding-top: 2.25rem; padding-bottom: 3rem; }
          .hero { background: linear-gradient(125deg, #111827 0%, #312e81 52%, #4f46e5 100%);
                  border-radius: 24px; padding: 2.5rem; color: white; box-shadow: 0 20px 45px rgba(49,46,129,.22); }
          .hero p { color: #c7d2fe; margin: .5rem 0 0; font-size: 1.05rem; }
          .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 800; color: #a5b4fc; }
          .metric-card { background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.1rem 1.2rem; min-height: 112px; }
          .metric-label { color: #64748b; font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
          .metric-value { color: #0f172a; font-size: 1.8rem; font-weight: 800; margin-top: .15rem; }
          .risk-pill { display: inline-block; background: #fff1f2; color: #be123c; border: 1px solid #fecdd3; border-radius: 999px; padding: .22rem .65rem; font-size: .78rem; font-weight: 800; }
          .approved-pill { display: inline-block; background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; border-radius: 999px; padding: .22rem .65rem; font-size: .78rem; font-weight: 800; }
          .stButton button { border-radius: 10px; border: 0; font-weight: 700; padding: .55rem 1rem; }
          .stButton button[kind="primary"] { background: #4f46e5; }
          [data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(sorted(missing)))
    df["MRR"] = pd.to_numeric(df["MRR"], errors="coerce").fillna(0)
    df["Last Login Days Ago"] = pd.to_numeric(
        df["Last Login Days Ago"], errors="coerce"
    ).fillna(0).astype(int)
    df["Payment Status"] = df["Payment Status"].astype(str).str.strip()
    df["Payment status normalized"] = df["Payment Status"].str.lower()
    return df


def risk_reason(row: pd.Series) -> str:
    reasons = []
    if row["Last Login Days Ago"] >= 14:
        reasons.append(f"no login in {row['Last Login Days Ago']} days")
    if row["Payment status normalized"] not in {"paid", "active", "current"}:
        reasons.append(f"payment status is {row['Payment Status']}")
    return "; ".join(reasons)


def founder_prompt(row: pd.Series, founder_name: str) -> str:
    return f"""Write a brief, thoughtful check-in email from {founder_name}, founder of a B2B SaaS company, to the customer account below.

Customer account: {row['Name']}
Plan: {row['Plan']}
Monthly recurring revenue: ${row['MRR']:,.0f}
Last login: {row['Last Login Days Ago']} days ago
Payment status: {row['Payment Status']}
Reason for reaching out: {row['Risk reason']}

Requirements:
- Sound genuinely human, warm, and concise (90-140 words).
- Do not mention scoring, "at risk," automation, or internal data.
- If payment is not current, acknowledge there may have been an issue and offer help without pressure.
- Ask one open-ended question about how the team is getting value from the product.
- Offer a short call with the founder.
- Include a specific subject line on the first line as: Subject: ...
- End with a simple signature from {founder_name}.
"""


def fallback_draft(row: pd.Series, founder_name: str) -> str:
    payment_note = ""
    if row["Payment status normalized"] not in {"paid", "active", "current"}:
        payment_note = (
            " I also wanted to make sure there isn't anything we can help untangle "
            "on the account or billing side."
        )
    return (
        f"Subject: A quick check-in from {founder_name}\n\n"
        f"Hi {row['Name']} team,\n\n"
        f"I wanted to personally check in and see how things are going with us. "
        f"We haven't seen your team in the product for a little while, and I want to make sure "
        f"you're getting the value you expected from the {row['Plan']} plan.{payment_note}\n\n"
        "What would make the product more useful for your team right now? If it would help, "
        f"I'm happy to jump on a quick call and listen.\n\n"
        f"Best,\n{founder_name}"
    )


def llm_draft(row: pd.Series, founder_name: str, model: str) -> tuple[str, str]:
    """Return (draft, source). Falls back locally when no API key is configured."""
    # Prefer an environment variable; fall back to Streamlit secrets when configured.
    # Accessing st.secrets without a secrets.toml file can raise an exception,
    # so keep that lookup guarded and allow the local starter draft to work.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = str(st.secrets["OPENAI_API_KEY"]).strip()
        except Exception:
            # Covers a missing secrets.toml file as well as a missing key.
            api_key = ""

    if not api_key:
        return (
            fallback_draft(row, founder_name),
            "Starter draft (add OPENAI_API_KEY for LLM drafts)",
        )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=founder_prompt(row, founder_name),
        )
        return response.output_text.strip(), f"Generated with {model}"
    except Exception as error:  # Let the founder still finish outreach during outages.
        return fallback_draft(row, founder_name), f"Starter draft (LLM unavailable: {error})"


def csv_download(approved: dict) -> str:
    output = io.StringIO()
    fields = ["Account", "Plan", "MRR", "Risk reason", "Approved at (UTC)", "Email draft"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(approved.values())
    return output.getvalue()


inject_tailwind()
if "approved" not in st.session_state:
    st.session_state.approved = {}
if "drafts" not in st.session_state:
    st.session_state.drafts = {}

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Founder workspace</div>
      <h1 style="font-size:2.2rem; font-weight:800; margin:.35rem 0 0;">Customer check-ins, with the human touch intact.</h1>
      <p>Upload your SaaS customer database, focus on accounts that need attention, and approve a personal outreach draft in one click.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")
left, right = st.columns([2.1, 1])
with left:
    uploaded = st.file_uploader("Upload customer database (.csv)", type="csv")
with right:
    founder_name = st.text_input("Founder name", value="Alex")
    model = st.text_input("LLM model", value="gpt-4.1-mini", help="Used only when OPENAI_API_KEY is configured.")

if uploaded is None:
    st.info("Upload a CSV with: Name, Plan, MRR, Last Login Days Ago, Payment Status.")
    st.stop()

try:
    data = normalize_data(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

data["Risk reason"] = data.apply(risk_reason, axis=1)
risk = data[data["Risk reason"] != ""].copy()
risk = risk.sort_values(["MRR", "Last Login Days Ago"], ascending=[False, False])

total_mrr = data["MRR"].sum()
risk_mrr = risk["MRR"].sum()
failed_payment = risk[~risk["Payment status normalized"].isin(["paid", "active", "current"])].shape[0]
metrics = st.columns(4)
for column, label, value in zip(
    metrics,
    ["Accounts", "At-risk accounts", "MRR at risk", "Payment issues"],
    [f"{len(data):,}", f"{len(risk):,}", f"${risk_mrr:,.0f}", f"{failed_payment:,}"],
):
    column.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("At-risk accounts")
st.caption("Flagged when the account has not logged in for 14+ days or has a payment status other than Paid, Active, or Current.")

if risk.empty:
    st.success("No at-risk accounts found. Nice work.")
    st.stop()

display = risk[["Name", "Plan", "MRR", "Last Login Days Ago", "Payment Status", "Risk reason"]].copy()
display["MRR"] = display["MRR"].map(lambda amount: f"${amount:,.0f}")
st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()
account_names = risk["Name"].tolist()
selected_name = st.selectbox("Choose an account to review", account_names)
account = risk[risk["Name"] == selected_name].iloc[0]

info, action = st.columns([1.2, 2])
with info:
    st.markdown('<span class="risk-pill">Needs attention</span>', unsafe_allow_html=True)
    st.markdown(f"### {account['Name']}")
    st.write(f"**{account['Plan']}** plan · **${account['MRR']:,.0f} MRR**")
    st.write(f"Reason: {account['Risk reason']}")
with action:
    key = account["Name"]
    if st.button("Draft check-in email", type="primary", use_container_width=True):
        with st.spinner("Writing a founder-ready draft..."):
            draft, source = llm_draft(account, founder_name, model)
            st.session_state.drafts[key] = {"text": draft, "source": source}

if selected_name in st.session_state.drafts:
    draft_data = st.session_state.drafts[selected_name]
    st.caption(draft_data["source"])
    edited = st.text_area(
        "Review and personalize before approving",
        value=draft_data["text"],
        height=290,
        key=f"editor_{selected_name}",
    )
    approve_col, status_col = st.columns([1, 2])
    with approve_col:
        if st.button("Approve email", type="primary", use_container_width=True):
            st.session_state.approved[selected_name] = {
                "Account": account["Name"],
                "Plan": account["Plan"],
                "MRR": account["MRR"],
                "Risk reason": account["Risk reason"],
                "Approved at (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "Email draft": edited,
            }
    with status_col:
        if selected_name in st.session_state.approved:
            st.markdown('<span class="approved-pill">Approved and ready to export</span>', unsafe_allow_html=True)

if st.session_state.approved:
    st.divider()
    st.subheader(f"Approved outreach ({len(st.session_state.approved)})")
    st.download_button(
        "Download approved emails as CSV",
        data=csv_download(st.session_state.approved),
        file_name="approved_founder_checkins.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption(f"Portfolio MRR: ${total_mrr:,.0f} · At-risk MRR: ${risk_mrr:,.0f}")
