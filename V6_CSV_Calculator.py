from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import networkx as nx

import fundingutils

ALGORITHM_OPTIONS = [
    "QF",
    "COCM",
    "COCM og",
    "COCM pct_friends",
    "half-and-half",
    "pctCOCM",
    "pairwise",
    "donation_profile_clustermatch",
]
COCM_BASED_ALGORITHMS = {"COCM", "COCM og", "COCM pct_friends", "half-and-half", "pctCOCM"}
ALGORITHM_HELPERS = {
    "QF": "Standard quadratic funding with no cluster-awareness.",
    "COCM": "Cluster-aware matching using the Markov-style COCM similarity.",
    "COCM og": "Original COCM variant using the legacy cluster similarity logic.",
    "COCM pct_friends": "COCM variant based on the fraction of each donor's cluster-friends in each cluster.",
    "half-and-half": "Even blend of normalized COCM and standard QF outcomes.",
    "pctCOCM": "Adjustable blend between COCM and standard QF using the COCM blend slider.",
    "pairwise": "Pairwise matching that rewards breadth across distinct donor pairs.",
    "donation_profile_clustermatch": "Clusters donors by donation profile, then runs cluster matching.",
}


@dataclass(frozen=True)
class RoundParams:
    round_name: str
    matching_pool_usd: float
    matching_cap_percentage: float
    min_donation_threshold_usd: float
    min_passport_score: Optional[float]
    min_model_score: Optional[float]
    engine: str  # One of ALGORITHM_OPTIONS
    token_symbol: str
    token_decimals: Optional[int]
    token_price_usd: Optional[float]

def _guess_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for col_lower, col in cols_lower.items():
            if cand in col_lower:
                return col
    return None


def _coerce_address_series(s: pd.Series) -> pd.Series:
    # keep as strings; lower-case; allow blanks (some exports may not include payout address)
    return s.astype(str).str.strip().str.lower()


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    # Handles "$1,234.56" etc.
    cleaned = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _filter_donations(
    donations_df: pd.DataFrame,
    *,
    min_donation_threshold_usd: float,
    min_passport_score: Optional[float] = None,
    min_model_score: Optional[float] = None,
) -> pd.DataFrame:
    df = donations_df.copy()
    df = df[df["amountUSD"].notna()]
    df = df[df["amountUSD"] > 0]
    if min_donation_threshold_usd > 0:
        df = df[df["amountUSD"] >= float(min_donation_threshold_usd)]

    # Filter by Passport Score threshold if the column exists and a threshold is set.
    if min_passport_score is not None and min_passport_score > 0 and "score" in df.columns:
        df = df[df["score"].fillna(0) >= float(min_passport_score)]

    # Filter by Model Score threshold if the column exists and a threshold is set.
    if min_model_score is not None and min_model_score > 0 and "mbdScore" in df.columns:
        df = df[df["mbdScore"].fillna(0) >= float(min_model_score)]

    # If we have payout addresses, drop self-votes.
    if "recipient_address" in df.columns and df["recipient_address"].astype(str).str.len().gt(0).any():
        df = df[df["voter"] != df["recipient_address"]]

    return df


def _build_canonical_donations_df(
    raw_df: pd.DataFrame,
    *,
    voter_col: str,
    project_name_col: Optional[str],
    amount_col: str,
    project_id_col: Optional[str],
    payout_address_col: Optional[str],
    score_col: Optional[str],
    mbd_score_col: Optional[str],
    treat_amount_as_usd: bool,
    token_price_usd: Optional[float],
) -> pd.DataFrame:
    voter = _coerce_address_series(raw_df[voter_col])
    if project_name_col:
        project_name = raw_df[project_name_col].astype(str).str.strip()
    elif project_id_col:
        project_name = raw_df[project_id_col].astype(str).str.strip()
    else:
        raise ValueError("Provide either a project name column or a project id column.")
    amount = _coerce_numeric_series(raw_df[amount_col])

    if not treat_amount_as_usd:
        if token_price_usd is None or token_price_usd <= 0:
            raise ValueError("Token price (USD) must be provided if amount is not already USD.")
        amount_usd = amount * float(token_price_usd)
    else:
        amount_usd = amount

    out = pd.DataFrame(
        {
            "voter": voter,
            "project_name": project_name,
            "amountUSD": amount_usd,
        }
    )

    if payout_address_col:
        out["recipient_address"] = _coerce_address_series(raw_df[payout_address_col])
    else:
        out["recipient_address"] = ""

    if project_id_col:
        # Keep IDs as strings (v6 often uses UUID-ish ids).
        out["project_id"] = raw_df[project_id_col].astype(str).str.strip()
        out["application_id"] = out["project_id"]
    else:
        # Still provide stable ids for downstream export.
        out["project_id"] = out["project_name"]
        out["application_id"] = out["project_name"]

    # Optional score columns
    if score_col:
        out["score"] = _coerce_numeric_series(raw_df[score_col])
    if mbd_score_col:
        out["mbdScore"] = _coerce_numeric_series(raw_df[mbd_score_col])

    # placeholders (some existing helper logic expects these names to exist)
    out["chain_id"] = 0
    out["round_id"] = ""
    out["rawScore"] = 1.0

    return out


@st.cache_data(show_spinner=False)
def _compute_matching(
    donations_df: pd.DataFrame,
    *,
    engine: str,
    matching_pool_usd: float,
    matching_cap_percentage: float,
    pct_cocm: float = 1.0,
    harsh: bool = True,
) -> pd.DataFrame:
    donation_matrix = fundingutils.pivot_votes(donations_df)
    # Use donation profiles as the "cluster" signal for cluster-dependent algorithms.
    matching_df = fundingutils.get_qf_matching(
        engine,
        donation_matrix,
        matching_cap_percentage,
        matching_pool_usd,
        cluster_df=donation_matrix,
        pct_cocm=float(pct_cocm),
        harsh=harsh,
    )
    # Normalize column naming to "matchedUSD"
    matching_df = matching_df.rename(columns={"matching_amount": "matchedUSD"})
    return matching_df.sort_values("matchedUSD", ascending=False)


def _build_results_export(
    donations_df: pd.DataFrame,
    matching_df: pd.DataFrame,
    params: RoundParams,
) -> pd.DataFrame:
    agg = (
        donations_df.groupby(
            ["application_id", "project_id", "project_name", "recipient_address"], as_index=False
        )
        .agg(totalReceivedUSD=("amountUSD", "sum"), contributionsCount=("voter", "count"))
        .rename(
            columns={
                "application_id": "applicationId",
                "project_id": "projectId",
                "project_name": "projectName",
                "recipient_address": "payoutAddress",
            }
        )
    )
    out = pd.merge(
        agg,
        matching_df[["project_name", "matchedUSD"]].rename(columns={"project_name": "projectName"}),
        on="projectName",
        how="left",
    )
    out["matchedUSD"] = out["matchedUSD"].fillna(0.0).astype(float)

    # Optional token-denominated matched (smallest unit) for systems that require it.
    if params.token_decimals is not None and params.token_price_usd is not None and params.token_price_usd > 0:
        decimals = int(params.token_decimals)
        token_price = float(params.token_price_usd)
        matched_token = (out["matchedUSD"] / token_price).fillna(0) * (10**decimals)
        # floor to int; keep as int64 to avoid scientific notation in CSV
        out["matched"] = np.floor(matched_token).astype("int64")
    else:
        out["matched"] = ""

    out["roundName"] = params.round_name
    out["engine"] = params.engine
    out["matchingPoolUSD"] = float(params.matching_pool_usd)
    out["matchingCapPercentage"] = float(params.matching_cap_percentage)
    out["minDonationThresholdUSD"] = float(params.min_donation_threshold_usd)
    out["tokenSymbol"] = params.token_symbol

    return out[
        [
            "roundName",
            "engine",
            "matchingPoolUSD",
            "matchingCapPercentage",
            "minDonationThresholdUSD",
            "tokenSymbol",
            "applicationId",
            "projectId",
            "projectName",
            "payoutAddress",
            "totalReceivedUSD",
            "contributionsCount",
            "matchedUSD",
            "matched",
        ]
    ].sort_values("matchedUSD", ascending=False)


def _chart_matching_histogram(results_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        results_df,
        x="matchedUSD",
        nbins=30,
        title="Matching distribution (USD)",
        labels={"matchedUSD": "Matched (USD)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def _chart_top_projects(results_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    df = results_df.head(int(top_n)).copy()
    df = df.sort_values("matchedUSD", ascending=True)
    fig = px.bar(
        df,
        x="matchedUSD",
        y="projectName",
        orientation="h",
        title=f"Top {len(df)} projects by matching (USD)",
        labels={"matchedUSD": "Matched (USD)", "projectName": "Project"},
    )
    fig.update_layout(template="plotly_white", height=600)
    return fig


def _chart_donor_amount_distribution(donations_df: pd.DataFrame) -> go.Figure:
    by_voter = donations_df.groupby("voter", as_index=False).agg(totalUSD=("amountUSD", "sum"))
    bin_edges = [0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, np.inf]
    bin_labels = ["0-1", "1-2", "2-3", "3-5", "5-10", "10-20", "20-50", "50-100", "100-500", "500-1000", "1000+"]
    by_voter["bin"] = pd.cut(by_voter["totalUSD"], bins=bin_edges, labels=bin_labels, right=False)
    fig = px.histogram(
        by_voter,
        x="bin",
        category_orders={"bin": bin_labels},
        title="Donor total contribution distribution (USD)",
        labels={"bin": "Total donated (USD bin)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def _chart_connection_graph(
    donations_df: pd.DataFrame,
    *,
    max_edges: int = 1500,
    max_donors: int = 800,
    seed: int = 42,
) -> go.Figure:
    """
    Lightweight 2D bipartite graph of donors ↔ projects.
    We aggressively sample to avoid freezing on large exports.
    """
    grouped = (
        donations_df.groupby(["voter", "project_name"], as_index=False)
        .agg(amountUSD=("amountUSD", "sum"))
    )

    # Keep only top donors (by total USD) to control node count
    donor_totals = grouped.groupby("voter", as_index=False).agg(totalUSD=("amountUSD", "sum"))
    keep_donors = set(donor_totals.sort_values("totalUSD", ascending=False).head(int(max_donors))["voter"])
    grouped = grouped[grouped["voter"].isin(keep_donors)]

    # Sample edges (weighted by donation size)
    if len(grouped) > int(max_edges):
        weights = grouped["amountUSD"].clip(lower=0).to_numpy()
        if weights.sum() > 0:
            probs = weights / weights.sum()
            grouped = grouped.sample(n=int(max_edges), random_state=seed, replace=False, weights=probs)
        else:
            grouped = grouped.sample(n=int(max_edges), random_state=seed, replace=False)

    donors = grouped["voter"].unique().tolist()
    projects = grouped["project_name"].unique().tolist()

    B = nx.Graph()
    B.add_nodes_from(donors, bipartite=0, kind="donor")
    B.add_nodes_from(projects, bipartite=1, kind="project")
    for _, r in grouped.iterrows():
        B.add_edge(r["voter"], r["project_name"], amountUSD=float(r["amountUSD"]))

    # 2D layout is much faster/less noisy than 3D here
    pos = nx.spring_layout(B, seed=seed, k=0.25, iterations=60)

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in B.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(120,120,120,0.35)"),
        hoverinfo="none",
    )

    # Node traces
    donor_x, donor_y, donor_text = [], [], []
    for d in donors:
        if d in pos:
            x, y = pos[d]
            donor_x.append(x)
            donor_y.append(y)
            donor_text.append(f"Donor: {d[:6]}…{d[-4:]}")

    project_x, project_y, project_text = [], [], []
    for p in projects:
        if p in pos:
            x, y = pos[p]
            project_x.append(x)
            project_y.append(y)
            project_text.append(f"Project: {p}")

    donor_trace = go.Scatter(
        x=donor_x,
        y=donor_y,
        mode="markers",
        name="Donors",
        marker=dict(size=6, color="#C4F092", line=dict(width=0)),
        text=donor_text,
        hoverinfo="text",
    )
    project_trace = go.Scatter(
        x=project_x,
        y=project_y,
        mode="markers",
        name="Projects",
        marker=dict(size=10, color="#00433B", line=dict(width=0)),
        text=project_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, donor_trace, project_trace])
    fig.update_layout(
        title="Connection graph (sampled donors/edges)",
        showlegend=True,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="v6 QF Calculator (CSV)", page_icon="🧮", layout="wide")
    st.title("v6 QF Calculator (CSV Upload)")
    st.caption("Internal analysis tool: read-only + compute-only. Upload a v6 donation export CSV, compute matching, visualize, and download results.")

    uploaded = st.file_uploader("Upload v6 QF Donation Export CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    raw_df = pd.read_csv(uploaded)
    st.subheader("CSV preview")
    st.dataframe(raw_df.head(50), use_container_width=True)

    st.subheader("Map CSV columns")
    default_voter = _guess_col(raw_df, ["voter", "donor", "sender", "from", "address"])
    default_project_name = _guess_col(raw_df, ["project_name", "project name", "project", "title", "name"])
    default_amount = _guess_col(raw_df, ["amountusd", "amount_usd", "usd", "amount"])
    default_project_id = _guess_col(raw_df, ["project_id", "projectid", "application_id", "applicationid", "id"])
    default_payout = _guess_col(raw_df, ["recipient_address", "payout", "to", "recipient", "project_address"])
    default_score = _guess_col(raw_df, ["score", "passport_score", "passportscore", "passport score"])
    default_mbd_score = _guess_col(raw_df, ["mbdscore", "mbd_score", "model_score", "modelscore", "model score"])

    c1, c2, c3 = st.columns(3)
    with c1:
        voter_col = st.selectbox("Donor address column", options=list(raw_df.columns), index=(list(raw_df.columns).index(default_voter) if default_voter in raw_df.columns else 0))
        payout_col = st.selectbox("Payout address column (optional)", options=["(none)"] + list(raw_df.columns), index=(1 + list(raw_df.columns).index(default_payout) if default_payout in raw_df.columns else 0))
    with c2:
        project_name_col = st.selectbox(
            "Project name column (optional if you have project id)",
            options=["(none)"] + list(raw_df.columns),
            index=(1 + list(raw_df.columns).index(default_project_name) if default_project_name in raw_df.columns else 0),
        )
        project_id_col = st.selectbox("Project ID column (optional)", options=["(none)"] + list(raw_df.columns), index=(1 + list(raw_df.columns).index(default_project_id) if default_project_id in raw_df.columns else 0))
    with c3:
        amount_col = st.selectbox("Donation amount column", options=list(raw_df.columns), index=(list(raw_df.columns).index(default_amount) if default_amount in raw_df.columns else 0))
        score_col = st.selectbox("Passport Score column (optional)", options=["(none)"] + list(raw_df.columns), index=(1 + list(raw_df.columns).index(default_score) if default_score in raw_df.columns else 0))
        mbd_score_col = st.selectbox("Model Score column (optional)", options=["(none)"] + list(raw_df.columns), index=(1 + list(raw_df.columns).index(default_mbd_score) if default_mbd_score in raw_df.columns else 0))

    st.subheader("Round parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        round_name = st.text_input("Round name", value="Uploaded round")
        engine = st.selectbox("Algorithm", options=ALGORITHM_OPTIONS, index=0)
        pct_cocm = st.slider(
            "COCM blend for pctCOCM",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            disabled=engine != "pctCOCM",
            help="Only used when algorithm is pctCOCM. 1.0 = pure COCM, 0.0 = pure QF.",
        )
        harsh = st.toggle(
            "Harsh (COCM)",
            value=True,
            disabled=engine not in COCM_BASED_ALGORITHMS,
            help="Used by COCM-based algorithms (`COCM`, `COCM og`, `COCM pct_friends`, and COCM/QF blends).",
        )
    with c2:
        matching_pool_usd = st.number_input("Matching pool (USD)", min_value=0.0, value=50000.0, step=1000.0)
        matching_cap_percentage = st.slider("Matching cap (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with c3:
        min_donation_threshold_usd = st.number_input("Min donation threshold (USD)", min_value=0.0, value=0.0, step=1.0)
        token_symbol = st.text_input("Token symbol (for export metadata)", value="USD")
    with c4:
        treat_amount_as_usd = st.toggle("Donation amounts are already USD", value=True)
        token_price_usd = None
        token_decimals = None
        if not treat_amount_as_usd:
            token_price_usd = st.number_input("Token price (USD)", min_value=0.0, value=1.0, step=0.01)
            token_decimals = st.number_input("Token decimals", min_value=0, max_value=36, value=18, step=1)
    with c5:
        _has_score_col = score_col != "(none)"
        _has_mbd_col = mbd_score_col != "(none)"
        min_passport_score = st.number_input(
            "Min Passport Score",
            min_value=0.0, value=0.0, step=1.0,
            disabled=not _has_score_col,
            help="Donations with a Passport Score below this threshold are excluded. Requires a Passport Score column.",
        )
        min_model_score = st.number_input(
            "Min Model Score",
            min_value=0.0, value=0.0, step=1.0,
            disabled=not _has_mbd_col,
            help="Donations with a Model Score below this threshold are excluded. Requires a Model Score column.",
        )
        if not _has_score_col:
            min_passport_score = None
        if not _has_mbd_col:
            min_model_score = None

    st.caption(f"Selected algorithm: `{engine}` - {ALGORITHM_HELPERS[engine]}")
    with st.expander("Algorithm quick guide", expanded=False):
        for algo in ALGORITHM_OPTIONS:
            st.markdown(f"- `{algo}`: {ALGORITHM_HELPERS[algo]}")

    params = RoundParams(
        round_name=round_name.strip() or "Uploaded round",
        matching_pool_usd=float(matching_pool_usd),
        matching_cap_percentage=float(matching_cap_percentage),
        min_donation_threshold_usd=float(min_donation_threshold_usd),
        min_passport_score=(float(min_passport_score) if min_passport_score is not None else None),
        min_model_score=(float(min_model_score) if min_model_score is not None else None),
        engine=engine,
        token_symbol=token_symbol.strip() or "USD",
        token_decimals=(int(token_decimals) if token_decimals is not None else None),
        token_price_usd=(float(token_price_usd) if token_price_usd is not None else None),
    )

    compute = st.button("Run matching", type="primary")
    if not compute:
        return

    try:
        donations_df = _build_canonical_donations_df(
            raw_df,
            voter_col=voter_col,
            project_name_col=None if project_name_col == "(none)" else project_name_col,
            amount_col=amount_col,
            project_id_col=None if project_id_col == "(none)" else project_id_col,
            payout_address_col=None if payout_col == "(none)" else payout_col,
            score_col=None if score_col == "(none)" else score_col,
            mbd_score_col=None if mbd_score_col == "(none)" else mbd_score_col,
            treat_amount_as_usd=treat_amount_as_usd,
            token_price_usd=params.token_price_usd,
        )
    except Exception as e:
        st.error(f"Failed to parse donations from CSV: {e}")
        return

    donations_df = _filter_donations(
        donations_df,
        min_donation_threshold_usd=params.min_donation_threshold_usd,
        min_passport_score=params.min_passport_score,
        min_model_score=params.min_model_score,
    )
    if donations_df.empty:
        st.warning("No eligible donations found after filtering.")
        return

    st.subheader("Computed stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Donations (rows)", f"{len(donations_df):,}")
    c2.metric("Unique donors", f"{donations_df['voter'].nunique():,}")
    c3.metric("Unique projects", f"{donations_df['project_name'].nunique():,}")
    c4.metric("Total donated (USD)", f"${donations_df['amountUSD'].sum():,.2f}")

    with st.spinner("Computing matching…"):
        matching_df = _compute_matching(
            donations_df,
            engine=params.engine,
            matching_pool_usd=params.matching_pool_usd,
            matching_cap_percentage=params.matching_cap_percentage,
            pct_cocm=(float(pct_cocm) if params.engine == "pctCOCM" else 1.0),
            harsh=harsh,
        )

    results_df = _build_results_export(donations_df, matching_df, params)

    st.subheader("Visualizations")
    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(_chart_matching_histogram(results_df), use_container_width=True)
        st.plotly_chart(_chart_donor_amount_distribution(donations_df), use_container_width=True)
    with v2:
        st.plotly_chart(_chart_top_projects(results_df, top_n=30), use_container_width=True)

    with st.expander("Connection graph (sampled)", expanded=False):
        st.caption("This is a sampled donor↔project graph to help spot clusters/coordination patterns without freezing.")
        c1, c2 = st.columns(2)
        with c1:
            max_edges = st.slider("Max edges", min_value=200, max_value=3000, value=1500, step=100)
        with c2:
            max_donors = st.slider("Max donors", min_value=100, max_value=2000, value=800, step=50)
        st.plotly_chart(
            _chart_connection_graph(donations_df, max_edges=int(max_edges), max_donors=int(max_donors)),
            use_container_width=True,
        )

    st.subheader("Results table")
    st.dataframe(
        results_df,
        use_container_width=True,
        column_config={
            "totalReceivedUSD": st.column_config.NumberColumn("Total received (USD)", format="$%.2f"),
            "matchedUSD": st.column_config.NumberColumn("Matched (USD)", format="$%.2f"),
        },
    )

    st.subheader("Download results")
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download calculated matching (CSV)",
        data=csv_bytes,
        file_name="v6_matching_results.csv",
        mime="text/csv",
    )

    st.caption("No writes are made back to v6. This tool only reads your CSV and computes results in-browser.")


if __name__ == "__main__":
    main()

