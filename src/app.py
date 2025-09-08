from __future__ import annotations

import io
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.data_utils import (
        filter_df,
        load_csv,
        normalize_columns,
        summarize_names_range,
        summarize_registration_range,
    )
    from src import charts as C
except ModuleNotFoundError:
    # Fallback when running the file directly via `streamlit run src/app.py`
    from data_utils import (  # type: ignore
        filter_df,
        load_csv,
        normalize_columns,
        summarize_names_range,
        summarize_registration_range,
    )
    import charts as C  # type: ignore

st.set_page_config(page_title="CGPA Explorer", layout="wide")
alt.data_transformers.disable_max_rows()


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    try:
        with open("data.csv", "rb") as f:
            return load_csv(f)
    except Exception:
        # Empty frame with columns
        return normalize_columns(pd.DataFrame(columns=[
            "Email", "Full Name", "UG Mark", "Campus",
            "Registration Number", "Department", "Degree And Specialization"
        ]))


def sidebar_data_source() -> pd.DataFrame:
    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
            st.sidebar.success(f"Loaded {len(df)} rows from upload")
            return df
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
    df = load_default_data()
    if df.empty:
        st.sidebar.warning("No data loaded. Place data.csv in project root or upload a file.")
    else:
        st.sidebar.info(f"Using default dataset with {len(df)} rows")
    return df


def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")
    name_q = st.sidebar.text_input("Search by name contains")
    email_q = st.sidebar.text_input("Search by email contains")
    reg_q = st.sidebar.text_input("Search by registration no.")

    def multiselect_or_all(label: str, col: str):
        opts = sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip()])
        return st.sidebar.multiselect(label, options=opts, default=[])

    campus = multiselect_or_all("Campus", "campus")
    dept = multiselect_or_all("Department", "department")
    degree = multiselect_or_all("Degree/Specialization", "degree_specialization")

    # Numeric ranges
    cgpa_min = float(np.nanmin(df["cgpa"])) if not df["cgpa"].dropna().empty else 0.0
    cgpa_max = float(np.nanmax(df["cgpa"])) if not df["cgpa"].dropna().empty else 10.0

    cgpa_rng = st.sidebar.slider("CGPA range", min_value=0.0, max_value=10.0, value=(cgpa_min, cgpa_max), step=0.1)
    only_cgpa = st.sidebar.checkbox("Only rows with CGPA", value=False)
    only_email = st.sidebar.checkbox("Only rows with email", value=False)

    return {
        "name_query": name_q,
        "email_query": email_q,
    "registration_query": reg_q,
        "campus": campus,
        "department": dept,
        "degree_specialization": degree,
        "cgpa_range": cgpa_rng,
    "only_with_cgpa": only_cgpa,
    "only_with_email": only_email,
    }


def _analyzer_visuals(matched: pd.DataFrame):
    """Show charts for a matched subset: CGPA histogram, campus and branch distribution."""
    if matched.empty:
        st.info("No matched records to visualize.")
        return
    st.markdown("#### Visualizations")
    v1, v2, v3 = st.tabs(["CGPA Histogram", "Campus Distribution", "Branch Distribution"])
    with v1:
        st.altair_chart(C.hist_numeric(matched, "cgpa", title="CGPA of matched"), use_container_width=True)
    with v2:
        st.altair_chart(C.bar_count(matched, "campus", title="Students per Campus"), use_container_width=True)
    with v3:
        st.altair_chart(C.bar_count(matched, "department", title="Students per Department"), use_container_width=True)


def names_uploader_panel(df: pd.DataFrame):
    st.subheader("Names list analyzer")
    st.caption("Upload a plain text file or CSV with a column of names. We'll compute CGPA range for those found.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload .txt or .csv of names", type=["txt", "csv"], key="names_upload")
    with col2:
        sample = st.checkbox("Show sample format help")

    if sample:
        st.info("""
        Accepted formats:
        - Plain text: one full name per line
        - CSV: must contain a column named 'Full Name' or 'full_name'. Other columns ignored.
        Names are matched case-insensitively to the dataset's 'Full Name'.
        """)

    names: List[str] = []
    error = None
    if uploaded is not None:
        try:
            if uploaded.type == "text/plain" or uploaded.name.lower().endswith(".txt"):
                content = uploaded.getvalue().decode("utf-8", errors="ignore")
                names = [line.strip() for line in content.splitlines() if line.strip()]
            else:
                ndf = pd.read_csv(uploaded)
                col = None
                for c in ["full_name", "Full Name", "name", "Name"]:
                    if c in ndf.columns:
                        col = c
                        break
                if col is None:
                    raise ValueError("CSV must include a 'Full Name' column")
                names = ndf[col].astype(str).tolist()
        except Exception as e:
            error = str(e)

    st.write("Or paste names (one per line):")
    names_text = st.text_area("Names textarea", key="names_textarea", height=120)
    if names_text.strip():
        pasted = [line.strip() for line in names_text.splitlines() if line.strip()]
        # Merge pasted with uploaded if both provided
        names = names + pasted if names else pasted

    if names:
        result = summarize_names_range(df, names)
        st.success(f"Found {result['count_found']} of {len(names)} names in dataset")
        cols = st.columns(4)
        cols[0].metric("CGPA min", result["cgpa_min"])
        cols[1].metric("CGPA max", result["cgpa_max"])
        # Show matched subset and allow download
        matched = df[df["name_lower"].isin([n.strip().lower() for n in names if str(n).strip()])]
        _analyzer_visuals(matched)
        with st.expander("Matched records", expanded=False):
            st.dataframe(matched[[
                "full_name", "email", "campus", "department", "degree_specialization",
                "cgpa", "registration_number"
            ]], use_container_width=True, height=300)
            st.download_button(
                "Download matched records",
                data=matched.to_csv(index=False).encode("utf-8"),
                file_name="matched_records.csv",
                mime="text/csv",
            )
        if result["missing_names"]:
            with st.expander("Missing names"):
                st.write("\n".join(result["missing_names"]))
    elif error:
        st.error(f"Could not process names: {error}")


def registration_uploader_panel(df: pd.DataFrame):
    st.subheader("Registration numbers list analyzer")
    st.caption("Upload a plain text file or CSV with a column of registration numbers. We'll compute CGPA range for those found.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload .txt or .csv of registration numbers", type=["txt", "csv"], key="reg_upload")
    with col2:
        sample = st.checkbox("Show sample format help (reg)")

    if sample:
        st.info("""
        Accepted formats:
        - Plain text: one registration number per line
        - CSV: must contain a column named 'Registration Number' or 'registration_number'. Other columns ignored.
        Matching is case-insensitive and whitespace-insensitive.
        """)

    regs: List[str] = []
    error = None
    if uploaded is not None:
        try:
            if uploaded.type == "text/plain" or uploaded.name.lower().endswith(".txt"):
                content = uploaded.getvalue().decode("utf-8", errors="ignore")
                regs = [line.strip() for line in content.splitlines() if line.strip()]
            else:
                ndf = pd.read_csv(uploaded)
                col = None
                for c in ["registration_number", "Registration Number", "reg_no", "Reg No", "RegNo"]:
                    if c in ndf.columns:
                        col = c
                        break
                if col is None:
                    raise ValueError("CSV must include a 'Registration Number' column")
                regs = ndf[col].astype(str).tolist()
        except Exception as e:
            error = str(e)

    st.write("Or paste registration numbers (one per line):")
    regs_text = st.text_area("Registration numbers textarea", key="reg_textarea", height=120)
    if regs_text.strip():
        pasted = [line.strip() for line in regs_text.splitlines() if line.strip()]
        regs = regs + pasted if regs else pasted

    if regs:
        result = summarize_registration_range(df, regs)
        st.success(f"Found {result['count_found']} of {len(regs)} registration numbers in dataset")
        cols = st.columns(4)
        cols[0].metric("CGPA min", result["cgpa_min"])
        cols[1].metric("CGPA max", result["cgpa_max"])
        reg_norm = [r.strip().lower() for r in regs if str(r).strip()]
        matched = df[df["registration_number"].astype(str).str.strip().str.lower().isin(reg_norm)]
        _analyzer_visuals(matched)
        with st.expander("Matched records", expanded=False):
            st.dataframe(matched[[
                "full_name", "email", "campus", "department", "degree_specialization",
                "cgpa", "registration_number"
            ]], use_container_width=True, height=300)
            st.download_button(
                "Download matched records (by reg)",
                data=matched.to_csv(index=False).encode("utf-8"),
                file_name="matched_by_registration.csv",
                mime="text/csv",
            )
        if result.get("missing_numbers"):
            with st.expander("Missing registration numbers"):
                st.write("\n".join(result["missing_numbers"]))
    elif error:
        st.error(f"Could not process registration numbers: {error}")


# Main App

def main():
    st.title("CGPA Explorer")
    st.caption("Explore and filter the student dataset. All processing is local in your browser session.")

    df = sidebar_data_source()

    filters = sidebar_filters(df)
    filtered = filter_df(df, **filters)

    st.markdown("### Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(filtered))
    c2.metric("Unique campuses", filtered["campus"].nunique())
    c3.metric("Unique departments", filtered["department"].nunique())
    c4.metric("With CGPA", int(filtered["cgpa"].notna().sum()))

    # Data preview and download
    with st.expander("Data preview", expanded=True):
        st.dataframe(
            filtered[[
                "full_name", "email", "campus", "department", "degree_specialization",
                "cgpa", "registration_number"
            ]], use_container_width=True, height=400
        )
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered.csv", mime="text/csv")

    st.markdown("### Charts")
    tab1, tab2, tab3, tab4 = st.tabs(["CGPA Histogram", "CGPA by Dept (Box)", "Campus Counts", "Dept x Campus Heatmap"])
    with tab1:
        st.altair_chart(C.hist_numeric(filtered, "cgpa", title="CGPA"), use_container_width=True)
    with tab2:
        st.altair_chart(C.box_numeric(filtered, "cgpa", by="department", title="CGPA by Department"), use_container_width=True)
    with tab3:
        st.altair_chart(C.bar_count(filtered, "campus", title="Students per Campus"), use_container_width=True)
    with tab4:
        # Aggregate average CGPA per dept x campus
        agg = filtered.groupby(["department", "campus"], dropna=False).agg(avg_cgpa=("cgpa", "mean")).reset_index()
        agg["avg_cgpa"] = agg["avg_cgpa"].round(3)
        st.altair_chart(C.heatmap(agg, x="department", y="campus", value="avg_cgpa", title="Avg CGPA Heatmap"), use_container_width=True)

    st.divider()
    st.markdown("### List Analyzer")
    a1, a2 = st.tabs(["By Names", "By Registration No."])
    with a1:
        names_uploader_panel(df)
    with a2:
        registration_uploader_panel(df)


if __name__ == "__main__":
    main()
