from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    mapping = {
        "Email": "email",
        "Full Name": "full_name",
        "UG Mark": "ug_mark",
        "Campus": "campus",
        "Registration Number": "registration_number",
        "Department": "department",
        "Degree And Specialization": "degree_specialization",
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    # Ensure expected columns exist
    for col in mapping.values():
        if col not in df.columns:
            df[col] = pd.Series([np.nan] * len(df), dtype="object")

    df["cgpa"] = pd.to_numeric(df["ug_mark"], errors="coerce")

    # Clean strings
    for c in [
        "email",
        "full_name",
        "campus",
        "registration_number",
        "department",
        "degree_specialization",
    ]:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype("string")
                .fillna("")
                .str.strip()
            )

    # Derive: name tokens for search
    df["name_lower"] = df["full_name"].str.lower()

    return df


def load_csv(path_or_buffer: str | io.StringIO | io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    return normalize_columns(df)


def filter_df(
    df: pd.DataFrame,
    *,
    name_query: str = "",
    email_query: str = "",
    registration_query: str = "",
    campus: Optional[Iterable[str]] = None,
    department: Optional[Iterable[str]] = None,
    degree_specialization: Optional[Iterable[str]] = None,
    cgpa_range: Optional[Tuple[float, float]] = None,
    only_with_cgpa: bool = False,
    only_with_email: bool = False,
) -> pd.DataFrame:
    """Apply filters to the DataFrame and return a view (copy)."""
    mask = pd.Series(True, index=df.index)

    if name_query:
        q = name_query.strip().lower()
        mask &= df["name_lower"].str.contains(re.escape(q), na=False)

    if email_query:
        q = email_query.strip().lower()
        mask &= df["email"].str.lower().str.contains(re.escape(q), na=False)

    if registration_query:
        q = registration_query.strip().lower()
        mask &= df["registration_number"].str.lower().str.contains(re.escape(q), na=False)

    def in_opt(col: str, values: Optional[Iterable[str]]):
        nonlocal mask
        if values:
            vset = {v.lower() for v in values}
            mask &= df[col].str.lower().isin(vset)

    in_opt("campus", campus)
    in_opt("department", department)
    in_opt("degree_specialization", degree_specialization)

    if cgpa_range:
        lo, hi = cgpa_range
        if lo is not None:
            mask &= df["cgpa"] >= lo
        if hi is not None:
            mask &= df["cgpa"] <= hi

    if only_with_cgpa:
        mask &= df["cgpa"].notna()

    if only_with_email:
        mask &= df["email"].astype(str).str.strip() != ""

    return df[mask].copy()


def summarize_names_range(df: pd.DataFrame, names: Iterable[str]) -> dict:
    """Compute CGPA range for the provided names list.

    Returns dict with keys: count_found, missing_names, cgpa_min, cgpa_max.
    """
    names = [n.strip().lower() for n in names if str(n).strip()]
    if not names:
        return {
            "count_found": 0,
            "missing_names": [],
            "cgpa_min": None,
            "cgpa_max": None,
        }

    sub = df[df["name_lower"].isin(names)]
    missing = sorted(set(names) - set(sub["name_lower"].tolist()))

    cgpa_min = float(sub["cgpa"].min()) if not sub["cgpa"].dropna().empty else None
    cgpa_max = float(sub["cgpa"].max()) if not sub["cgpa"].dropna().empty else None

    return {
        "count_found": int(len(sub)),
        "missing_names": missing,
        "cgpa_min": None if cgpa_min is None or math.isnan(cgpa_min) else round(cgpa_min, 3),
        "cgpa_max": None if cgpa_max is None or math.isnan(cgpa_max) else round(cgpa_max, 3),
    }


def summarize_registration_range(df: pd.DataFrame, reg_numbers: Iterable[str]) -> dict:
    """Compute CGPA range for the provided registration numbers list.

    Returns dict with keys: count_found, missing_numbers, cgpa_min, cgpa_max.
    """
    regs = [str(n).strip().lower() for n in reg_numbers if str(n).strip()]
    if not regs:
        return {
            "count_found": 0,
            "missing_numbers": [],
            "cgpa_min": None,
            "cgpa_max": None,
        }

    reg_series = df["registration_number"].astype(str).str.strip().str.lower()
    match_mask = reg_series.isin(regs)
    sub = df[match_mask]
    matched_set = set(reg_series[match_mask].tolist())
    missing = sorted(set(regs) - matched_set)

    cgpa_min = float(sub["cgpa"].min()) if not sub["cgpa"].dropna().empty else None
    cgpa_max = float(sub["cgpa"].max()) if not sub["cgpa"].dropna().empty else None

    return {
        "count_found": int(len(sub)),
        "missing_numbers": missing,
        "cgpa_min": None if cgpa_min is None or math.isnan(cgpa_min) else round(cgpa_min, 3),
        "cgpa_max": None if cgpa_max is None or math.isnan(cgpa_max) else round(cgpa_max, 3),
    }
