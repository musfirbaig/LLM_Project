"""
Excel Workbook Inspector
-------------------------
Iterates over every sheet in the NUST Bank knowledge-base Excel file and
prints a diagnostic summary: shape, column names, data types, missing-value
counts, unique-value counts, and preview rows.
"""

import os
import pandas as pd

WORKBOOK = os.path.join(os.path.dirname(__file__), "..", "NUST Bank-Product-Knowledge.xlsx")


def inspect_sheet(xls: pd.ExcelFile, name: str):
    """Print a diagnostic block for a single sheet."""
    df = pd.read_excel(xls, sheet_name=name)

    separator = "-" * 76
    print(f"\n{separator}")
    print(f"Sheet: {name}")
    print(separator)
    print(f"  Dimensions : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns    : {list(df.columns)}\n")

    print("  Column types:")
    for col in df.columns:
        print(f"    {col:40s} => {df[col].dtype}")

    print("\n  Missing values:")
    for col in df.columns:
        nulls = df[col].isna().sum()
        ratio = nulls / len(df) * 100 if len(df) else 0
        print(f"    {col:40s} => {nulls:>5}  ({ratio:.1f}%)")

    print("\n  Distinct values:")
    for col in df.columns:
        print(f"    {col:40s} => {df[col].nunique()}")

    display_opts = {"display.max_columns": None, "display.width": 200, "display.max_colwidth": 60}
    with pd.option_context(*[item for pair in display_opts.items() for item in pair]):
        print("\n  Head (5 rows):")
        print(df.head().to_string(index=False))
        print("\n  Tail (2 rows):")
        print(df.tail(2).to_string(index=False))


def main():
    xls = pd.ExcelFile(WORKBOOK, engine="openpyxl")

    banner = "=" * 76
    print(banner)
    print(f"Workbook  : {os.path.basename(WORKBOOK)}")
    print(f"Sheets    : {len(xls.sheet_names)}")
    print(f"Names     : {xls.sheet_names}")
    print(banner)

    for sheet_name in xls.sheet_names:
        inspect_sheet(xls, sheet_name)

    print(f"\n{banner}")
    print("Inspection finished.")
    print(banner)


if __name__ == "__main__":
    main()
