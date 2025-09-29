# Steps to build the PBIX and export as PBIT

> Estimated time: 3–5 minutes.

## 1) Create Parameters (file paths)
In Power BI Desktop:
1. Home ▶ **Transform Data** ▶ Power Query Editor.
2. Home ▶ **Manage Parameters** ▶ **New Parameter** (create both):
   - **LongCsvPath** (Type: Text) → set it to the full path of `comprehensive_long_powerbi.csv`.
   - **WideCsvPath** (Type: Text) → set it to the full path of `comprehensive_wide_powerbi.csv`.

## 2) Add the Queries (copy–paste M)
In Power Query Editor:
- Home ▶ **New Source** ▶ **Blank Query** ▶ Advanced Editor ▶ paste the content of `queries/Long.m` and click **Done**.
  - Rename the query to **Long**.
- Repeat for `queries/Wide.m` and rename to **Wide**.
- **Close & Apply**.

## 3) Model & Formatting
- Ensure **Long[date_iso]** is type **Date** (Model view).
- Optionally hide the **Wide** table if you prefer to focus on Long.
- Create the measures below (Model view ▶ Table: Long ▶ New measure) from `measures/DAX.txt`.

## 4) Build the Report Pages

### Page: *Trends*
- **Slicers**: `section`, `test` (orientation: vertical); optionally a date slicer on `date_iso`.
- **Line chart**: Axis = `date_iso`, Values = **[Value]**, Legend = `test`.
- **KPI cards** (x2): add cards bound to **[Latest Value]** and **[Delta %]**; filter the card visual to a specific `test` (e.g., *HGB*, *CRP*).

### Page: *Matrix*
- **Matrix**: Rows = `section`, `test`; Values = **[Value]**; Columns = **Year** or **date_iso** (as hierarchy) — or use **Wide** if you prefer pivoted columns.

## 5) Theme
- View ▶ **Themes** ▶ **Browse for themes** ▶ pick `theme.json` from this pack.

## 6) Export Template
- File ▶ **Export** ▶ **Power BI template** (.pbit) → Done.

Tip: To refresh in the future, open the `.pbit`, point it to new CSV paths, and publish.
