# Build & Export (.pbit) in ~3 minutes
1) Power BI Desktop ▶ **Transform Data**.
2) Home ▶ **New Source** ▶ **Blank Query** ▶ Advanced Editor:
   - Paste `queries/Long_fixed.m` → Name it **Long**.
   - Paste `queries/Wide_fixed.m` → Name it **Wide**.
3) Close & Apply.
4) Model view: set **Long[date_iso]** to **Date**.
5) Create measures (Model ▶ Long ▶ New measure) and paste **all** from `measures/DAX.txt`.
6) Report:
   - Page *Trends*: 2 slicers (`section`, `test`), 1 line chart (Axis: date_iso, Values: [Value], Legend: test), 2 KPI cards ([Latest Value], [Delta %]).
   - Page *Matrix*: Rows=section,test; Values=[Value]; Columns=date_iso (ή Year) — ή χρησιμοποίησε το Wide.
7) View ▶ Themes ▶ Browse ▶ `theme.json`.
8) File ▶ Export ▶ **Power BI template (.pbit)**.
