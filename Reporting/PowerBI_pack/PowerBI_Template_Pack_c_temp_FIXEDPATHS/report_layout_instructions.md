# Build & Export (.pbit) in ~2 minutes (fixed paths)
1) Power BI Desktop ▶ **Transform Data**.
2) Home ▶ **New Source** ▶ **Blank Query** ▶ Advanced Editor:
   - Paste `queries/Long_fixed.m` → Name it **Long**.
   - Paste `queries/Wide_fixed.m` → Name it **Wide**.
3) Close & Apply.
4) Model view: set **Long[date_iso]** to **Date**.
5) Create measures (Model ▶ Long ▶ New measure) and paste **all** from `measures/DAX.txt`.
6) View ▶ Themes ▶ Browse ▶ `theme.json`.
7) File ▶ Export ▶ **Power BI template (.pbit)**.
