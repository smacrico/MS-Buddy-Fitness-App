# Power BI Template Pack (Fixed Paths)

Pre-wired to your CSV locations:
- Long: `c:\\temp\\comprehensive_long_powerbi.csv`
- Wide: `c:\\temp\\comprehensive_wide_powerbi.csv`

Open **Power BI Desktop** → Transform Data → New Blank Query → Advanced Editor → paste `queries/Long_fixed.m` and `queries/Wide_fixed.m`, Close & Apply.
Then import measures from `measures/DAX.txt`, apply `theme.json`, build visuals per `report_layout_instructions.md`, and export as `.pbit`.
