let
    Source = Csv.Document(
        File.Contents("C\\temp\\comprehensive_wide_powerbi.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Cols = Table.ColumnNames(PromotedHeaders),
    DateCols = List.Select(Cols, each Text.Length(_) = 10 and Text.Middle(_,4,1) = "-" and Text.Middle(_,7,1) = "-"),
    Types1 = Table.TransformColumnTypes(PromotedHeaders, {{"section", type text}, {"test", type text}, {"units", type text}}),
    Types2 = Table.TransformColumnTypes(Types1, List.Transform(DateCols, each {_, type number}))
in
    Types2
