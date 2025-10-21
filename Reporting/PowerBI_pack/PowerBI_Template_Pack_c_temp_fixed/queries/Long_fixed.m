let
    Source = Csv.Document(
        File.Contents("c:\\temp\\comprehensive_long_powerbi.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Trimmed = Table.TransformColumns(PromotedHeaders, List.Transform(Table.ColumnNames(PromotedHeaders), each { {0} , Text.Trim, type text})),
    ChangedTypes = Table.TransformColumnTypes(Trimmed, { {1} } )
in
    ChangedTypes