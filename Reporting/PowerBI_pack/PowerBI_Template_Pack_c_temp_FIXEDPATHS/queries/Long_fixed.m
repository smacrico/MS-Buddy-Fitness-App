let
    Source = Csv.Document(
        File.Contents("C\\temp\\comprehensive_long_powerbi.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Trimmed = Table.TransformColumns(PromotedHeaders, List.Transform(Table.ColumnNames(PromotedHeaders), each {_, Text.Trim, type text})),
    ChangedTypes = Table.TransformColumnTypes(Trimmed, {{"date_iso", type date}, {"section", type text}, {"test", type text}, {"value", type number}, {"units", type text}})
in
    ChangedTypes
