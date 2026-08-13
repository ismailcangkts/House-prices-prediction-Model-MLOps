import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(here, "analysis_tables.json"), "utf8"));
const previewDir = path.join(here, "previews");
const outputPath = path.join(here, "SUNUM_GRAFIKLERI_VE_TABLOLARI.xlsx");
await fs.mkdir(previewDir, { recursive: true });

const COLORS = {
  dark: "#152022",
  green: "#1F7A63",
  blue: "#3C6E8F",
  orange: "#D87941",
  paper: "#F4F0E8",
  pale: "#DCE6DF",
  light: "#EAE5DA",
  white: "#FFFFFF",
  muted: "#66706C",
};

const workbook = Workbook.create();

function matrixFromRecords(records) {
  if (!records.length) return [[]];
  const headers = Object.keys(records[0]);
  return [headers, ...records.map((row) => headers.map((header) => row[header] ?? null))];
}

function styleTitle(sheet, title, subtitle, endCol = "H") {
  sheet.showGridLines = false;
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: COLORS.dark,
    font: { bold: true, color: COLORS.white, size: 18 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 34;
  sheet.mergeCells(`A2:${endCol}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${endCol}2`).format = {
    fill: COLORS.pale,
    font: { color: COLORS.muted, italic: true, size: 10 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${endCol}2`).format.rowHeight = 25;
}

function styleTable(sheet, rangeAddress, tableName) {
  const range = sheet.getRange(rangeAddress);
  range.format = {
    font: { size: 10, color: COLORS.dark },
    borders: {
      top: { color: "#D6D1C7", style: "thin" },
      bottom: { color: "#D6D1C7", style: "thin" },
      left: { color: "#E6E1D8", style: "thin" },
      right: { color: "#E6E1D8", style: "thin" },
    },
  };
  const table = sheet.tables.add(rangeAddress, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  sheet.freezePanes.freezeRows(4);
  return table;
}

function addRecordsSheet(name, title, subtitle, records, tableName, widths = {}) {
  const sheet = workbook.worksheets.add(name);
  const matrix = matrixFromRecords(records);
  const cols = matrix[0].length;
  const endCol = String.fromCharCode(64 + Math.min(cols, 26));
  styleTitle(sheet, title, subtitle, endCol);
  const target = sheet.getRangeByIndexes(3, 0, matrix.length, cols);
  target.values = matrix;
  styleTable(sheet, `A4:${endCol}${matrix.length + 3}`, tableName);
  for (let col = 0; col < cols; col += 1) {
    const header = matrix[0][col];
    const width = widths[header] ?? (String(header).length > 20 ? 24 : 16);
    sheet.getRangeByIndexes(3, col, matrix.length, 1).format.columnWidth = width;
  }
  sheet.getRange(`A4:${endCol}${matrix.length + 3}`).format.rowHeight = 20;
  return sheet;
}

const summary = workbook.worksheets.add("Ozet");
styleTitle(summary, "Sunum Grafik ve Tablo Paketi", "Ames House Prices · gerçek proje verilerinden üretilmiş sunum kaynak tabloları", "N");
summary.getRange("A4:B4").values = [["Veri Özeti", "Değer"]];
const summaryRows = data.dataset_summary.map((row) => [row.Metrik, row.Değer]);
summary.getRangeByIndexes(4, 0, summaryRows.length, 2).values = summaryRows;
styleTable(summary, `A4:B${summaryRows.length + 4}`, "DatasetSummaryTable");
summary.getRange("A:B").format.columnWidth = 22;
summary.getRange("B5:B20").format.numberFormat = "#,##0";

summary.getRange("D4:F4").values = [["Model", "Validation RMSE ($K)", "Test RMSE ($K)"]];
summary.getRange("D5:F8").values = data.model_comparison.map((row) => [row.Model, row["Validation RMSE"] / 1000, row["Test RMSE"] / 1000]);
summary.getRange("D4:F8").format = { fill: COLORS.light, font: { color: COLORS.dark, size: 9 } };
summary.getRange("D4:F4").format = { fill: COLORS.blue, font: { bold: true, color: COLORS.white } };
summary.getRange("D:F").format.columnWidth = 20;
summary.getRange("E5:F8").format.numberFormat = "$0.0\"K\"";
const modelChart = summary.charts.add("bar", summary.getRange("D4:F8"));
modelChart.title = "Model RMSE Karşılaştırması";
modelChart.hasLegend = true;
modelChart.xAxis = { axisType: "textAxis" };
modelChart.yAxis = { numberFormatCode: "$0.0\"K\"" };
modelChart.setPosition("H4", "N15");

summary.getRange("D11:F11").values = [["Yöntem", "Validation RMSE %", "Test RMSE %"]];
summary.getRange("D12:F18").values = data.feature_engineering_comparison.map((row) => [row.method, row.validation_rmse_improvement_pct, row.test_rmse_improvement_pct]);
summary.getRange("D11:F18").format = { fill: COLORS.light, font: { color: COLORS.dark, size: 9 } };
summary.getRange("D11:F11").format = { fill: COLORS.green, font: { bold: true, color: COLORS.white } };
summary.getRange("E12:F18").format.numberFormat = "+0.00\"%\";-0.00\"%\"";
const feChart = summary.charts.add("bar", summary.getRange("D11:F18"));
feChart.title = "Feature Engineering RMSE Etkisi";
feChart.hasLegend = true;
feChart.xAxis = { axisType: "textAxis" };
feChart.yAxis = { numberFormatCode: "+0.0\"%\";-0.0\"%\"" };
feChart.setPosition("H17", "N30");

summary.getRange("A14:B18").values = [
  ["Sunumda vurgulanacaklar", ""],
  ["1", "SalePrice skewness: 1.88 → 0.12 (log1p)"],
  ["2", "OverallQual ve GrLivArea ana fiyat sinyalleri"],
  ["3", "7 feature grubu birlikte validation + test kazanmadı"],
  ["4", "Gradient Boosting testte bozuldu; lineer modeller kararlı"],
];
summary.getRange("A14:B14").format = { fill: COLORS.dark, font: { bold: true, color: COLORS.white } };
summary.getRange("A15:A18").format = { fill: COLORS.orange, font: { bold: true, color: COLORS.white }, horizontalAlignment: "center" };
summary.getRange("B15:B18").format = { fill: COLORS.pale, font: { color: COLORS.dark }, wrapText: true };
summary.getRange("B14:B18").format.columnWidth = 46;
summary.getRange("A14:B18").format.rowHeight = 25;

const correlations = addRecordsSheet("Korelasyonlar", "Korelasyon Analizi", "SalePrice ilişkileri ve multicollinearity adayları", data.top_correlations, "CorrelationsTable", { Feature: 24, Korelasyon: 16 });
correlations.getRange("B5:B40").format.numberFormat = "0.000";
correlations.getRange("D4:F4").values = [["Feature 1", "Feature 2", "Korelasyon"]];
correlations.getRangeByIndexes(4, 3, data.multicollinearity_pairs.length, 3).values = data.multicollinearity_pairs.map((row) => [row["Feature 1"], row["Feature 2"], row.Korelasyon]);
styleTable(correlations, `D4:F${data.multicollinearity_pairs.length + 4}`, "MulticollinearityTable");
correlations.getRange("D:F").format.columnWidth = 22;
correlations.getRange("F5:F30").format.numberFormat = "0.000";

const outlier = addRecordsSheet("Outlier", "Outlier Analizi", "1.5×IQR sınırları ve işaretlenen gözlem sayıları", data.outlier_summary, "OutlierTable", { Feature: 20, "Alt Sınır": 18, "Üst Sınır": 18, "Outlier Adedi": 16, "Outlier Oranı %": 18 });
outlier.getRange("B5:C30").format.numberFormat = "#,##0.0";
outlier.getRange("D5:D30").format.numberFormat = "#,##0";
outlier.getRange("E5:E30").format.numberFormat = "0.0\"%\"";

const neighborhood = addRecordsSheet("Mahalleler", "Mahalle Bazlı Fiyat Özeti", "Medyan sırasına göre Neighborhood istatistikleri", data.neighborhood_summary, "NeighborhoodTable", { Neighborhood: 22, count: 12, mean: 18, median: 18, std: 18, min: 18, max: 18 });
neighborhood.getRange("C5:G50").format.numberFormat = "$#,##0";

const featureEng = addRecordsSheet("Feature_Eng", "Feature Engineering Deneyleri", "Validation ve bağımsız test sonuçları; pozitif yüzde hata azalmasıdır", data.feature_engineering_comparison, "FeatureEngineeringTable", { method: 25, features_added: 60, decision: 34 });
featureEng.getRange("F5:M50").format.numberFormat = "0.00";

const models = addRecordsSheet("Modeller", "Model Karşılaştırması", "Aynı preprocessing ile validation ve bağımsız test benchmark'ı", data.model_comparison, "ModelComparisonTable", { Model: 24, "RMSE Genelleme Farkı": 24 });
models.getRange("B5:C20").format.numberFormat = "$#,##0";
models.getRange("D5:D20").format.numberFormat = "0.000";
models.getRange("E5:E20").format.numberFormat = "0.000";
models.getRange("F5:G20").format.numberFormat = "$#,##0";
models.getRange("H5:H20").format.numberFormat = "0.000";
models.getRange("I5:I20").format.numberFormat = "0.000";
models.getRange("J5:J20").format.numberFormat = "$#,##0";

const residual = addRecordsSheet("Hata_Analizi", "Fiyat Bandına Göre Ridge Hatası", "Validation residual özeti; bias = actual − prediction", data.residual_by_price_band, "ResidualBandsTable", { "Fiyat Bandı": 20, MAE: 18, Bias: 18, "Gözlem": 14 });
residual.getRange("B5:C20").format.numberFormat = "$#,##0";
residual.getRange("D5:D20").format.numberFormat = "#,##0";

const features = addRecordsSheet("Feature_Listesi", "Seçili 30 Feature", "Domain grubu ve veri tipi ile model girdisi sözlüğü", data.feature_list, "FeatureListTable", { Grup: 20, Feature: 28, "Veri Tipi": 18 });

const target = addRecordsSheet("Hedef_Ozeti", "SalePrice İstatistik Özeti", "Ham hedef dağılımı ve log dönüşüm göstergeleri", data.target_summary, "TargetSummaryTable", { Metrik: 24, "Değer": 22 });
target.getRange("B5:B20").format.numberFormat = "#,##0.000";

const semantic = addRecordsSheet("Temizlik", "Semantik Temizlik Özeti", "Temiz veri içindeki anlamlı yokluk kategorileri", data.semantic_cleaning_counts, "SemanticCleaningTable", { Durum: 26, Adet: 16 });
semantic.getRange("B5:B20").format.numberFormat = "#,##0";

const sheetNames = ["Ozet", "Korelasyonlar", "Outlier", "Mahalleler", "Feature_Eng", "Modeller", "Hata_Analizi", "Feature_Listesi", "Hedef_Ozeti", "Temizlik"];
for (const sheetName of sheetNames) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({ kind: "table", range: "Ozet!A1:N30", include: "values,formulas", tableMaxRows: 30, tableMaxCols: 14, maxChars: 8000 });
await fs.writeFile(path.join(previewDir, "inspection.ndjson"), inspection.ndjson, "utf8");
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
await fs.writeFile(path.join(previewDir, "formula-errors.ndjson"), errors.ndjson, "utf8");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: sheetNames.length, previews: sheetNames.length, inspectionChars: inspection.ndjson.length, errorScan: errors.ndjson }, null, 2));
