# Sunumda Kullanılacak Görsel Paketi

## Önerilen anlatım sırası

1. **Problem ve veri kapsamı** — `01_dataset_overview`: 1.451 gözlemi ve 30 feature'lık kontrollü model alanını tanıtın.
2. **Veri bölme disiplini** — `02_data_split`: train/validation/test/reference/production rollerini ve leakage riskini anlatın.
3. **Data cleaning** — `03_semantic_cleaning`: NaN değerlerinin her zaman gerçek eksik olmadığını; bazılarının “garaj yok” gibi fiziksel anlam taşıdığını söyleyin.
4. **Hedef değişken** — `04_saleprice_distribution` ve `05_saleprice_boxplot`: sağa çarpıklığı, log1p kararını ve uzun fiyat kuyruğunu açıklayın.
5. **Outlier analizi** — `06_grlivarea_saleprice_outliers` ve `07_outlier_counts`: outlier bayrağının otomatik silme kararı olmadığını vurgulayın.
6. **Korelasyonlar** — `08_top_correlations` ve `09_correlation_heatmap`: OverallQual ve GrLivArea'nın güçlü olduğunu; korelasyonun nedensellik olmadığını söyleyin.
7. **Kategorik değişkenler** — `10_neighborhood_boxplot` ve `11_overallqual_boxplot`: konum ve kalite etkisini örnekleyin.
8. **Feature seçimi ve pipeline** — `12_feature_groups` ve `13_pipeline_workflow`: 30 feature'ın domain kapsamını ve fit/transform sınırını anlatın.
9. **Feature engineering deneyleri** — `14_feature_engineering_comparison`: yedi grubun validation + test kriterini birlikte geçemediğini gösterin.
10. **Model karşılaştırması** — `15_model_comparison`: Gradient Boosting'in validation lideri olmasına rağmen testte bozulduğunu; lineer modellerin daha kararlı olduğunu anlatın.
11. **Hata analizi** — `16_ridge_residual_analysis`: üst fiyat çeyreğindeki hata yoğunluğunu ve sonraki deneyleri açıklayın.

## Kullanabileceğiniz ana cümleler

- “Veriyi yalnız temizlemedik; eksikliklerin fiziksel anlamını koruduk.”
- “Korelasyonları feature seçiminin başlangıcı olarak kullandık, nihai karar olarak değil.”
- “Outlier'ları kör biçimde silmedik; model etkisini ölçülecek risk olarak işaretledik.”
- “Feature engineering hipotezlerini tek tek test ederek hangi değişikliğin sonucu etkilediğini izole ettik.”
- “Validation skoru tek başına yeterli değil; bağımsız testteki kararlılık model seçiminde belirleyici.”

## Kritik not

Projede ham, temizlik öncesi veri dosyası bulunmadığı için temizlik görseli uydurma “önceki missing sayıları” göstermez. Kodda tanımlanan kuralları ve temizlenmiş verideki sonucu gösterir.
