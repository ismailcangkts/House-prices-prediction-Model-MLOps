# Feature Engineering Deney Raporu

## Yönetici özeti

- Baz model: Ridge Regression (`alpha=10`), standartlaştırılmış girdiler ve log-dönüşümlü hedef.
- Validation bazı: RMSE **26296.49**, MAE **17368.97**.
- Bağımsız test bazı: RMSE **26494.28**, MAE **17236.03**.
- Hem validation hem testte RMSE ve MAE'yi birlikte iyileştiren yöntemler: **Yok**.
- Validation sonucuna göre birleşim için seçilenler: **Yok**.
- Seçilen yöntemlerin birleşimi testte RMSE'yi **+0.00%**, MAE'yi **+0.00%** değiştirdi.
- Mevcut proje feature engineering'i baz modele göre validation RMSE'yi **+1.10%**, test RMSE'yi **-1.38%** değiştirdi.

Pozitif yüzde baz modele göre hata azalmasını, negatif yüzde hata artışını gösterir.

## Metrik bazında kısmi olumlu sonuçlar

- **Domain-based** feature'lar test MAE'yi **+6.20%** ve RMSLE'yi **+6.16%** iyileştirdi; ancak RMSE **-1.07%** kötüleşti. Tipik/tahmini oransal hata azalırken büyük hatalar arttı.
- **Dönüştürme** test MAE'yi **+1.77%** ve RMSLE'yi **+2.29%** iyileştirdi; buna karşılık RMSE **-7.31%** kötüleşti.
- **Tüm yöntemlerin birleşimi** test MAE'de **+4.15%**, RMSLE'de **+3.45%** kazanç sağladı; RMSE'de **-8.08%** kayıp oluşturdu. Projenin ana seçim metriği RMSE olduğu için production adayı sayılmadı.

## Sonuç tablosu

| Yöntem | Validation RMSE | Test RMSE | Validation MAE | Test MAE | Karar |
|---|---:|---:|---:|---:|---|
| Birleştirme | -0.08% | -0.01% | -0.01% | -0.04% | Olumlu sonuç vermedi |
| Ayrıştırma | -0.21% | -0.44% | +0.13% | -0.24% | Olumlu sonuç vermedi |
| Domain-based | -1.75% | -1.07% | +3.21% | +6.20% | Olumlu sonuç vermedi |
| Binning / Gruplama | +0.34% | -2.52% | -0.98% | -3.57% | Olumlu sonuç vermedi |
| Binary / Indicator | -2.31% | -2.84% | -1.46% | -2.71% | Olumlu sonuç vermedi |
| Etkileşim | -8.52% | -3.71% | -0.86% | -0.42% | Olumlu sonuç vermedi |
| Dönüştürme | -6.22% | -7.31% | +8.60% | +1.77% | Olumlu sonuç vermedi |

## Deney protokolü

- Projenin mevcut `train.csv` (798), `validation.csv` (218) ve daha önce model seçimi için kullanılmamış `test.csv` (145) ayrımları kullanıldı.
- İlk aşamada her yöntem yalnızca train üzerinde eğitilip validation üzerinde ölçüldü.
- İkinci aşamada train ve validation birleştirildi; her yöntem yeniden eğitilip bağımsız test setinde ölçüldü.
- Her deneyde model, hedef dönüşümü, imputation, ordinal encoding, one-hot encoding ve ölçekleme aynı tutuldu. Değişen tek unsur eklenen feature grubudur.
- “Olumlu ve testte doğrulandı” kararı için hem RMSE'nin hem MAE'nin baz modele göre iki ayrımda da düşmesi şartı uygulandı.

## Denenen feature'lar

- **Birleştirme:** TotalSF, TotalBathrooms
- **Ayrıştırma:** BuildDecade, BuildYearInDecade, RemodelDecade, RemodelYearInDecade
- **Dönüştürme:** LogLotArea, LogGrLivArea, LogLotFrontage, LogOpenPorchSF, LogTotalBsmtSF, Log1stFlrSF, Log2ndFlrSF, LogGarageArea
- **Etkileşim:** Qual_x_GrLivArea, Qual_x_TotalBsmtSF, Qual_x_GarageCars
- **Binary / Indicator:** HasGarage, HasBasement, HasFireplace, HasSecondFloor, HasOpenPorch, WasRemodeled
- **Binning / Gruplama:** Qual_Low/Mid/High, Built_Pre1946/1946_1970/1971_1999/2000Plus
- **Domain-based:** HouseAge2010, RemodelAge2010, GarageAge2010, LivingAreaPerRoom, GarageAreaPerCar, FinishedBasementRatio


## Yorum ve öneri

- Tek bir validation ayrımındaki küçük kazançlar tesadüfi olabilir; üretim kararında bağımsız test doğrulaması esas alınmalıdır.
- Bu deney yalnızca mevcut Ridge modeli için nedensel karşılaştırmadır. Ağaç tabanlı modeller etkileşimleri ve eşikleri kendileri öğrenebildiği için sıralama değişebilir.
- `HouseAge2010` için sabit 2010 referansı kullanıldı; seçili 30 kolonda `YrSold` bulunmuyor. İleride `YrSold` modele geri alınırsa `HouseAge = YrSold - YearBuilt` tercih edilmelidir.
- Mevcut `test_feature_engineering.py` içindeki alt sınıf önce `super().transform()` çağırdığı için kaynak alan kolonları siliniyor; ardından yeniden hesaplanan `TotalSF` ve `TotalBathrooms` sıfıra dönüşüyor. Bu dosyadaki görünen combo iyileşmesi geçerli bir feature-engineering sonucu değildir. Tekrarlanabilir deney için bu raporu üreten `scripts/run_feature_engineering_experiments.py` kullanılmalıdır.
- Üretim pipeline'ına yalnızca bağımsız testte doğrulanan feature gruplarının kontrollü bir aday model olarak alınması önerilir.
