import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import os

# Create artifacts directory for plots
plot_dir = "/Users/ismailcangoktas/.gemini/antigravity-ide/brain/9564c20d-875f-4f71-85ad-f3577e6bd0d6/plots"
os.makedirs(plot_dir, exist_ok=True)

# Load Data
df = pd.read_csv("cleaned_train.csv")

report = []
report.append("# Kapsamlı Keşifçi Veri Analizi (EDA) Raporu\n")
report.append("Bu rapor `cleaned_train.csv` veriseti üzerinden kapsamlı istatistiksel ve görsel analizler sunmaktadır.\n")

# 1. Dataset Overview
report.append("## 1. Veriseti Genel Bakış\n")
report.append(f"- **Toplam Satır:** {df.shape[0]}")
report.append(f"- **Toplam Kolon:** {df.shape[1]}")
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns
report.append(f"- **Sayısal Değişkenler:** {len(num_cols)}")
report.append(f"- **Kategorik Değişkenler:** {len(cat_cols)}\n")

# 2. Missing Values
report.append("## 2. Eksik Veri (Missing Values) Analizi\n")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    report.append("> [!SUCCESS]\n> **Eksik Veri Bulunmamaktadır.** Bu veri seti daha önce detaylı temizleme süreçlerinden geçmiş ve modellemeye hazır hale getirilmiştir.\n")
else:
    report.append("Aşağıdaki kolonlarda eksik veri mevcuttur:\n")
    report.append(missing.to_markdown())

# 3. Target Variable Analysis
report.append("## 3. Hedef Değişken (SalePrice) Analizi\n")
sp = df["SalePrice"]
report.append("Evin satış fiyatına ait temel istatistikler:\n")
report.append(f"- **Ortalama (Mean):** ${sp.mean():,.2f}")
report.append(f"- **Ortanca (Median):** ${sp.median():,.2f}")
report.append(f"- **Standart Sapma:** ${sp.std():,.2f}")
report.append(f"- **Çarpıklık (Skewness):** {skew(sp):.2f} (Sağa Çarpık)")
report.append(f"- **Basıklık (Kurtosis):** {kurtosis(sp):.2f}\n")
report.append("> [!NOTE]\n> `SalePrice` değişkeninin çarpıklık değeri (skewness > 1) yüksek olduğu için, sağa doğru asimetrik bir kuyruğu vardır. Makine öğrenmesi modellerinin (özellikle lineer algoritmaların) daha iyi çalışması için bu değişkene model eğitiminden önce `log1p` (logaritma) dönüşümü uygulanmaktadır.\n")

plt.figure(figsize=(10,6))
sns.histplot(sp, kde=True, color='blue', bins=50)
plt.title('SalePrice Dağılımı')
plt.xlabel('SalePrice')
plt.ylabel('Frekans')
dist_plot = os.path.join(plot_dir, "saleprice_dist.png")
plt.savefig(dist_plot)
plt.close()
report.append(f"![SalePrice Dağılımı]({dist_plot})\n")

# 4. Outlier Analysis
report.append("## 4. Aykırı Değer (Outlier) Analizi\n")
report.append("Büyük metrekareye (GrLivArea) sahip ancak çok ucuza satılmış ekstrem evler genelde modelin dengesini bozar. Aşağıdaki grafikte yaşam alanı ile satış fiyatı arasındaki ilişkiyi ve muhtemel aykırı değerleri görüyoruz.\n")

plt.figure(figsize=(10,6))
sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], alpha=0.6, color='purple')
plt.title('GrLivArea vs SalePrice')
plt.xlabel('Zemin Üstü Yaşam Alanı (SqFt)')
plt.ylabel('SalePrice ($)')
scatter_plot = os.path.join(plot_dir, "grlivarea_saleprice.png")
plt.savefig(scatter_plot)
plt.close()
report.append(f"![GrLivArea vs SalePrice]({scatter_plot})\n")

# 5. Correlation Analysis
report.append("## 5. Korelasyon Analizi\n")
corrs = df[num_cols].corr()['SalePrice'].sort_values(ascending=False)
report.append("### En Çok Fiyatı Artıran İlk 10 Özellik (Pozitif Korelasyon)\n")
report.append(corrs[1:11].to_frame(name='Korelasyon').reset_index().rename(columns={'index':'Özellik'}).to_markdown(index=False) + "\n")

report.append("### En Çok Fiyatı Düşüren 5 Özellik (Negatif Korelasyon)\n")
report.append(corrs.tail(5).to_frame(name='Korelasyon').reset_index().rename(columns={'index':'Özellik'}).to_markdown(index=False) + "\n")

plt.figure(figsize=(12,10))
top_cols = corrs.head(12).index
sns.heatmap(df[top_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('En Önemli Sayısal Özellikler Arası Korelasyon Matrisi')
plt.tight_layout()
corr_plot = os.path.join(plot_dir, "heatmap.png")
plt.savefig(corr_plot)
plt.close()
report.append(f"![Korelasyon Heatmap]({corr_plot})\n")

# 6. Categorical Analysis
report.append("## 6. Kategorik Değişken Analizi\n")
report.append("Ev fiyatını en çok etkileyen kategorik faktörlerden biri evin bulunduğu mahalledir (`Neighborhood`). Aşağıdaki kutu grafiği (boxplot), mahallelere göre ev fiyatlarındaki medyan ve varyans değişimlerini net bir şekilde göstermektedir.\n")

plt.figure(figsize=(14,8))
order = df.groupby('Neighborhood')['SalePrice'].median().sort_values().index
sns.boxplot(x='Neighborhood', y='SalePrice', data=df, order=order, palette='viridis')
plt.xticks(rotation=90)
plt.title('Mahallelere (Neighborhood) Göre Ev Fiyatları Dağılımı')
plt.tight_layout()
box_plot = os.path.join(plot_dir, "neighborhood_boxplot.png")
plt.savefig(box_plot)
plt.close()
report.append(f"![Neighborhood Boxplot]({box_plot})\n")

# Save report
report_path = "/Users/ismailcangoktas/.gemini/antigravity-ide/brain/9564c20d-875f-4f71-85ad-f3577e6bd0d6/detayli_eda_raporu.md"
with open(report_path, "w") as f:
    f.write("\n".join(report))

print(f"Rapor oluşturuldu: {report_path}")
