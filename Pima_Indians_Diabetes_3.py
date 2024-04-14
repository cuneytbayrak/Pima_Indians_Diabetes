#!/usr/bin/env python
# coding: utf-8

# In[467]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from numpy.linalg import eig
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler,StandardScaler,QuantileTransformer
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree
from sklearn import tree


# # Data Preprocessing

# Veri setimiz de kolon isimleri olmadığı için öncelikle veri setine kolon isimleri ekleyerek işe başlayalım.
# 
# Number of times pregnant:Pregnancies
# 
# Plasma glucose concentration a 2 hours in an oral glucose tolerance test : Glucose
# 
# Diastolic blood pressure (mm Hg). : BloodPressure
# 
# Triceps skinfold thickness (mm). : SkinThickness
# 
# 2-Hour serum insulin (mu U/ml). : Insulin
# 
# Body mass index (weight in kg/(height in m)^2). : BMI
# 
# Diabetes pedigree function. : DPF
# 
# Age (years). : Age
# 
# Class variable (0 or 1). : ClassVariable

# In[178]:


column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                "Insulin", "BMI", "DPF","Age", "ClassVariable"]


# In[179]:


data = pd.read_csv('./data/veri-seti.txt',names=column_names ,delimiter = "\t")


# In[180]:


data.info()


# datamız 768 satır ve 9 kolondan oluşuyor. datada ki değerlerin hepsi sayısal değerler. iki kolon da ise float değerler söz konusu.

# In[181]:


data.head()


# data da hiç null değer var mı bir bakalım. bu arada ilk 5 veriyi yazdırınca göze çarpan ilk şey datanın bazı kolonlarında 0 verilerinin olduğu. mesela Insulin değerleri arasında 0 verisi var ve bunun olması imkansız. Muhtemelen datada kayıp veri var. buna da bakmamız gerekecek.

# In[182]:


print(data.isnull().any())


# data da hiç null veri yok. birde datanın min max ort ve std verilerine bakalım.

# In[183]:


data.describe().T


# Fakat datanın describ'ine baktığımız da min değerleri 0 olan özellikler var.
# Aşağıdaki özniteliklerde sıfır değerleri var ve bu öz nitelikleri için sıfır değeri bir anlam ifade etmez ve bu nedenle eksik değerlere işaret eder.
# 
# 1.Glucose
# 
# 2.BloodPressure
# 
# 3.SkinThickness
# 
# 4.Insulin
# 
# 5.BMI
# 
# bu eksik değerleri NaN olarak değiştirelim ve kaç adet eksik değer varmış bir görelim. eğer eksik değer az ise datadan atabiliriz.

# In[184]:


data_copy = data.copy(deep = True)
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

print(data_copy.isnull().sum())


# In[185]:


# Veri setindeki kayıp değerlerin yüzdelik oranlarını hesaplayalım
missing_values_percent = (data_copy.isnull().sum() / len(data_copy)) * 100

# Grafik oluşturalım
plt.figure(figsize=(10, 6))
bars = missing_values_percent.plot(kind='bar', color='red')

# Çubukların üzerine yüzde değerlerini yazdıralım
for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}%", ha='center', va='bottom')

plt.title('Kayıp Veri Yüzdesi')
plt.ylabel('%')
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# Kayıp Veriler :
# 
# Insulin       = 48.70%  - 374
# 
# SkinThickness = 29.56%  - 227
# 
# BloodPressure = 4.56%   -  35
# 
# BMI           = 1.43%   -  11
# 
# Glucose       = 0.65%   -   5
# 
# Eksik değerler çok fazla olduğu için datadan atmak veri kaybına neden olacak. bu durumda eksik değerleri hangi yöntemle dolduracağımıza karar vermek için histogramına ve boxplotuna bakalım. aykırı değerlerin çokça olduğu veri setlerinde median, olmadığı yada az olduğu veri setlerinde mean yöntemini kullanarak veriyi doldurabiliriz. 

# Bir kişinin diyabete yakalanma olasılığını potansiyel olarak etkileyebilecek tüm değişkenlerin histogramlarını çizelim. 
# 
# İlk olarak veri çerçevesindeki tüm kişiler için değişkenlerin histogramlarına bakalım.
# Daha sonra hasta bireyler için ayrı sağlıklı bireyler için ayrı ayrı histogramlara bakalım.
# Bunun için positive ne negative sınıfları ayrı ayrı elde edelim öncelikle.

# In[186]:


data_copy_positive = data[data_copy['ClassVariable']==1]
data_copy_positive.head()


# In[187]:


data_copy_negative = data[data_copy['ClassVariable']==0]
data_copy_negative.head()


# In[188]:


# tüm datanın histogramı
fig, ax = plt.subplots(4,2, figsize=(15,15))
sns.histplot(data_copy.Pregnancies, bins = 10, color = 'orange',kde=True, ax=ax[0,0]) 
sns.histplot(data_copy.Glucose, bins = 10, color = 'orange',kde=True, ax=ax[0,1]) 
sns.histplot(data_copy.BloodPressure, bins = 10, color = 'orange',kde=True, ax=ax[1,0]) 
sns.histplot(data_copy.SkinThickness, bins = 10, color = 'orange',kde=True, ax=ax[1,1]) 
sns.histplot(data_copy.Insulin, bins = 10, color = 'orange',kde=True, ax=ax[2,0])
sns.histplot(data_copy.BMI, bins = 10, color = 'orange',kde=True, ax=ax[2,1])
sns.histplot(data_copy.DPF, bins = 10, color = 'orange',kde=True, ax=ax[3,0]) 
sns.histplot(data_copy.Age, bins = 10,color = 'orange',kde=True, ax=ax[3,1]) 


# Şimdi, benzer şekilde, yalnızca veri çerçevesindeki diyabet testi negatif çıkan kişiler için yani sağlıklı kişiler için değişkenlerin histogramlarını çizelim.

# In[189]:


# sağlıklı kşilerin histogramı
fig, ax = plt.subplots(4,2, figsize=(15,15))
sns.histplot(data_copy_negative.Pregnancies, bins = 10, color = 'green',kde=True, ax=ax[0,0]) 
sns.histplot(data_copy_negative.Glucose, bins = 10, color = 'green',kde=True, ax=ax[0,1]) 
sns.histplot(data_copy_negative.BloodPressure, bins = 10, color = 'green',kde=True, ax=ax[1,0]) 
sns.histplot(data_copy_negative.SkinThickness, bins = 10, color = 'green',kde=True, ax=ax[1,1]) 
sns.histplot(data_copy_negative.Insulin, bins = 10, color = 'green',kde=True, ax=ax[2,0])
sns.histplot(data_copy_negative.BMI, bins = 10, color = 'green',kde=True, ax=ax[2,1])
sns.histplot(data_copy_negative.DPF, bins = 10, color = 'green',kde=True, ax=ax[3,0]) 
sns.histplot(data_copy_negative.Age, bins = 10,color = 'green',kde=True, ax=ax[3,1]) 


# In[190]:


# hasta kşilerin histogramı
fig, ax = plt.subplots(4,2, figsize=(15,15))
sns.histplot(data_copy_positive.Pregnancies, bins = 10, color = 'red',kde=True, ax=ax[0,0]) 
sns.histplot(data_copy_positive.Glucose, bins = 10, color = 'red',kde=True, ax=ax[0,1]) 
sns.histplot(data_copy_positive.BloodPressure, bins = 10, color = 'red',kde=True, ax=ax[1,0]) 
sns.histplot(data_copy_positive.SkinThickness, bins = 10, color = 'red',kde=True, ax=ax[1,1]) 
sns.histplot(data_copy_positive.Insulin, bins = 10, color = 'red',kde=True, ax=ax[2,0])
sns.histplot(data_copy_positive.BMI, bins = 10, color = 'red',kde=True, ax=ax[2,1])
sns.histplot(data_copy_positive.DPF, bins = 10, color = 'red',kde=True, ax=ax[3,0]) 
sns.histplot(data_copy_positive.Age, bins = 10,color = 'red',kde=True, ax=ax[3,1]) 


# Tüm eksik değerler NaN değeriyle kodlanmış oldu.
# Datanın histogramına  bakıldı ve buradan hareketle bize eksik verilerin tamamlanması için bir fikir verdi.
# Burada dikkat edilmesi gereken husus şudur; mean ile doldurulacak kayıp verileri ClassVariable değerlerinin her biri için ayrı ayrı hesaplamak lazım gerektiğidir. Çünkü sağlıklı bir kişinin Insulin değerleri ile hasta kişinin insulin değerleri aynı olmayacaktır. O bakımdan sağlıklı ve hasta veriler için ayrı ayrı mean hesaplanıp kayıp veriler buna dikkat edilerek doldurulması gerekir.
# Bunun için bir fonksiyon yazalım ve her bir öznitelik için ortalamaları hasta ve sağlıklı bireyler için ayrı ayrı hesaplayarak kayıp verileri dolduralım.

# In[193]:


def mean_target(var):   
    temp = data[data_copy[var].notnull()]
    temp = temp[[var, 'ClassVariable']].groupby(['ClassVariable'])[[var]].mean().reset_index()
    return temp


# In[194]:


mean_target('Insulin')


# Etiket değerlerimize göre insülinin medyanları gerçekten farklı! 
# Sağlıklı bir kişi için 130.287879 iken diyabetik bir kişi için 206.846154 olmaktadır.Kayıp veriler doldurulurken bu husus göz önünde bulundurularak sağlıklı etiket değerlerinde ki kayıp veriler 130.287879 ile hasta etiket değerinde ki kayıp veriler 206.846154 ile değiştirilmesi gerekir. Bu kural diğer tüm kayıp veri barındıran öznitelikler için aynı şekilde yapılması gerekir. 

# In[195]:


data_copy.loc[(data_copy['ClassVariable'] == 0 ) & (data_copy['Insulin'].isnull()), 'Insulin'] = 130.287879
data_copy.loc[(data_copy['ClassVariable'] == 1 ) & (data_copy['Insulin'].isnull()), 'Insulin'] = 206.846154


# In[196]:


data.head(5)


# In[197]:


data_copy.head(5)


# Dikkat edilecek olursa; manipüle edilmemiş data matrisinin ilk 3 insulin değeri 0 idi.data_copy matrisinde bu ilk 3 değer ilgili etiket değerine göre insulin özniteliğinin mean değeri ile değişmiş durumda. 

# In[198]:


mean_target('SkinThickness')


# In[199]:


data_copy.loc[(data_copy['ClassVariable'] == 0 ) & (data_copy['SkinThickness'].isnull()), 'SkinThickness'] = 27.235457
data_copy.loc[(data_copy['ClassVariable'] == 1 ) & (data_copy['SkinThickness'].isnull()), 'SkinThickness'] = 33.000000


# In[200]:


mean_target('BloodPressure')


# In[201]:


data_copy.loc[(data_copy['ClassVariable'] == 0 ) & (data_copy['BloodPressure'].isnull()), 'BloodPressure'] = 70.877339
data_copy.loc[(data_copy['ClassVariable'] == 1 ) & (data_copy['BloodPressure'].isnull()), 'BloodPressure'] = 75.321429


# In[202]:


mean_target('BMI')


# In[203]:


data_copy.loc[(data_copy['ClassVariable'] == 0 ) & (data_copy['BMI'].isnull()), 'BMI'] = 30.859674
data_copy.loc[(data_copy['ClassVariable'] == 1 ) & (data_copy['BMI'].isnull()), 'BMI'] = 35.406767


# In[204]:


mean_target('Glucose')


# In[205]:


data_copy.loc[(data_copy['ClassVariable'] == 0 ) & (data_copy['Glucose'].isnull()), 'Glucose'] = 110.643863
data_copy.loc[(data_copy['ClassVariable'] == 1 ) & (data_copy['Glucose'].isnull()), 'Glucose'] = 142.319549


# In[206]:


print(data_copy.isnull().sum())


# In[207]:


data_copy.describe().T


# In[208]:


plt.style.use('ggplot') # Using ggplot2 style visuals 

# 'ClassVariable' sütununu veri kümesinden kaldırma
data_no_class = data_copy.drop(columns=['ClassVariable'])

f, ax = plt.subplots(figsize=(11, 11))

ax.set_facecolor('#fafafa')
plt.ylabel('Variables')
plt.title("Veri Setinin BoxPlot Görünümü")
ax = sns.boxplot(data = data_no_class, orient = 'v', palette = 'Set2')
ax.tick_params(axis='x', labelsize=8)


# # Winsorization İşlemi

# Veri setinde, özellikle Insulin değerlerinde çok fazla aykırı değerler var. Bu aykırı değerleri baskılamanın çeşitli yöntemleri olmakla beraber, Winsorization, veri setindeki aykırı değerlerin etkilerini azaltmak veya ortadan kaldırmak için kullanılan bir yöntemdir. Bu yöntem, veri setindeki en yüksek ve en düşük değerleri sınırlı bir aralığa taşır ve bu sınırlar genellikle verinin yüzde x ve yüzde z dilimleridir. 
# 
# Örneğin, %90 Winsorization, verilerin en üst %5 ve en alt %5'inin değiştirilmesi anlamına gelir. Verilerin en üstteki %5'inin yerine 95. yüzdelik dilimdeki verinin değeri, en alttaki %5'in yerine ise 5. yüzdelik dilimdeki verinin değeri yerleştirilir. 
# 
# Şimdi Winsorization yapalım.

# In[210]:


data_copy_w=data_copy.copy()
data_copy_w["Pregnancies"] = winsorize(data_copy_w["Pregnancies"],(0.01,0.01)) 
data_copy_w["Glucose"] = winsorize(data_copy_w["Glucose"],(0.01,0.01)) 
data_copy_w["BloodPressure"] = winsorize(data_copy_w["BloodPressure"],(0.02,0.02)) 
data_copy_w["SkinThickness"] = winsorize(data_copy_w["SkinThickness"],(0.05,0.05)) 
data_copy_w["Insulin"] = winsorize(data_copy_w["Insulin"],(0.05,0.05)) 
data_copy_w["BMI"] = winsorize(data_copy_w["BMI"],(0.02,0.02)) 
data_copy_w["DPF"] = winsorize(data_copy_w["DPF"],(0.05,0.05)) 
data_copy_w["Age"] = winsorize(data_copy_w["Age"],(0.02,0.02)) 


# In[211]:


plt.style.use('ggplot') # Using ggplot2 style visuals 

# 'ClassVariable' sütununu veri kümesinden kaldırma
data_no_class_w = data_copy_w.drop(columns=['ClassVariable'])

f, ax = plt.subplots(figsize=(11, 11))

ax.set_facecolor('#fafafa')
plt.ylabel('Variables')
plt.title("Veri Setinin BoxPlot Görünümü")
ax = sns.boxplot(data = data_no_class_w, orient = 'v', palette = 'Set2')
ax.tick_params(axis='x', labelsize=8)


# 
# Yukarıda ki boxplotta görüleceği gibi Winsorization yöntemi ile Veri setinde ki aykırı değerler baskılanmış oldu.
# 
# 
# 
# Hedefimiz, bir kişinin diyabet hastası olup olmadığını tahmin etmek olduğu için bunu doğru bir şekilde yapabilmek adına tüm kayıp verileri düzeltmiş olduk. Veri çerçevesine baktığımızda, Insulin, SkinThickness, BloodPressure, BMI ve Glucose'da çok sayıda 0 değeri olduğunu gözlemlemiştik. Ancak bu değerlerden herhangi birinin bir kişi için sıfır olması mümkün değildi. Bunun nedeni, tüm bu sıfırların aslında eksik girişler olduğuydu. Bu nedenle, Insulin, SkinThickness, BloodPressure, BMI ve Glucose için 0 değerlerini kaldırdık ve bunları bu sütunların her biri için ortalama değerlerle değiştirdik.
# 
# Pregnancies (Hamilelik) sütunundaki 0 değerlerini değiştirmedik. Çünkü bu sadece kişinin hamile olmadığı anlamına gelir,kayıp veri değildir. 
# 

# In[222]:


p=sns.pairplot(data_copy_w, hue = 'ClassVariable')


# # Data Scaling
# 
# 

# In[223]:


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(data_copy_w.drop(["ClassVariable"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DPF', 'Age'])


# In[224]:


X_scaled.head()


# In[234]:


y = data_copy_w.ClassVariable


# # PCA - Principle Component Analiz
# 
# 

# In[235]:


pca = PCA(n_components=2)
X_scaled_pca = pca.fit(X_scaled).transform(X_scaled)


# In[236]:


eigen_values = pca.explained_variance_
eigen_vectors = pca.components_


# In[237]:


eigen_values


# In[238]:


eigen_vectors


# In[268]:


# En yüksek ağırlığa sahip olan özniteliklerin indekslerini bulma
first_max_index = np.argmax(np.abs(first_component_weights))
second_max_index = np.argmax(np.abs(second_component_weights))

# En yüksek ağırlıklara sahip olan özniteliklerin isimlerini ve değerlerini bulma
first_max_feature = X_scaled.columns[first_max_index]
second_max_feature = X_scaled.columns[second_max_index]

first_max_weight = first_component_weights[first_max_index]
second_max_weight = second_component_weights[second_max_index]

print("En yüksek birinci öz nitelik: {} ({})".format(first_max_feature, first_max_weight))
print("En yüksek ikinci öz nitelik: {} ({})".format(second_max_feature, second_max_weight))


# In[240]:


# İlk iki bileşenin varyans oranlarını alalım
explained_variance_ratio = pca.explained_variance_ratio_

# İlk iki bileşenin varyans oranlarını toplayalım
total_variance_explained_ratio = np.sum(explained_variance_ratio[:2])

print("En yüksek özdeğere sahip iki bileşenle orijinal veri setinin açıkladığı varyans oranı: %", np.round(total_variance_explained_ratio*100,2))


# In[367]:


plt.figure(figsize=(8, 6))
plt.scatter(X_scaled_pca[:, 0], X_scaled_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class variable')
plt.grid(True)
plt.show()


# # LDA- Lineer Discriminant Analiz

# In[370]:


lda = LDA(n_components=1) 
X_scaled_lda = lda.fit_transform(X_scaled, y)
print("LDA Components:\n", lda.scalings_)


# In[371]:


# Öz niteliklerin isimleri
feature_names = X_scaled.columns

# LDA bileşenlerini ve öznitelikleri eşleştirme
lda_components = lda.scalings_.flatten()  # Düzleştirme işlemi
for i, component in enumerate(lda_components):
    feature_name = feature_names[i]  # Öz nitelik ismi
    print(f"{feature_name}: {component}")


# en yüksek iki öz nitelik Insulin: 0,603 ve Glucose:0.529 

# # SPLİT İşlemi

# Veri setimizi split etmeden önce etiket verilerimizin dağılımına bakalım.

# In[241]:


colors = ['gold', 'mediumturquoise']
labels = ['Sağlıklı','Hasta']
values = data_copy_w['ClassVariable'].value_counts()/data_copy_w['ClassVariable'].shape[0]

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title_text="ClassVariable", width=400, height=400)
fig.show()


# Yukarıdaki grafik verilerin dengesiz olduğunu göstermektedir. 
# 
# Sağlıklı bireylerin yüzdesi %65.1, diyabet hastası bireylerin yüzdesi ise %34.9
# 
# Bu durumda split işleminde bir detaya dikkat etmek gerekir.
# 
# Stratify parametresi, üretilen örneklemin değerlerinin oranının, stratify parametresine verilen değerlerin oranıyla aynı olmasını sağlayarak bir split yapar.
# 
# Örneğin, değişken y ikili kategorik bir değişkense ve değerleri 0 ve 1 ise ve sıfırların %25'i ve birlerin %75'i varsa, stratify=y, rastgele bölmenizin %25'inin 0 ve %75'inin 1 olduğundan emin olur.
# 

# In[242]:


#Ana Veri setini eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42,stratify=y)


# In[251]:


#PCA uygulanmış transformasyon öznitelik matrisini eğitim ve test verisi olarak ayırma
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_scaled_pca, y, test_size=0.3, random_state=42,stratify=y)


# # Ana Veri Seti Çoklu Doğrusal Regresyon Analizi

# Çoklu doğrusal regresyon analizi etiket değerlerinin sürekli olduğu durumlarda kullanılır esasında. 
# 
# Veri setinde etiket değerlerimiz kategorik (0 ve 1) olduğu için doğrusal bir regresyon analizi iyi sonuçlar vermeyecektir.
# 
# Ancak tahmin edilen değerleri sürekli halden bir fonksiyondan geçirerek binary hale getirirsek o zaman lineer regresyon bir sonuç verebilir.

# In[374]:


# Çoklu doğrusal regresyon modelini eğitme
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)


# In[375]:


# Test kümesi üzerinde tahminlerin yapılması
y_pred = linear_reg_model.predict(X_test)


# In[376]:


y_pred


# Tahmin değerlerine bakıldığında negatif ve pozitif değerlerden oluşmakta.
# 
# Önce bu değerleri sigmoid fonksiyonu ile normalize edip sonuçları 0 ile 1 arasında bir değere dönüştüreceğim.
# 
# Ardından bu normalleştirilmiş değerlere eşik değeri uygulayarak binary tahminler elde edeceğim.

# In[377]:


# Sigmoid fonksiyonu tanımlama
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[378]:


# Tahminlerin normalize edilmesi
normalized_predictions = sigmoid(y_pred)


# In[379]:


normalized_predictions


# In[392]:


# Normalleştirilmiş tahminlerin ortalamasını bulma
mean_normalized_predictions = np.mean(normalized_predictions)

print("Normalleştirilmiş Tahminlerin Ortalaması:", mean_normalized_predictions)


# Normalleştirilmiş Tahminlerin Ortalaması: 0.58 çıktığı için eşik değerini 0.6 almaya karar verdim.
# 
# 0.6 dan küçük değerler 0
# 
# 0.6 dan büyük değerler 1 olarak işaretlenecek

# In[396]:


# Eşik değerini belirleme
threshold = 0.6

# Tahminleri binary hale getirme
binary_predictions = (normalized_predictions > threshold).astype(int)


# In[397]:


binary_predictions


# Lineer regresyon sonucu oluşan tahminlerimizi binary hale getirmiş olduk.

# In[420]:


# Sınıflandırma raporunu oluşturma
class_report = classification_report(y_test, binary_predictions)

print("Ana Veri Seti Çoklu Doğrusal Regresyon Sınıflandırma Raporu:")
print(class_report)


# In[421]:


# Sınıflandırma raporunu oluşturma
class_report = classification_report(y_test, binary_predictions, output_dict=True)

# Sensitivity (Duyarlılık) hesaplama
sensitivity = class_report['1']['recall']

# Specificity (Özgünlük) hesaplama
specificity = class_report['0']['recall']

print("Ana Veri Seti Çoklu Doğrusal Regresyon Sensitivity:", sensitivity)
print("Ana Veri Seti Çoklu Doğrusal Regresyon Specificity:", specificity)


# In[425]:


# Konfüzyon matrisini hesaplama
conf_matrix_lrm = confusion_matrix(y_test, binary_predictions)

# Konfüzyon matrisini görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_lrm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Ana Veri Seti Çoklu Doğrusal Regresyon Konfüzyon Matrisi', fontsize=12)
plt.show()


# In[426]:


# ROC eğrisini hesaplama
fpr_lrm, tpr_lrm, thresholds_lrm = roc_curve(y_test, binary_predictions)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_lrm, tpr_lrm, color='darkorange', lw=2, label='ROC Eğrisi (AUC = %0.2f)' % auc(fpr_lrm, tpr_lrm))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ana Veri Seti Çoklu Doğrusal Regresyon ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# # PCA Veri Seti Çoklu Doğrusal Regresyon analizi

# In[403]:


# PCA-Çoklu doğrusal regresyon modelini eğitme
linear_reg_model_pca = LinearRegression()
linear_reg_model_pca.fit(X_train_pca, y_train_pca)


# In[404]:


# Test kümesi üzerinde tahminlerin yapılması
y_pred_pca = linear_reg_model_pca.predict(X_test_pca)


# In[405]:


y_pred_pca


# Tahmin değerlerine bakıldığında gene negatif ve pozitif değerlerden oluşmakta.
# 
# Önce bu değerleri sigmoid fonksiyonu ile normalize edip sonuçları 0 ile 1 arasında bir değere dönüştüreceğim.
# 
# Ardından bu normalleştirilmiş değerlere eşik değeri uygulayarak binary tahminler elde edeceğim.

# In[406]:


# Tahminlerin normalize edilmesi
normalized_predictions_pca = sigmoid(y_pred_pca)


# In[407]:


normalized_predictions_pca


# In[408]:


# Normalleştirilmiş tahminlerin ortalamasını bulma
mean_normalized_predictions_pca = np.mean(normalized_predictions_pca)

print("Normalleştirilmiş Tahminlerin Ortalaması:", mean_normalized_predictions_pca)


# Normalleştirilmiş Tahminlerin Ortalaması: 0.58 çıktığı için eşik değerini 0.6 almaya karar verdim.
# 
# 0.6 dan küçük değerler 0
# 
# 0.6 dan büyük değerler 1 olarak işaretlenecek

# In[409]:


# Eşik değerini belirleme
threshold = 0.6

# Tahminleri binary hale getirme
binary_predictions_pca = (normalized_predictions_pca > threshold).astype(int)


# In[410]:


binary_predictions_pca


# In[427]:


# Sınıflandırma raporunu oluşturma
class_report_pca = classification_report(y_test_pca, binary_predictions_pca)

print("PCA Veri Seti Çoklu Doğrusal Regresyon Sınıflandırma Raporu:")
print(class_report_pca)


# In[428]:


# Sınıflandırma raporunu oluşturma
class_report_pca = classification_report(y_test_pca, binary_predictions_pca, output_dict=True)

# Sensitivity (Duyarlılık) hesaplama
sensitivity_pca = class_report_pca['1']['recall']

# Specificity (Özgünlük) hesaplama
specificity_pca = class_report_pca['0']['recall']

print("PCA Veri Seti Çoklu Doğrusal Regresyon Sensitivity:", sensitivity_pca)
print("PCA Veri Seti Çoklu Doğrusal Regresyon Specificity:", specificity_pca)


# In[429]:


# Konfüzyon matrisini hesaplama
conf_matrix_lrm_pca = confusion_matrix(y_test_pca, binary_predictions_pca)

# Konfüzyon matrisini görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_lrm_pca, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('PCA Veri Seti Çoklu Doğrusal Regresyon Konfüzyon Matrisi')
plt.show()


# In[430]:


# ROC eğrisini hesaplama
fpr_lrm_pca, tpr_lrm_pca, thresholds_lrm_pca = roc_curve(y_test_pca, binary_predictions_pca)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_lrm_pca, tpr_lrm_pca, color='darkorange', lw=2, label='ROC Eğrisi (AUC = %0.2f)' % auc(fpr_lrm_pca, tpr_lrm_pca))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PCA Veri Seti Çoklu Doğrusal Regresyon ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# # Ana Veri Seti Multinominal Lojistik Regresyon Analizi

# In[243]:


# Multinominal Lojistik Regresyon analizi
logistic_reg_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logistic_reg_model.fit(X_train, y_train)


# In[244]:


# Öznitelik isimlerini ve katsayıları eşleştirme
feature_names = X_train.columns

# Multinominal Lojistik Regresyon modelinin katsayılarını raporlama
coefficients = logistic_reg_model.coef_
intercept = logistic_reg_model.intercept_

print("Katsayılar:")
for i, coef in enumerate(coefficients):
    for feature_name, value in zip(feature_names, coef):
        print(f"  {feature_name}: {value}")
    print(f"  Sabit terim: {intercept[i]}")
    print()


# In[245]:


# Modelin performansını test verisi üzerinde değerlendirme
logistic_reg_predictions = logistic_reg_model.predict(X_test)


# In[431]:


# Model performanslarını değerlendirme
logistic_reg_report = classification_report(y_test, logistic_reg_predictions)
print("Ana Veri Seti Multinominal Lojistik Regresyon Sınıflandırma Raporu:")
print(logistic_reg_report)


# In[432]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_MLR = confusion_matrix(y_test, logistic_reg_predictions)
tn_MLR, fp_MLR, fn_MLR, tp_MLR = conf_matrix_MLR.ravel()
sensitivity_MLR = tp_MLR / (tp_MLR + fn_MLR)
specificity_MLR = tn_MLR / (tn_MLR + fp_MLR)
print("Ana Veri Seti Multinominal Lojistik Regresyon Sensitivity:", sensitivity_MLR)
print("Ana Veri Seti Multinominal Lojistik Regresyon Specificity:", specificity_MLR)


# In[433]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_MLR, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Ana Veri Seti Multinominal Lojistik Regresyon Konfüzyon Matrisi')
plt.show()


# In[434]:


# ROC eğrisi ve AUC hesaplama
fpr_MLR, tpr_MLR, thresholds_MLR = roc_curve(y_test, logistic_reg_predictions)
roc_auc_MLR = auc(fpr_MLR, tpr_MLR)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_MLR, tpr_MLR, color='darkorange', lw=2, label='Multinominal Lojistik Regresyon ROC Eğrisi (AUC = %0.2f)' % roc_auc_MLR)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ana Veri Seti Multinominal Lojistik Regresyon ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# # PCA Veri Seti Multinominal Lojistik Regresyon

# In[252]:


# PCA uygulanmış transformasyon öznitelik matrisine Multinominal Lojistik Regresyon analizi
logistic_reg_model_pca = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logistic_reg_model_pca.fit(X_train_pca, y_train_pca)


# In[256]:


# Öznitelik isimlerini elde etme
feature_names_pca = pca.get_feature_names_out()

coefficients_pca = logistic_reg_model_pca.coef_
intercept_pca = logistic_reg_model_pca.intercept_

print("Katsayılar:")
for i, coef in enumerate(coefficients_pca):
    for feature_name, value in zip(feature_names_pca, coef):
        print(f"  {feature_name}: {value}")
    print(f"  Sabit terim: {intercept_pca[i]}")
    print()


# In[257]:


# Modelin performansını test verisi üzerinde değerlendirme
logistic_reg_predictions_pca = logistic_reg_model_pca.predict(X_test_pca)


# In[435]:


# Model performanslarını değerlendirme
logistic_reg_report_pca = classification_report(y_test_pca, logistic_reg_predictions_pca)
print("PCA Veri Seti Multinominal Lojistik Regresyon Sınıflandırma Raporu:")
print(logistic_reg_report_pca)


# In[436]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_MLR_pca = confusion_matrix(y_test_pca, logistic_reg_predictions_pca)
tn_MLR_pca, fp_MLR_pca, fn_MLR_pca, tp_MLR_pca = conf_matrix_MLR_pca.ravel()
sensitivity_MLR_pca = tp_MLR_pca / (tp_MLR_pca + fn_MLR_pca)
specificity_MLR_pca = tn_MLR_pca / (tn_MLR_pca + fp_MLR_pca)
print("PCA Veri Seti Multinominal Lojistik Regresyon Sensitivity:", sensitivity_MLR_pca)
print("PCA Veri Seti Multinominal Lojistik Regresyon Specificity:", specificity_MLR_pca)


# In[437]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_MLR_pca, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('PCA Veri Seti Multinominal Lojistik Regresyon Konfüzyon Matrisi')
plt.show()


# In[438]:


# ROC eğrisi ve AUC hesaplama
fpr_MLR_pca, tpr_MLR_pca, thresholds_MLR_pca = roc_curve(y_test_pca, logistic_reg_predictions_pca)
roc_auc_MLR_pca = auc(fpr_MLR_pca, tpr_MLR_pca)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_MLR_pca, tpr_MLR_pca, color='darkorange', lw=2, label='PCA-Multinominal Lojistik Regresyon ROC Eğrisi (AUC = %0.2f)' % roc_auc_MLR_pca)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PCA Veri Seti Multinominal Lojistik Regresyon ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# # Ana Veri Karar Ağaç Sınıflandırma Algoritması

# In[312]:


# Karar Ağacı sınıflandırma modelini eğitme
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)


# In[313]:


# Test verisi üzerinde modelin performansını değerlendirme ve performans metriklerini hesaplama
decision_tree_predictions = decision_tree_model.predict(X_test)


# In[439]:


# Sınıflandırma raporu oluşturma
classification_rep_DT = classification_report(y_test, decision_tree_predictions)
print("Ana Veri Karar Ağaç Sınıflandırma Raporu:")
print(classification_rep_DT)


# In[440]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_DT = confusion_matrix(y_test, decision_tree_predictions)
tn_DT, fp_DT, fn_DT, tp_DT = conf_matrix_DT.ravel()
sensitivity_DT = tp_DT / (tp_DT + fn_DT)
specificity_DT = tn_DT / (tn_DT + fp_DT)
print("Ana Veri Karar Ağaç Sınıflandırma Sensitivity:", sensitivity_DT)
print("Ana Veri Karar Ağaç Sınıflandırma Specificity:", specificity_DT)


# In[441]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_DT, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Ana Veri Karar Ağaç Sınıflandırma Matrisi')
plt.show()


# In[442]:


# ROC eğrisi ve AUC hesaplama
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, decision_tree_predictions)
roc_auc_DT = auc(fpr_DT, tpr_DT)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_DT, tpr_DT, color='darkorange', lw=2, label='Karar Ağacı ROC Eğrisi (AUC = %0.2f)' % roc_auc_DT)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ana Veri Karar Ağaç Sınıflandırma ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# In[318]:


#Karar Ağacı Yapısı
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(decision_tree_model, feature_names=X_scaled.columns, class_names=["0", "1"], filled=True)
plt.show()


# # Hyper Tuning

# In[319]:


grid_param={"criterion":["gini","entropy"],
             "splitter":["best","random"],
             "max_depth":range(2,50,1),
             "min_samples_leaf":range(1,15,1),
             "min_samples_split":range(2,20,1) 
            }


# In[299]:


grid_search=GridSearchCV(estimator=decision_tree_model,param_grid=grid_param,cv=5,n_jobs=-1)


# In[300]:


grid_search.fit(X_train,y_train)


# In[301]:


print(grid_search.best_params_)


# In[320]:


# Karar Ağacı sınıflandırma modelini eğitme
decision_tree_model = DecisionTreeClassifier(random_state=42,criterion= 'entropy',max_depth= 5,min_samples_leaf= 7,min_samples_split= 2,splitter= 'best')
decision_tree_model.fit(X_train, y_train)


# In[321]:


# Test verisi üzerinde modelin performansını değerlendirme ve performans metriklerini hesaplama
decision_tree_predictions = decision_tree_model.predict(X_test)


# In[443]:


# Sınıflandırma raporu oluşturma
classification_rep_DT = classification_report(y_test, decision_tree_predictions)
print("Hypertuning-AnaVeri-Karar Ağaç Sınıflandırma Raporu:")
print(classification_rep_DT)


# In[444]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_DT = confusion_matrix(y_test, decision_tree_predictions)
tn_DT, fp_DT, fn_DT, tp_DT = conf_matrix_DT.ravel()
sensitivity_DT = tp_DT / (tp_DT + fn_DT)
specificity_DT = tn_DT / (tn_DT + fp_DT)
print("Hypertuning-AnaVeri-Karar Ağaç Sensitivity:", sensitivity_DT)
print("Hypertuning-AnaVeri-Karar Ağaç Specificity:", specificity_DT)


# In[445]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_DT, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Hypertuning-AnaVeri-Karar Ağaç Konfüzyon Matrisi')
plt.show()


# In[446]:


# ROC eğrisi ve AUC hesaplama
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, decision_tree_predictions)
roc_auc_DT = auc(fpr_DT, tpr_DT)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_DT, tpr_DT, color='darkorange', lw=2, label='Karar Ağacı ROC Eğrisi (AUC = %0.2f)' % roc_auc_DT)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Hypertuning-AnaVeri-Karar Ağaç ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# In[326]:


#Karar Ağacı Yapısı
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(decision_tree_model, feature_names=X_scaled.columns, class_names=["0", "1"], filled=True)
plt.show()


# # Post Pruning

# In[327]:


path=decision_tree_model.cost_complexity_pruning_path(X_train,y_train)
#path variable gives two things ccp_alphas and impurities
ccp_alphas,impurities=path.ccp_alphas,path.impurities
print("ccp alpha wil give list of values :",ccp_alphas)
print("***********************************************************")
print("Impurities in Decision Tree :",impurities)


# In[328]:


clfs=[]   #will store all the models here
for ccp_alpha in ccp_alphas:
    clf=DecisionTreeClassifier(random_state=42,ccp_alpha=ccp_alpha)
    clf.fit(X_train,y_train)
    clfs.append(clf)
print("Last node in Decision tree is {} and ccp_alpha for last node is {}".format(clfs[-1].tree_.node_count,ccp_alphas[-1]))


# In[329]:


train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",drawstyle="steps-post")
ax.legend()
plt.show()


# In[340]:


clf=DecisionTreeClassifier(random_state=42,ccp_alpha=0.01)
clf.fit(X_train,y_train)
plt.figure(figsize=(8,8), dpi=300)
plot_tree(clf, feature_names=X_scaled.columns, class_names=["0", "1"], filled=True)
plt.show()


# In[341]:


accuracy_score(y_test,clf.predict(X_test))


# # PCA Veri Seti Karar Ağaç Sınıflandırma Algoritması

# In[276]:


# PCA-Karar Ağacı sınıflandırma modelini eğitme
decision_tree_model_pca = DecisionTreeClassifier(random_state=42)
decision_tree_model_pca.fit(X_train_pca, y_train_pca)


# In[277]:


# Test verisi üzerinde modelin performansını değerlendirme ve performans metriklerini hesaplama
decision_tree_predictions_pca = decision_tree_model_pca.predict(X_test_pca)


# In[447]:


# Sınıflandırma raporu oluşturma
classification_rep_DT_pca = classification_report(y_test_pca, decision_tree_predictions_pca)
print("PCA Veri Seti Karar Ağaç Sınıflandırma Raporu:")
print(classification_rep_DT_pca)


# In[448]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_DT_pca = confusion_matrix(y_test_pca, decision_tree_predictions_pca)
tn_DT_pca, fp_DT_pca, fn_DT_pca, tp_DT_pca = conf_matrix_DT_pca.ravel()
sensitivity_DT_pca = tp_DT_pca / (tp_DT_pca + fn_DT_pca)
specificity_DT_pca = tn_DT_pca / (tn_DT_pca + fp_DT_pca)
print("PCA Veri Seti Karar Ağaç Sensitivity:", sensitivity_DT_pca)
print("PCA Veri Seti Karar Ağaç Specificity:", specificity_DT_pca)


# In[449]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_DT_pca, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('PCA Veri Seti Karar Ağaç Konfüzyon Matrisi')
plt.show()


# In[450]:


# ROC eğrisi ve AUC hesaplama
fpr_DT_pca, tpr_DT_pca, thresholds_DT_pca = roc_curve(y_test_pca, decision_tree_predictions_pca)
roc_auc_DT_pca = auc(fpr_DT_pca, tpr_DT_pca)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_DT_pca, tpr_DT_pca, color='darkorange', lw=2, label='Karar Ağacı ROC Eğrisi (AUC = %0.2f)' % roc_auc_DT_pca)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PCA Veri Seti Karar Ağaç ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# In[342]:


#Karar Ağacı Yapısı
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(decision_tree_model_pca, feature_names=X_scaled.columns, class_names=["0", "1"], filled=True)
plt.show()


# # Hyper Tuning

# In[343]:


grid_param_pca={"criterion":["gini","entropy"],
             "splitter":["best","random"],
             "max_depth":range(2,50,1),
             "min_samples_leaf":range(1,15,1),
             "min_samples_split":range(2,20,1) 
            }


# In[344]:


grid_search_pca=GridSearchCV(estimator=decision_tree_model_pca,param_grid=grid_param_pca,cv=5,n_jobs=-1)


# In[345]:


grid_search_pca.fit(X_train_pca,y_train_pca)


# In[346]:


print(grid_search.best_params_)


# In[347]:


# Karar Ağacı sınıflandırma modelini eğitme
decision_tree_model_pca = DecisionTreeClassifier(random_state=42,criterion= 'entropy',max_depth= 5,min_samples_leaf= 7,min_samples_split= 2,splitter= 'best')
decision_tree_model_pca.fit(X_train_pca, y_train_pca)


# In[348]:


# Test verisi üzerinde modelin performansını değerlendirme ve performans metriklerini hesaplama
decision_tree_predictions_pca = decision_tree_model_pca.predict(X_test_pca)


# In[451]:


# Sınıflandırma raporu oluşturma
classification_rep_DT_pca = classification_report(y_test_pca, decision_tree_predictions_pca)
print("HyperT-PCA Veri Seti Karar Ağaç Sınıflandırma Raporu:")
print(classification_rep_DT_pca)


# In[452]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_DT_pca = confusion_matrix(y_test_pca, decision_tree_predictions_pca)
tn_DT_pca, fp_DT_pca, fn_DT_pca, tp_DT_pca = conf_matrix_DT_pca.ravel()
sensitivity_DT_pca = tp_DT_pca / (tp_DT_pca + fn_DT_pca)
specificity_DT_pca = tn_DT_pca / (tn_DT_pca + fp_DT_pca)
print("HyperT-PCA Veri Seti Karar Ağaç Sensitivity:", sensitivity_DT_pca)
print("HyperT-PCA Veri Seti Karar Ağaç Specificity:", specificity_DT_pca)


# In[453]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_DT_pca, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('HyperT-PCA Veri Seti Karar Ağaç Konfüzyon Matrisi')
plt.show()


# In[454]:


# ROC eğrisi ve AUC hesaplama
fpr_DT_pca, tpr_DT_pca, thresholds_DT_pca = roc_curve(y_test_pca, decision_tree_predictions_pca)
roc_auc_DT_pca = auc(fpr_DT_pca, tpr_DT_pca)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_DT_pca, tpr_DT_pca, color='darkorange', lw=2, label='Karar Ağacı ROC Eğrisi (AUC = %0.2f)' % roc_auc_DT_pca)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('HyperT-PCA Veri Seti Karar Ağaç ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# In[353]:


#Karar Ağacı Yapısı
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(decision_tree_model_pca, feature_names=X_scaled.columns, class_names=["0", "1"], filled=True)
plt.show()


# # Post Pruning

# In[354]:


path_pca=decision_tree_model_pca.cost_complexity_pruning_path(X_train_pca,y_train_pca)
#path variable gives two things ccp_alphas and impurities
ccp_alphas_pca,impurities_pca=path_pca.ccp_alphas,path_pca.impurities
print("ccp alpha wil give list of values :",ccp_alphas_pca)
print("***********************************************************")
print("Impurities in Decision Tree :",impurities_pca)


# In[355]:


clfs_pca=[]   #will store all the models here
for ccp_alpha in ccp_alphas_pca:
    clf_pca=DecisionTreeClassifier(random_state=42,ccp_alpha=ccp_alpha)
    clf_pca.fit(X_train_pca,y_train_pca)
    clfs_pca.append(clf_pca)
print("Last node in Decision tree is {} and ccp_alpha for last node is {}".format(clfs_pca[-1].tree_.node_count,ccp_alphas_pca[-1]))


# In[356]:


train_scores_pca = [clf_pca.score(X_train_pca, y_train_pca) for clf_pca in clfs_pca]
test_scores_pca = [clf_pca.score(X_test_pca, y_test_pca) for clf_pca in clfs_pca]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas_pca, train_scores_pca, marker='o', label="train",drawstyle="steps-post")
ax.plot(ccp_alphas_pca, test_scores_pca, marker='o', label="test",drawstyle="steps-post")
ax.legend()
plt.show()


# In[365]:


clf_pca=DecisionTreeClassifier(random_state=42,ccp_alpha=0.01)
clf_pca.fit(X_train_pca,y_train_pca)
plt.figure(figsize=(8,8), dpi=300)
plot_tree(clf_pca, feature_names=X_scaled.columns, class_names=["0", "1"], filled=True)
plt.show()


# In[366]:


accuracy_score(y_test_pca,clf_pca.predict(X_test_pca))


# # Ana Veri Seti Naive Bayes

# In[282]:


#Naive Bayes sınıflandırıcısını eğitimi
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)


# In[283]:


#Test verisi üzerinde modelin performansını değerlendirme ve performans metriklerini hesaplama
naive_bayes_predictions = naive_bayes_model.predict(X_test)


# In[462]:


# Sınıflandırma raporu oluşturma
classification_rep_NB = classification_report(y_test, naive_bayes_predictions)
print("Ana Veri Seti Naive Bayes Sınıflandırma Raporu:")
print(classification_rep_NB)


# In[461]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_NB = confusion_matrix(y_test, naive_bayes_predictions)
tn_NB, fp_NB, fn_NB, tp_NB = conf_matrix_NB.ravel()
sensitivity_NB = tp_NB / (tp_NB + fn_NB)
specificity_NB = tn_NB / (tn_NB + fp_NB)
print("Ana Veri Seti Naive Bayes Sensitivity:", sensitivity_NB)
print("Ana Veri Seti Naive Bayes Specificity:", specificity_NB)


# In[460]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_NB, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Ana Veri Seti Naive Bayes Konfüzyon Matrisi')
plt.show()


# In[459]:


# ROC eğrisi ve AUC hesaplama
fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, naive_bayes_predictions)
roc_auc_NB = auc(fpr_NB, tpr_NB)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_NB, tpr_NB, color='darkorange', lw=2, label='Navie Bayes ROC Eğrisi (AUC = %0.2f)' % roc_auc_NB)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ana Veri Seti Naive Bayes ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# # PCA Veri Seti Naive Bayes

# In[288]:


#PCA-Naive Bayes sınıflandırıcısını eğitimi
naive_bayes_model_pca = GaussianNB()
naive_bayes_model_pca.fit(X_train_pca, y_train_pca)


# In[289]:


#Test verisi üzerinde modelin performansını değerlendirme ve performans metriklerini hesaplama
naive_bayes_predictions_pca = naive_bayes_model_pca.predict(X_test_pca)


# In[463]:


# Sınıflandırma raporu oluşturma
classification_rep_NB_pca = classification_report(y_test_pca, naive_bayes_predictions_pca)
print("PCA Veri Seti Naive Bayes Sınıflandırma Raporu:")
print(classification_rep_NB_pca)


# In[464]:


# Hassasiyet (Sensitivity) ve Özgünlük (Specificity) hesaplama
conf_matrix_NB_pca = confusion_matrix(y_test_pca, naive_bayes_predictions_pca)
tn_NB_pca, fp_NB_pca, fn_NB_pca, tp_NB_pca = conf_matrix_NB_pca.ravel()
sensitivity_NB_pca = tp_NB_pca / (tp_NB_pca + fn_NB_pca)
specificity_NB_pca = tn_NB_pca / (tn_NB_pca + fp_NB_pca)
print("PCA Veri Seti Naive Bayes Sensitivity:", sensitivity_NB_pca)
print("PCA Veri Seti Naive Bayes Specificity:", specificity_NB_pca)


# In[465]:


# Konfüzyon matrisi hesaplama
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_NB_pca, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('PCA Veri Seti Naive Bayes Konfüzyon Matrisi')
plt.show()


# In[466]:


# ROC eğrisi ve AUC hesaplama
fpr_NB_pca, tpr_NB_pca, thresholds_NB_pca = roc_curve(y_test_pca, naive_bayes_predictions_pca)
roc_auc_NB_pca = auc(fpr_NB_pca, tpr_NB_pca)

# ROC eğrisini çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr_NB_pca, tpr_NB_pca, color='darkorange', lw=2, label='Navie Bayes ROC Eğrisi (AUC = %0.2f)' % roc_auc_NB_pca)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PCA Veri Seti Naive Bayes ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




