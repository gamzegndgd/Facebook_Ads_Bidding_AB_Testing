##################################################################
## AB TESTİ ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
##################################################################

################################## AB TEST PROJECT ######################################

## Facebook ta maxımum bıddıng ve average bıddıng adında iki ürün var
## Biz bombabombacom olarak facebook reklamlarını max bıddıng ya da average bıddıng ıle verebılırız.
## max bıddıng , average bıddıng ne demek ? bombabombacom un reklam vermek ıstedıgı noktada hepsıburada ,
## yatcazkalkcaz.com da reklam vermek ıstıyor. dolayısıyla burda reklam vermek ısteyen baska kısıler var.
## aynı noktada bırden fazla kısı reklam vermek ıstedıgınde acık arttırma olur bu e-tıcaret tarafında otel sektorunde vs
## reklam gosterılecek bırcok noktada bıddıng kavramı vardır.
## facebook un da ıkı yontemı varmıs max bıddıng : bıddıngı kazananın fıyatı en fazla bıddı veren tarafından olur
## average bıddıng : bıddıngı kazandıktan sonra reklam vermenın bırım fıyatının ort. bid fiyatı
## mesela 3 kısı fıyat verdı bu 3 kısının verdıgı fıyatların ort. sı uzerınden reklam verılır
## bombabombacom dıyor kı ben hangı yontemle facebookta reklam vermelıyım?? bılmıyorum ama bır guzellık
## yapıp ıkı tarafta da reklam verdım ve bunların kaydını tuttum sana getırdım verı bılımcı. 4 konuda beklentım var.
## bu 4 konudakı beklentıme karsılık bana bır rapor olustur ve benı ıkna et der bombabombacom
## hangı yontemı secmemız gerektıgını anlamamız ıcın ab testı yapmamız beklenıyor.

## iki grup var kontrol ve test grubu
## odagımız satın alma ( purchase )
###########################################################################################

###### İŞ PROBLEMİ #######
## Facebook kısa süre önce mevcut maximum bidding adı verilen teklif verme türüne alternatif olarak
## yeni bir teklif türü olan average bidding’i tanıttı.Müşterilerimizden biri olan bombabomba.com,
## bu yeni özelliği test etmeye karar verdi ve averagebidding’in, maximumbidding’den daha fazla
## dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.
## Maximum Bidding: Maksimum teklif verme / Average Bidding: Average teklif verme

###### VERİ SETİ HİKAYESİ #######
## bombabomba.com’un web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
## reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.
## Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır.

###### DEĞİŞKENLER #######
## Impression – Reklam görüntüleme sayısı
## Click – Tıklama ( Görüntülenen reklama tıklanma sayısını belirtir. )
## Purchase – Satın alım ( Tıklanan reklamlar sonrası satın alınan ürün sayısını belirtir.)
## Earning – Kazanç ( Satın alınan ürünler sonrası elde edilen kazanç )

## control group = maximum bidding
## test group = average bidding

########### GEREKLİ IMPORT İŞLEMLERİ ##############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import shapiro, levene, ttest_ind
import itertools
import seaborn as sns
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#####################################################
# Proje Görevleri
#####################################################

#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.


dataframe_control = pd.read_excel("ab_testing.xlsx" , sheet_name="Control Group")
dataframe_test = pd.read_excel("ab_testing.xlsx" , sheet_name="Test Group")

df_control = dataframe_control.copy()
df_test = dataframe_test.copy()


# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df_control)
check_df(df_test)


#df_control.shape   ### 40 gözlem 4 değişken
#df_test.shape      ### 40 gözlem 4 değişken

#df_control.isnull().sum()   ### CONTROL GRUBUNDA KAYIP GÖZLEMİMİZ YOK
#df_test.isnull().sum()      ### TEST GRUBUNDA KAYIP GÖZLEMİMİZ YOK

## VERİ SETİNDE AYKIRI GÖZLEM SAYILABİLECEK UÇ DEĞERDE BİR GÖZLEM YOK AYRICA VERI SETİMİZİN GÖZLEM SAYISI OLDUKÇA AZ
## OLDUĞU İÇİN DAHA FAZLA BİLGİ KAYBETMEMEK ADINA AYKIRI GÖZLEM TEMİZLEME İŞLEMİ YAPILMAMIŞTIR.

#df_control.info()
#df_test.info()


## TANIMLAYICI İSTATİSTİKLERE BAKIYORUZ. İLGİLENDİĞİMİZ PURCHASE OLDUGU ICIN;

df_control.describe().T   ## CONTROL GROUP PURCHASE MEAN : 550.89406

##               count         mean         std
## Impression 40.00000 101711.44907 20302.15786
## Click      40.00000   5100.65737  1329.98550
## Purchase   40.00000    550.89406   134.10820
## Earning    40.00000   1908.56830   302.91778

df_test.describe().T      ## TEST GROUP PURCHASE MEAN : 582.10610

##               count         mean         std
## Impression 40.00000 120512.41176 18807.44871
## Click      40.00000   3967.54976   923.09507
## Purchase   40.00000    582.10610   161.15251
## Earning    40.00000   2514.89073   282.73085

## GÜVEN ARALIKLARI
sms.DescrStatsW(df_control["Purchase"]).tconfint_mean()
## CONTROL GROUP PURCHASE CONFIDENCE INTERVAL (508.0041754264924, 593.7839421139709)

sms.DescrStatsW(df_test["Purchase"]).tconfint_mean()
## TEST GROUP PURCHASE CONFIDENCE INTERVAL (530.5670226990063, 633.645170597929)

## KONTROL GRUBU VE TEST GRUBU PURCHASE ORTALAMALARI BIRBIRINDEN FARKLI
## AVERAGE BIDDINGIN GETIRDIGI DONUSUM MAXIMUM BIDDINGIN GETIRDIGI DONUSUMDEN FAZLA GÖRUNUYOR.
## ANCAK BU FARKLILIK ŞANS ESERİ Mİ OLUŞTU YOKSA İSTATİSTİKSEL OLARAK BİR ANLAMI VAR MI??
## İSTATİSTİKSEL OLARAK ANLAMLILIĞI OLUP OLMADIĞINI GÖRMEK İÇİN HİPOTEZ TESTİ YAPARIZ.

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

# iki farklı veri setini birleştirmeden önce ayrımlarını yapabilmemiz için 2 veriyi de etiketliyoruz.
# group isimli bir kolon oluşturup hangisinin hangi gruba ait olduğunu yazıyoruz.
df_control["group"] = "control"
df_test["group"] = "test"

# alt alta birlestirme olacağından axis=0 yapıyoruz. eğer 1 verseydik yan yana birleşirdi.
df = pd.concat([df_control,df_test], axis=0,ignore_index=True)
# ignore_index=True vererek de ilk veri biitp 2.veri setine geçtiğinde indeksi 0'dan değil kaldığı yerden birleştirmeye devam edecek
df.head()

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# H0 : M1 = M2  (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
# H1 : M1!= M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)

### İKİ GRUP FARKLILIKLARINI KARŞILAŞTIRMAK İÇİN BAĞIMSIZ ÖRNEKLEM T TESTİ KULLANILIR.

## H0 : M1 >= M2 (AVERAGE BIDDING'IN GETIRDIGI DONUSUM MAXIMUM BIDDING'IN GETIRDIGI DONUSUMDEN FAZLADIR.)

## H0 : M1 = M2 (AVERAGE BIDDING'IN GETIRDIGI DONUSUM İLE MAXIMUM BIDDING'IN GETIRDIGI DONUSUM ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK YOKTUR.)

## HİPOTEZİ İKİ ŞEKİLDE KURABİLİRİZ. MEVCUT FARKLILIGIN HANGI GRUPTAN KAYNAKLI OLDUĞU BELLİDİR. BU SEBEPLE İKİ YÖNLÜ HİPOTEZ İLE DEVAM EDİYORUZ.

## H0 : M1 = M2 (AVERAGE BIDDING'IN GETIRDIGI DONUSUM İLE MAXIMUM BIDDING'IN GETIRDIGI DONUSUM ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK YOKTUR.)
## H1 : M1 != M2 (AVERAGE BIDDING'IN GETIRDIGI DONUSUM İLE MAXIMUM BIDDING'IN GETIRDIGI DONUSUM ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.)


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

df.groupby("group").agg({"Purchase": "mean"})

##          Purchase
## group
## control 550.89406
## test    582.10610


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.
# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır
# p < 0.05 H0 RED
# p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ?
# Elde edilen p-valuedeğerlerini yorumlayınız.


test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.5891
# HO reddedilemez. Control grubunun değerleri normal dağılım varsayımını sağlamaktadır.

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
## Test Stat = 0.9589, p-value = 0.1541
## P-VALUE > ALFA = 0.05 OLDUGU ICIN H0 REDDEDILEMEZ. NORMAL DAGILIM VARSAYIMI TEST GRUBU İÇİN SAĞLANMAKTADIR.


### VARYANS HOMOJENLİĞİ VARSAYIMI

## H0: VARYANSLAR HOMOJENDİR.
## H1: VARYANSLAR HOMOJEN DEĞİLDİR.
# p < 0.05 H0 RED
# p > 0.05 H0 REDDEDİLEMEZ

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.1083
# HO reddedilemez. Control ve Test grubunun değerleri varyans homojenliği varsayımını sağlamaktadır.
# Varyanslar Homojendir.


# Adım 2: Normallik Varsayımı ve VaryansHomojenliği sonuçlarına göre uygun testi seçiniz

## İKİ VARSAYIM DA SAĞLANDIĞI İÇİN BAĞIMSIZ ÖRNEKLEM T TESTİ UYGULANIR.

### HİPOTEZİMİZİ TEKRAR GETİRİYORUZ.

## H0 : M1 = M2 (AVERAGE BIDDING'IN GETIRDIGI DONUSUM İLE MAXIMUM BIDDING'IN GETIRDIGI DONUSUM ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK YOKTUR.)
## H1 : M1 != M2 (AVERAGE BIDDING'IN GETIRDIGI DONUSUM İLE MAXIMUM BIDDING'IN GETIRDIGI DONUSUM ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.)
# p < 0.05 H0 RED # p > 0.05 H0 REDDEDİLEMEZ

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Adım 3: Test sonucunda elde edilen p_valuedeğerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

## Test Stat = -0.9416, p-value = 0.3493
## P-VALUE > ALFA = 0.05 OLDUGU ICIN H0 REDDEDILEMEZ.
## %95 GÜVENLE AVERAGE BIDDING'IN GETIRDIGI DONUSUM(SATIN ALMA ORT.) İLE MAXIMUM BIDDING'IN GETIRDIGI DONUSUM(SATIN ALMA ORT.)
## ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK YOKTUR.

## Kontrol ve test grubu satın alma ortalamaları arasında istatistiksel olarak anlamlı farklılık yoktur.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

### NORMALLIK VARSAYIMI İÇİN SHAPIRO-WILK TESTİ, VARYANS HOMOJENLIGI VARSAYIMI İÇİN LEVENE TESTİ KULLANILMIŞTIR.
### VARSAYIMLAR SAĞLANDIĞI İÇİN İKİ GRUP FARKLILIĞINI KARŞILAŞTIRMAK İÇİN BAĞIMSIZLIK ÖRNEKLEM T TESTİ KULLANILMIŞTIR.
### EĞER NORMALLİK VARSAYIMI BİR GRUP İÇİN BİLE SAĞLANMASA İKİ GRUP FARKLILIĞINI KARŞILAŞTIRMAK İÇİN MANNWHITNEYU TESTİ KULLANILIRDI.
### EĞER VARYANS HOMOJENLIĞI VARSAYIMI SAĞLANMASAYDI İKİ GRUP FARKLILIĞINI KARŞILAŞTIRMAK İÇİN WELCH TESTİ KULLANILIRDI.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.


## HIPOTEZ TESTİ SONUCU YENI ALTERNATIF URUN OLAN AVERAGE BIDDING'IN GETIRDIGI SATIN ALMA ORTALAMALARI İLE
## ESKİ ÜRÜN MAXIMUM BIDDING'IN GETIRDIGI SATIN ALMA ORTALAMALARI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK
## OLMADIĞI SONUCUNA ULAŞILDI.

## BU TEST 40 AR GÖZLEM İÇEREN VERİ SETLERİ İLE YAPILDI. GÖZLEM SAYISI NE KADAR FAZLA OLURSA GERÇEĞE YAKIN SONUÇ ÜRETME
## DURUMU DA ARTACAĞI İÇİN VERİ SETİNDEKİ GÖZLEM SAYISINI ARTIMAK AMACIYLA ESKİ VE YENİ ÜRÜNLERİ BİR SÜRE DAHA İNCELEMEK
## DAHA İYİ OLACAKTIR. HER İKİ ÜRÜNDEKİ SATIN ALMA MİKTARLARININ ARTMASI SAĞLANARAK YENİ VERİ SETLERİ İLE DAHA SAĞLIKLI
## SONUÇLARA ULAŞMAK İÇİN TESTLER TEKRARLANMALIDIR. BÖYLE BİR İMKAN YA DA VAKİT YOKSA MEVCUT KULLANILAN ÜRÜN İLE
## DEVAM EDİLMESİ ÖNERİLİR.

## AYRICA MEVCUT VERİ SETLERİNDE GÖZLEM SAYISININ AZ OLMASI SEBEBİYLE VE DAHA FAZLA GÖZLEM ÇIKARMAMAK AMACIYLA AYKIRI
## GÖZLEM ÇALIŞMASI YAPILMAMIŞTIR. GÖZLEM SAYISI ARTTIĞI VE ELDE YETERLİ VERİ OLDUĞU DURUMDA DOĞRU BİR AYKIRI GÖZLEM
## VE VERİ TEMİZLİĞİ VE ÖN İŞLEMESİ YAPILMALIDIR.

## ÖTE YANDAN ARAŞTIRMALARIMIZI GENİŞLETEBİLİR, TEK BAŞINA SATIN ALMA KARŞILAŞTIRMASINA YÖNELİK DEĞİL
## TIKLANMA İSTATİSTİKLERİNE, İZLENME SAYILARINA BAKILARAK TIKLANMANIN GETİRDİĞİ SATIN ALMA ORANLARI İLE
## SİTEYİ ZİYARET EDEN KULLANICILARIN GEİTRDİĞİ SATIN ALMA ORANLARINA BAKILARAK HANGİ ÜRÜNÜN DAHA KAZANÇLI OLDUĞU
## ARAŞTIRMASI YAPILABİLİR.

## BU SEBEPLE FARKLI SONUÇLARA ULAŞMAK AMACIYLA İKİ ÖRNEKLEM ORAN TESTLERİ İLE ARAŞTIRMAMIZA DEVAM EDİYORUZ.

############################# KARŞILAŞTIRMA İÇİN FARKLI METRİKLERE BAKIYORUZ ##########################################

## CONVERSION RATE : DÖNÜŞÜM ORANI ( PURCHASE / IMPRESSION ) ( SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ SATIN ALMA ORANI )
## H0 : M1 = M2 (AVERAGE BIDDING İLE MAXIMUM BIDDING'IN SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ SATIN ALMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK YOKTUR.)
## H1 : M1 != M2 (AVERAGE BIDDINGİLE MAXIMUM BIDDING'IN SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ SATIN ALMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.)

control_prop = df_control["Purchase"].sum()/df_control["Impression"].sum() * 100   ### Kontrol grubunda gösterimlerin yaklaşık %0.54’ü satın almaya dönüşmüş.
test_prop = df_test["Purchase"].sum()/df_test["Impression"].sum() * 100            ### Test grubunda gösterimlerin yaklaşık %0.48’i satın almaya dönüşmüş.
                                                                                   ## ORANDA DÜŞÜŞ VAR

test_stat, pvalue = proportions_ztest(count=[df_control["Purchase"].sum(), df_test["Purchase"].sum()],
                                      nobs=[df_control["Impression"].sum(), df_test["Impression"].sum()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

### Test Stat = 12.2212, p-value = 0.0000
## P-VALUE < ALFA = 0.05 OLDUGU ICIN H0 RED. %95 GÜVENLE AVERAGE BIDDING İLE MAXIMUM BIDDING'IN SİTEYİ ZİYARET EDENLERİN
## GETİRDİĞİ SATIN ALMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.
## YENİ ÖZELLİK OLAN AVERAGE BIDDING İLE SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ SATIN ALMA ORANININ İSTATİSTİKSEL
## OLARAK AZALDIĞINI SÖYLEYEBİLİRİZ.


## CLICK THROUGH RATE : TIKLANMA ORANI ( CLICK / IMPRESSION ) ( SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ TIKLANMA ORANI )
## H0 : M1 = M2 (AVERAGE BIDDING İLE MAXIMUM BIDDING'IN SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ TIKLANMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK YOKTUR.)
## H1 : M1 != M2 (AVERAGE BIDDINGİLE MAXIMUM BIDDING'IN SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ TIKLANMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.)

control_prop = df_control["Click"].sum()/df_control["Impression"].sum() * 100      ### Kontrol grubunda gösterilen reklamların yaklaşık %5’i tıklamaya dönüşmüştür.
test_prop = df_test["Click"].sum()/df_test["Impression"].sum() * 100               ### Test grubunda reklamı görüntüleyenlerin %3,3'ü reklama tıklamış.
                                                                                   ## ORANDA DÜŞÜŞ VAR
test_stat, pvalue = proportions_ztest(count=[df_control["Click"].sum(), df_test["Click"].sum()],
                                      nobs=[df_control["Impression"].sum(), df_test["Impression"].sum()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

### Test Stat = 129.3305, p-value = 0.0000
## P-VALUE < ALFA = 0.05 OLDUGU ICIN H0 RED. %95 GÜVENLE AVERAGE BIDDING İLE MAXIMUM BIDDING'IN SİTEYİ ZİYARET EDENLERİN
## GETİRDİĞİ TIKLANMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.
## YENİ ÖZELLİK OLAN AVERAGE BIDDING İLE SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ TIKLANMA ORANININ İSTATİSTİKSEL
## OLARAK AZALDIĞINI SÖYLEYEBİLİRİZ.


## TIKLANMA BAŞINA SATIN ALMA ORANI ( PURCHASE / CLICK ) ( TIKLANMALARIN GETİRDİĞİ SATIN ALMA ORANI )
## H0 : M1 = M2 (AVERAGE BIDDING İLE MAXIMUM BIDDING'IN TIKLANMALARIN GETİRDİĞİ SATIN ALMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK YOKTUR.)
## H1 : M1 != M2 (AVERAGE BIDDINGİLE MAXIMUM BIDDING'IN TIKLANMALARIN GETİRDİĞİ SATIN ALMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.)

control_prop = df_control["Purchase"].sum()/df_control["Click"].sum() * 100    ### %10
test_prop = df_test["Purchase"].sum()/df_test["Click"].sum() * 100             ### %14      ## ORANDA ARTIŞ VAR

test_stat, pvalue = proportions_ztest(count=[df_control["Purchase"].sum(), df_test["Purchase"].sum()],
                                      nobs=[df_control["Click"].sum(), df_test["Click"].sum()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

### Test Stat = -34.9800, p-value = 0.0000
## P-VALUE < ALFA = 0.05 OLDUGU ICIN H0 RED. %95 GÜVENLE AVERAGE BIDDING İLE MAXIMUM BIDDING'IN SİTEYİ ZİYARET EDENLERİN
## GETİRDİĞİ SATIN ALMA ORANI ARASINDA İSTATİSTİKSEL OLARAK ANLAMLI BİR FARK VARDIR.
## YENİ ÖZELLİK OLAN AVERAGE BIDDING İLE TIKLANMALARIN GETİRDİĞİ SATIN ALMA ORANININ İSTATİSTİKSEL
## OLARAK ARTTIĞINI SÖYLEYEBİLİRİZ.


### YUKARIDAKİ METRİK ARAŞTIRMALARI YAPILDIKTAN SONRA ELDE EDİLEN SONUÇLARA GÖRE;
### SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ SATIN ALMA ORANININ YENİ ÖZELLİK OLAN AVERAGE BIDDING İLE DÜŞTÜĞÜ,
### SİTEYİ ZİYARET EDENLERİN GETİRDİĞİ TIKLANMA ORANININ YENİ ÖZELLİK OLAN AVERAGE BIDDING İLE DÜŞTÜĞÜ,
### ANCAK YENİ ÖZELLİK AVERAGE BIDDING İLE TIKLANMALARIN GETİRDİĞİ SATIN ALMA ORANININ ARTTIĞI İSTATİSTİKSEL OLARAK
### %95 GÜVENLE GÖZLENMİŞTİR.

### GÖREV-4 DE YER ALAN AÇIKLAMALAR GEÇERLİ OLMAKLA BİRLİKTE TIKLANMA İLE SATIN ALMA İLİŞKİSİ DE YAKINDAN İNCELENEBİLİR.