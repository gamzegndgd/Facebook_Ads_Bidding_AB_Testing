##################################################################
## AB TESTİ ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
##################################################################

################################## AB TEST PROJECT ######################################
###### İŞ PROBLEMİ #######
## Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif olarak yeni bir teklif türü olan "average bidding"’i tanıttı.
Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test etmeye karar verdi ve average bidding'inmaximumbidding'den daha fazla dönüşüm getirip 
getirmediğini anlamak için bir A/Btesti yapmak istiyor. A/B testi 1 aydır devam ediyor ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını 
analiz etmenizi bekliyor.Bombabomba.com için nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.

###########################################################################################

## Maximum Bidding: Maksimum teklif verme 
## Average Bidding: Average teklif verme

#################################### VERİ SETİ HİKAYESİ #####################################
Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra
buradan gelen kazanç bilgileri yer almaktadır. Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri
ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır. Kontrol grubuna Maximum Bidding, test grubuna Average
Bidding uygulanmıştır.

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
import statsmodels.stats.api as sms
from scipy.stats import shapiro, levene, ttest_ind
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


########################## Proje Görevleri ##################################################### 

########################## Görev 1:  Veriyi Hazırlama ve Analiz Etme ########################## 

### Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

dataframe_control = pd.read_excel("ab_testing.xlsx" , sheet_name="Control Group")
dataframe_test = pd.read_excel("ab_testing.xlsx" , sheet_name="Test Group")

df_control = dataframe_control.copy()
df_test = dataframe_test.copy()


### Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

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

#df_control.isnull().sum()   ### control grubunda kayıp gözlemimiz yok
#df_test.isnull().sum()      ### test grubunda kayıp gözlemimiz yok

## Veri setinde aykırı gözlem sayılabilecek uç değerde bir gözlem yok ayrıca verı setimizin gözlem sayısı oldukça az olduğu için daha fazla bilgi 
# kaybetmemek adına aykırı gözlem temizleme işlemi yapılmamıştır.

#df_control.info()
#df_test.info()


# Tanımlayıcı istatistiklere bakılır.
df_control.describe().T   ## Control group purchase mean : 550.89406

##               count         mean         std
## Impression 40.00000 101711.44907 20302.15786
## Click      40.00000   5100.65737  1329.98550
## Purchase   40.00000    550.89406   134.10820
## Earning    40.00000   1908.56830   302.91778

df_test.describe().T      ## Test control group purchase mean : 582.10610

##               count         mean         std
## Impression 40.00000 120512.41176 18807.44871
## Click      40.00000   3967.54976   923.09507
## Purchase   40.00000    582.10610   161.15251
## Earning    40.00000   2514.89073   282.73085

## Güven Aralıkları

sms.DescrStatsW(df_control["Purchase"]).tconfint_mean()
## Control group purchase confıdence ınterval (508.0041754264924, 593.7839421139709)

sms.DescrStatsW(df_test["Purchase"]).tconfint_mean()
## Test group purchase confıdence ınterval (530.5670226990063, 633.645170597929)

## Kontrol grubu ve test grubu purchase ortalamaları bırbırınden farklı
## average bıddıngın getırdıgı donusum maxımum bıddıngın getırdıgı donusumden fazla görunuyor.
## ancak bu farklılık şans eseri mi oluştu yoksa istatistiksel olarak bir anlamı var mı?
## istatistiksel olarak anlamlılığı olup olmadığını görmek için hipotez testi yapılır.


### Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

# iki farklı veri setini birleştirmeden önce ayrımlarını yapabilmemiz için 2 veriyi de etiketlenir.
# group isimli bir kolon oluşturup hangisinin hangi gruba ait olduğunu yazılır.

df_control["group"] = "control"
df_test["group"] = "test"


df = pd.concat([df_control,df_test], axis=0,ignore_index=True)
# alt alta birlestirme olacağından axis=0 yapılır, eğer 1 verseydik yan yana birleşirdi.
# ignore_index=True vererek de ilk veri bitip 2.veri setine geçtiğinde indeksi 0'dan değil kaldığı yerden birleştirmeye devam edecek
df.head()

################################### Görev 2:  A/B Testinin Hipotezinin Tanımlanması ####################################

### Adım 1: Hipotezi tanımlayınız.

# H0 : M1 = M2  (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
# H1 : M1!= M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)

# İki grup farklılıklarını karşılaştırmak için bağımsız örneklem T testi kullanılır.

# H0 : M1 >= M2 (Average bıddıng'ın getırdıgı donusum maxımum bıddıng'ın getırdıgı donusumden fazladır.)
# H0 : M1 = M2 (Average bıddıng'ın getırdıgı donusum ile maxımum bıddıng'ın getırdıgı donusum arasında istatistiksel olarak anlamlı bir fark yoktur)

# Hipotezi iki şekilde kurabiliriz. mevcut farklılıgın hangı gruptan kaynaklı olduğu bellidir. bu sebeple iki yönlü hipotez ile devam ediyoruz.

# H0 : M1 = M2 (Average bıddıng'ın getırdıgı donusum ile maxımum bıddıng'ın getırdıgı donusum arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1 : M1 != M2 (Average bıddıng'ın getırdıgı donusum ile maxımum bıddıng'ın getırdıgı donusum arasında istatistiksel olarak anlamlı bir fark vardır.)


### Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

df.groupby("group").agg({"Purchase": "mean"})

##          Purchase
## group
## control 550.89406
## test    582.10610


######################################## GÖREV 3: Hipotez Testinin Gerçekleştirilmesi #######################################################

### Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.
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
# Test Stat = 0.9589, p-value = 0.1541
# P-VALUE > ALFA = 0.05 oldugu ıcın h0 reddedılemez. normal dagılım varsayımı test grubu için sağlanmaktadır.

# Varyans Homojenliği Varsayımı:
# H0: varyanslar homojendir.
# H1: varyanslar homojen değildir.
# p < 0.05 H0 RED
# p > 0.05 H0 REDDEDİLEMEZ

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.1083
# HO reddedilemez. Control ve Test grubunun değerleri varyans homojenliği varsayımını sağlamaktadır.
# Varyanslar Homojendir.


### Adım 2: Normallik Varsayımı ve VaryansHomojenliği sonuçlarına göre uygun testi seçiniz

# İki varsayım da sağlandığı için bağımsız örneklem t testi uygulanır.
# Hipotezimizi tekrar getiriyoruz.
# H0 : M1 = M2 (average bıddıng'ın getırdıgı donusum ile maxımum bıddıng'ın getırdıgı donusum arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1 : M1 != M2 (average bıddıng'ın getırdıgı donusum ile maxımum bıddıng'ın getırdıgı donusum arasında istatistiksel olarak anlamlı bir fark vardır.)

# p < 0.05 H0 RED # p > 0.05 H0 REDDEDİLEMEZ

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


### Adım 3: Test sonucunda elde edilen p_valuedeğerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki 
#olarak anlamlı bir fark olup olmadığını yorumlayınız.

# Test Stat = -0.9416, p-value = 0.3493
# P-VALUE > ALFA = 0.05 oldugu ıcın h0 reddedılemez.
# %95 güvenle average bıddıng'ın getırdıgı donusum(satın alma ort.) ile maxımum bıddıng'ın getırdıgı donusum(satın alma ort.)
# arasında istatistiksel olarak anlamlı bir fark yoktur.
# Kontrol ve test grubu satın alma ortalamaları arasında istatistiksel olarak anlamlı farklılık yoktur.

######################################### GÖREV 4 : Sonuçların Analizi ###################################################################

### Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# Normallık varsayımı için shapıro-wılk testi, varyans homojenlıgı varsayımı için levene testi kullanılmıştır.
# varsayımlar sağlandığı için iki grup farklılığını karşılaştırmak için bağımsızlık örneklem t testi kullanılmıştır.
# eğer normallik varsayımı bir grup için bile sağlanmasa iki grup farklılığını karşılaştırmak için mannwhıtneyu testi kullanılırdı.
# eğer varyans homojenlığı varsayımı sağlanmasaydı iki grup farklılığını karşılaştırmak için welch testi kullanılırdı.



### Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Hıpotez testi sonucu yenı alternatıf urun olan average bıddıng'ın getırdıgı satın alma ortalamaları ile
# eski ürün maxımum bıddıng'ın getırdıgı satın alma ortalamaları arasında istatistiksel olarak anlamlı bir fark olmadığı sonucuna ulaşıldı.

# Bu test 40'ar gözlem içeren veri setleri ile yapıldı. gözlem sayısı ne kadar fazla olursa gerçeğe yakın sonuç üretme
# durumu da artacağı için veri setindeki gözlem sayısını artımak amacıyla eski ve yeni ürünleri bir süre daha incelemek
# daha iyi olacaktır. her iki üründeki satın alma miktarlarının artması sağlanarak yeni veri setleri ile daha sağlıklı
# sonuçlara ulaşmak için testler tekrarlanmalıdır. böyle bir imkan ya da vakit yoksa mevcut kullanılan ürün ile devam edilmesi önerilir.

# Ayrıca mevcut veri setlerinde gözlem sayısının az olması sebebiyle ve daha fazla gözlem çıkarmamak amacıyla aykırı gözlem çalışması yapılmamıştır. 
# gözlem sayısı arttığı ve elde yeterli veri olduğu durumda doğru bir aykırı gözlem ve veri temizliği ve ön işlemesi yapılmalıdır.

# Öte yandan araştırmalarımızı genişletebilir, tek başına satın alma karşılaştırmasına yönelik değil tıklanma istatistiklerine, izlenme sayılarına bakılarak tıklanmanın
# getirdiği satın alma oranları ile siteyi ziyaret eden kullanıcıların geitrdiği satın alma oranlarına bakılarak hangi ürünün daha kazançlı olduğu araştırması yapılabilir.

# Bu sebeple farklı sonuçlara ulaşmak amacıyla iki örneklem oran testleri ile araştırmamıza devam edebiliriz.

###################################### KARŞILAŞTIRMA İÇİN FARKLI METRİKLERE BAKIlMASI ##########################################

# Conversıon rate : dönüşüm oranı ( purchase / ımpressıon ) ( siteyi ziyaret edenlerin getirdiği satın alma oranı )
# H0 : M1 = M2 (average bıddıng ile maxımum bıddıng'ın siteyi ziyaret edenlerin getirdiği satın alma oranı arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1 : M1 != M2 (average bıddıngile maxımum bıddıng'ın siteyi ziyaret edenlerin getirdiği satın alma oranı arasında istatistiksel olarak anlamlı bir fark vardır.)


control_prop = df_control["Purchase"].sum()/df_control["Impression"].sum() * 100   ### Kontrol grubunda gösterimlerin yaklaşık %0.54’ü satın almaya dönüşmüş.
test_prop = df_test["Purchase"].sum()/df_test["Impression"].sum() * 100            ### Test grubunda gösterimlerin yaklaşık %0.48’i satın almaya dönüşmüş.
                                                                                   ## ORANDA DÜŞÜŞ VAR

test_stat, pvalue = proportions_ztest(count=[df_control["Purchase"].sum(), df_test["Purchase"].sum()],
                                      nobs=[df_control["Impression"].sum(), df_test["Impression"].sum()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

### Test Stat = 12.2212, p-value = 0.0000
# P-value < alfa = 0.05 oldugu ıcın h0 red. %95 güvenle average bıddıng ile maxımum bıddıng'ın siteyi ziyaret edenlerin getirdiği satın alma oranı arasında istatistiksel olarak anlamlı bir fark vardır.
# Yeni özellik olan average bıddıng ile siteyi ziyaret edenlerin getirdiği satın alma oranının istatistiksel olarak azaldığını söyleyebiliriz.



## Clıck through rate : tıklanma oranı ( clıck / ımpressıon ) ( siteyi ziyaret edenlerin getirdiği tıklanma oranı )
## H0 : M1 = M2 (average bıddıng ile maxımum bıddıng'ın siteyi ziyaret edenlerin getirdiği tıklanma oranı arasında istatistiksel olarak anlamlı bir fark yoktur.)
## H1 : M1 != M2 (average bıddıngile maxımum bıddıng'ın siteyi ziyaret edenlerin getirdiği tıklanma oranı arasında istatistiksel olarak anlamlı bir fark vardır.)


control_prop = df_control["Click"].sum()/df_control["Impression"].sum() * 100      ### Kontrol grubunda gösterilen reklamların yaklaşık %5’i tıklamaya dönüşmüştür.
test_prop = df_test["Click"].sum()/df_test["Impression"].sum() * 100               ### Test grubunda reklamı görüntüleyenlerin %3,3'ü reklama tıklamış.
                                                                                   ## ORANDA DÜŞÜŞ VAR

test_stat, pvalue = proportions_ztest(count=[df_control["Click"].sum(), df_test["Click"].sum()],
                                      nobs=[df_control["Impression"].sum(), df_test["Impression"].sum()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 129.3305, p-value = 0.0000
# P-value < alfa = 0.05 oldugu ıcın h0 red. %95 güvenle average bıddıng ile maxımum bıddıng'ın siteyi ziyaret edenlerin getirdiği tıklanma oranı arasında istatistiksel olarak anlamlı bir fark vardır.
# yeni özellik olan average bıddıng ile siteyi ziyaret edenlerin getirdiği tıklanma oranının istatistiksel olarak azaldığını söyleyebiliriz.


# Tıklanma başına satın alma oranı ( purchase / clıck ) ( tıklanmaların getirdiği satın alma oranı )
# H0 : M1 = M2 (average bıddıng ile maxımum bıddıng'ın tıklanmaların getirdiği satın alma oranı arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1 : M1 != M2 (average bıddıngile maxımum bıddıng'ın tıklanmaların getirdiği satın alma oranı arasında istatistiksel olarak anlamlı bir fark vardır.)


control_prop = df_control["Purchase"].sum()/df_control["Click"].sum() * 100    ### %10
test_prop = df_test["Purchase"].sum()/df_test["Click"].sum() * 100             ### %14      ## ORANDA ARTIŞ VAR

test_stat, pvalue = proportions_ztest(count=[df_control["Purchase"].sum(), df_test["Purchase"].sum()],
                                      nobs=[df_control["Click"].sum(), df_test["Click"].sum()])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = -34.9800, p-value = 0.0000
# P-value < alfa = 0.05 oldugu ıcın h0 red. %95 güvenle average bıddıng ile maxımum bıddıng'ın siteyi ziyaret edenlerin getirdiği satın alma oranı arasında istatistiksel
# olarak anlamlı bir fark vardır.
# yeni özellik olan average bıddıng ile tıklanmaların getirdiği satın alma oranının istatistiksel olarak arttığını söyleyebiliriz.


### Yukarıdaki metrik araştırmaları yapıldıktan sonra elde edilen sonuçlara göre;
# siteyi ziyaret edenlerin getirdiği satın alma oranının yeni özellik olan average bıddıng ile düştüğü,
# siteyi ziyaret edenlerin getirdiği tıklanma oranının yeni özellik olan average bıddıng ile düştüğü,
# ancak yeni özellik average bıddıng ile tıklanmaların getirdiği satın alma oranının arttığı 
# istatistiksel olarak %95 güvenle gözlenmiştir.

### Görev-4 de yer alan açıklamalar geçerli olmakla birlikte tıklanma ile satın alma ilişkisi de yakından incelenebilir.
