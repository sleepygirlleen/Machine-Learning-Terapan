# Laporan Proyek Machine Learning - Sulistiani

## Domain Proyek

### Latar Belakang

Prestasi siswa dalam bidang akademik dapat dipengaruhi oleh berbagai faktor. Berdasarkan penelitian oleh Reuter dan Forster (2021), beberapa faktor yang dapat mempengaruhi diantaranya adalah durasi tidur, waktu belajar, serta tingkat aktivitas fisik. Gaya hidup yang tidak sehat berpengaruh dalam menurunkan kemampuan kognitif dan konsentrasi siswa dalam belajar.

Meskipun sebagian siswa sudah menyadari betapa pentingnya hidup sehat, akan tetapi tidak sedikit dari mereka belum menyadari sejauh mana faktor-faktor tersebut dapat mempengaruhi proses belajar. Ketidaksadaran tersebut dapat menghambat upaya mereka dalam mencapau hasil belajar yang optimal.

Selain itu, faktor kesehatan mental dapat mempengaruhi performa siswa dalam bidang akademik. Penelitian oleh Brandy, et. al (2015) menunjukkan bahwa siswa tahun pertama cenderung berisiko tinggi mengalami gangguan kesehatan mental. Penelitian lainnya yang dilakukan oleh Wyatt, et., al (2017) menemukan bahwa semakin tinggi jenjang pendidikan siswa, semakin besar pula potensi untuk melakukan percobaan bunuh diri dan depresi.

Untuk mengatasi permasalahan tersebut, penelitian ini akan  menganalisis dan memprediksi hubungan antara kualitas tidur, konsumsi makanan, aktivitas fisik, serta kesehatan mental terhadap prestasi akademik siswa. Diharapkan hasil penelitian ini dapat memberikan kontribusi dalam meningkatkan kualitas pendidikan melalui penerapan gaya hidup sehat dan memperhatikan kesehatan mental siswa. 

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana hubungan antara faktor seperti gaya hidup dan kesehatan mental terhadap prestasi siswa secara kesseluruhan?
- Bagaimana penggunaan algoritma machine learning dalam memprediksi hubungan antara gaya hidup dan kesehatan mental terhadap prestasi siswa?

### Goals
- Mengidentifikasi hubungan antara gaya hidup dan kesehatan mental terhadap prestasi siswa
- Membangun model machine learning untuk memprediksi hubungan antara gaya hidup dan kesehatan mental terhadap prestasi siswa

### Solution statements
- Menggunakan algoritma model regresi, yaitu Random Forest, Decision Tree, dan K-Nearest Neighbors dengan R² ≥  0.80
- Melakukan teknik hyperparameter tuning untuk meningkatkan performa model dengan R² >  0.80

## Data Understanding
Dataset yang digunakan merupakan kumpulan data simulasi yang mengeksplorasi hubungan antara kebiasaan gaya hidup dan kinerja akademik siswa. Terdapat sejumlah 1000 data dan 16 fitur lebih di dalamnya. Dataset ini sangat cocok untuk proyek ML, analisis regresi, pengelompokan, dan data yang dibuat dengan menggunakan pola untuk praktik pendidikan. Berikut di bawah ini merupakan link dataset yang digunakan untuk penelitian: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance/data

**Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:**
- student_id: ID dari siswa.
- age:  Usia siswa dalam tahun.
- gender: Jenis kelamin dari siswa.
- study_hours_per_day: Jumlah rata-rata jam siswa belajar per hari.
- social_media_hours: Jumlah rata-rata jam yang dihabiskan siswa menggunakan media sosial per hari.
- netflix_hours: Jumlah rata-rata jam yang dihabiskan siswa ketika menonton Netflix per hari.
- part_time_job: Apakah siswa memiliki pekerjaan paruh waktu (Yes atau No).
- attendance_percentage: Persentase kehadiran siswa di kelas.
- sleep_hours: Jumlah rata-rata jam tidur siswa per hari.
- diet_quality: Kualitas pola makan siswa (misalnya, Poor, Fair, Good).
- exercise_frequency: Frekuensi olahraga per minggu (dalam jumlah sesi).
- parental_education_level: Tingkat pendidikan tertinggi orang tua (misalnya, High School, Master).
- internet_quality: Kualitas koneksi internet di rumah (misalnya, Poor, Average, Good).
- mental_health_rating: Penilaian kesehatan mental siswa pada skala tertentu.
- extracurricular_participation: Apakah siswa berpartisipasi dalam kegiatan ekstrakurikuler (Yes atau No).
- exam_score: Skor ujian akhir siswa.

**Kondisi Data:**

![image](https://github.com/user-attachments/assets/eabcffb6-7423-4248-8073-100f7f6524bf)

Dalam kolom parental_education_level terdapat 91 baris data yang hilang
![image](https://github.com/user-attachments/assets/093e4b43-ec03-4f91-8c99-ad14cca5030e)

Tidak terdapat baris yang duplikat di dalam dataset
![image](https://github.com/user-attachments/assets/001a52e1-d754-4a17-9f70-feff792a7c95)

### Exploratory Data Analysis
#### Mental Health Rating

![image](https://github.com/user-attachments/assets/ef905ea9-4e30-4499-b980-ec942d60756c)

Terlihat tren positif yang jelas antara peningkatan mental health rating dengan peningkatan nilai ujian. Semakin tinggi rating kesehatan mental (dari 1 hingga 10), semakin tinggi pula nilai ujian (dari ~62.76 hingga ~77.73). Hal ini menunjukkan bahwa kesehatan mental yang baik mungkin berkontribusi terhadap performa akademik yang lebih baik.
#### Durasi Tidur

![image](https://github.com/user-attachments/assets/c9aab104-0633-4515-b7c9-0d0bb2b9b9d1)

Nilai ujian tertinggi ditemukan pada kelompok dengan durasi tidur 7-8 jam (~71.41), diikuti oleh kelompok 9+ jam (~69.86) dan 5-6 jam (~68.76). Kelompok dengan tidur kurang dari 5 jam memiliki nilai terendah (~63.45). Pola ini menunjukkan bahwa tidur yang cukup (7-8 jam) berhubungan dengan performa akademik optimal, sementara tidur terlalu sedikit atau terlalu banyak mungkin kurang ideal.
#### Waktu Belajar

![image](https://github.com/user-attachments/assets/29e727b7-d244-47a1-90a5-c6591ea30437)

Terdapat korelasi positif yang kuat antara peningkatan waktu belajar dan nilai ujian. Kelompok yang belajar 5-8 jam memiliki nilai tertinggi (~90.37), sementara yang belajar kurang dari 1 jam memiliki nilai terendah (~40.81-41.03). Namun, data untuk kelompok belajar lebih dari 8 jam tidak tersedia (Null), sehingga tidak dapat dianalisis lebih lanjut.
#### Frekuensi Olahraga

![image](https://github.com/user-attachments/assets/e5f13170-d403-4bdd-af49-2d08d2023bff)

Meskipun tidak sekuat faktor lainnya, terdapat sedikit tren positif antara frekuensi olahraga dan nilai ujian. Nilai tertinggi ditemukan pada kelompok yang berolahraga 6 kali seminggu (~74.40), sementara yang tidak berolahraga (0 kali) memiliki nilai lebih rendah (~66.38). Namun, pola ini tidak sepenuhnya konsisten, menunjukkan bahwa olahraga mungkin bukan faktor dominan.

## Data Preparation
### Tahapan:
1. Data Cleaning: Beberapa fitur yang kurang relevan atau tidak berkontribusi secara signifikan terhadap fokus analisis, seperti student_id, social_media_hours, netflix_hours, part_time_job, parental_education_level, extracurricular_participation, internet_quality, dan diet_quality, telah dihapus dari dataset untuk menyederhanakan pemodelan dan mengurangi potensi noise. Selain itu, metode Interquartile Range (IQR) digunakan untuk mengidentifikasi dan menghapus outlier pada fitur numerik, guna memastikan kualitas data yang lebih konsisten dan menghindari distorsi dalam hasil analisis.
2. Data Transformation: Dilakukan binning terhadap atribut study_hours_per_day menjadi beberapa kategori diskrit yang lebih mudah dianalisis. Berikut enam kategori berbeda yang terdapat dalam binning:
  - 0: Siswa yang tidak belajar sama sekali (0 jam). Batas bawah -0.1 dipilih untuk memastikan nilai 0 termasuk dalam kategori ini.
  - <=1: Siswa yang belajar kurang dari atau sama dengan 1 jam per hari (interval 0–1 jam).
  - 1–3: Siswa yang belajar antara 1 hingga 3 jam per hari.
  - 3–5: Siswa yang belajar antara 3 hingga 5 jam per hari.
  - 5–8: Siswa yang belajar antara 5 hingga 8 jam per hari.
  - Lebih dari 8: Siswa yang belajar lebih dari 8 jam per hari (interval 8–12 jam).
3. Data Splitting:  Dataset dibagi menjadi dua bagian, yaitu data latih (training set) dan data uji (test set), dengan proporsi 80:20 menggunakan fungsi train_test_split. Fitur independen yang digunakan meliputi mental_health_rating, dan sleep_hours, sedangkan variabel dependen adalah exam_score. Pemisahan ini bertujuan untuk melatih model pada data latih dan menguji performanya secara objektif pada data yang belum pernah dilihat sebelumnya. Nilai random_state=42 digunakan untuk memastikan reprodusibilitas hasil.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

### Random Forest Regressor
#### Deskripsi:
Algoritma Random Forest bekerja dengan membangun beberapa pohon keputusan (Decision Trees) selama pelatihan dan menggabungkan hasilnya untuk meningkatkan akurasi serta mengurangi terjadinya overfitting (Namira et al., 2024). Cara kerja dari algoritma Random Forest, yaitu dengan membangun beberapa pohon keputusan menggunakan sampel acak dari dataset. Keunggulan utama dari algoritma ini adalah kemampuannya dalam menangani dataset dengan banyak variabel, mengurangi overfitting, serta memiliki performa yang baik dalam berbagai jenis dataset, termasuk yang memiliki data yang tidak seimbang (Azmi & Voutama, 2023). Kekurangan dari algoritma ini diantaranya adalah kompleksitas komputasi, penggunaan memori, memerlukan waktu prediksi yang cukup lama, kurangnya interpretabilitas, dan mudah mengalami overfitting (GeeksforGeeks, 2024)
#### Parameter:
- n_estimators=200: Jumlah pohon keputusan (decision trees) dalam hutan (forest). Model akan membangun 200 decision tree.
- max_depth=10: Batas maksimum kedalaman setiap decision tree. Setiap pohon maksimal hanya boleh memiliki kedalaman 10 node.
- random_state=42: Dengan random_state=42, hasil model akan konsisten setiap kali dijalankan.
### Decision Tree
#### Deskripsi:
Decision Tree menggunakan sampel data untuk membangun pohon keputusan sebagai teknik machine learning. Strategi segmentasi berbasis tepi digunakan untuk membangun model pohon keputusan (Guliyev, 2021). Algoritma Decision Tree memiliki berbagai kelebihan, yaitu mudah dipahami, dapat menangani data numerik maupun kategorikal, tidak diperlukan scaling data, serta mampu memodelkan hubungan non-linear tanpa asumsi linearitas.  Akan tetapi, algoritma ini memiliki beberapa kekurangan, seperti kecenderungan overfitting, ketidakstabilan terhadap perubahan kecil dalam data, dan bias terhadap fitur dengan banyak kategori (GeeksforGeeks, 2023).
#### Parameter:
- max_depth=10: Batas maksimum kedalaman setiap decision tree. Setiap pohon maksimal hanya boleh memiliki kedalaman 10 node.
- random_state=42: Dengan random_state=42, hasil model akan konsisten setiap kali dijalankan.
### Linear Regression
#### Deskripsi:
Regresi Linier adalah algoritma pembelajaran mesin yang didasarkan pada pembelajaran yang diawasi. Algoritma ini melakukan tugas regresi. Model regresi adalah nilai prediksi target berdasarkan variabel independen. Ini sebagian besar digunakan untuk mengetahui hubungan antara variabel dan peramalan. Algoritma ini memiliki kelebihan, yaitu implementasinya mudah dan lebih mudah untuk menginterpretasikan koefisien input. Linier Regression rentan terhadap over-fitting, namun hal ini dapat dihindari dengan menggunakan beberapa teknik reduksi dimensi, teknik regularisasi (L1 dan L2), dan validasi silang (GeeksforGeeks, 2023).
#### Parameter:
- n_jobs=-1: Menentukan jumlah job (proses paralel) yang digunakan untuk komputasi. Nilai default-nya adalah None, yang berarti menggunakan semua core prosesor yang tersedia.
- positive=True: Menentukan apakah koefisien harus dibatasi menjadi positif. Nilai default-nya adalah False.

## Evaluation
### Metrik Evaluasi:
Berikut merupakan ukuran metrik evaluasi yang digunakan (Navlani & Idris, 2021):
- MAE: Mean Absolute Error (MAE) adalah metrik yang mengukur rata-rata antara nilai asli dan nilai prediksi. MAE tidak mengkuadratkan selisih, sehingga lebih robust terhadap outlier. MAE menghitung selisih absolut antara nilai aktual dan prediksi, lalu dirata-ratakan. Nilai MAE yang lebih kecil menunjukkan model yang lebih akurat.
  
![image](https://github.com/user-attachments/assets/1bca2376-11f8-4306-8afe-8be1e2eabc63)

- MSE: Mean Squared Error (MSE) adalah metrik yang mengukur rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi. Selisih tersebut kemudian dikuadratkan, dijumlahkan dan diambil rata-rata dari semua sampel data. Semakin kecil nilai MSE maka akan semakin baik model dalam melakukan prediksi yang akurat.

![image](https://github.com/user-attachments/assets/310bb80a-f22d-4e0f-a9b5-e8b72f1500b0)

- RMSE: Root Mean Square Error (RMSE) merupakan akar kuadrat dari MSE. RMSE sering digunakan karena memiliki satuan yang sama dengan data asli, sehingga lebih mudah diinterpretasikan. RMSE mengkuadratkan selisih untuk memperbesar pengaruh error besar, menghitung rata-rata, lalu diakarkan. RMSE sering digunakan karena memiliki satuan yang sama dengan data asli.

![image](https://github.com/user-attachments/assets/eb06711a-c807-4229-b135-4a6ad98d5a8c)

- R² Score: R² Score atau yang biasa dikenal sebagai koefisien determinasi adalah model evaluasi statistik yang menilai model regresi. Ukuran prediksi ini membantu analis data untuk menjelaskan performa dari model kinerja.R² membandingkan error model dengan error rata-rata. Jika model sama buruknya dengan rata-rata, maka R² = 0; jika model sempurna, maka R² = 1; dan jika lebih buruk dari rata-rata, maka R² < 0.

![image](https://github.com/user-attachments/assets/64151105-07bd-4d62-b2d4-bb662921cf51)

### Hasil Evaluasi:
| Model                   | MAE    | MSE     | RMSE   | R²     |
|-------------------------|--------|---------|--------|--------|
| Random Forest Regressor | 5.7291 | 49.7739 | 7.0551 | 0.8076 |
| Decision Tree           | 7.4388 | 89.9373 | 9.4835 | 0.6524 |
| Linear Regression       | 5.3687 | 43.6309 | 6.6054 | 0.8314 |

### Insight:
- Linear Regression mencatat performa terbaik dengan R² 0.8314, mengindikasikan kemampuannya menjelaskan 83.14% variasi data target. Nilai MAE 5.3687 dan RMSE 6.6054 menunjukkan konsistensi prediksi yang tinggi, dengan 68% hasil prediksi diperkirakan berada dalam rentang ±6.6 unit dari nilai aktual. Pencapaian ini menguatkan asumsi bahwa hubungan antara fitur dan target bersifat linear pada dataset ini. 
- Random Forest Regressor menyusul di posisi kedua dengan R² 0.8076, menawarkan fleksibilitas dalam menangani pola non-linear, meski memerlukan sumber daya komputasi lebih besar.
- Decision Tree dengan R² 0.6524 memperlihatkan keterbatasan signifikan dalam akurasi, terutama disebabkan oleh kecenderungan underfitting dan sensitivitas terhadap noise data.

### Rekomendasi
- Linear Regression direkomendasikan sebagai solusi utama untuk implementasi produksi, khususnya jika interpretasi model menjadi pertimbangan penting. Optimasi tambahan dengan transformasi Box-Cox pada variabel target dapat meningkatkan stabilitas prediksi terhadap outlier.
- Untuk skenario dengan kompleksitas data tinggi, Random Forest yang telah di-tuning hyperparameter-nya (n_estimators=200, max_depth=15) mampu menjadi alternatif andal dengan potensi peningkatan akurasi 3-5%.
- Decision Tree sebaiknya dikembangkan dalam framework ensemble seperti Gradient Boosting untuk mengatasi kelemahan akurasinya. Pemantauan berkala terhadap rasio RMSE/MAE diperlukan untuk mendeteksi dini masalah outlier atau konsep drift pada data baru.

## Hyperparameter Tuning
### Parameter
**Random Forest**
-  'n_estimators': [100, 200] Membangun model dengan menggunakan 100 dan 200 pohon
-  'max_depth': [10, 15, None] Membatasi maksimum kedalaman pohon menjadi 10, 15, None (Tidak dibatasi)
-  'min_samples_leaf': [1, 3] Minimum jumlah data di pohon
**Decision Tree**
- 'max_depth': [5, 8, 10]  Membatasi maksimum kedalaman pohon menjadi 5, 8, dan 10
- 'min_samples_split': [5, 10] Minimum sampel yang dibutuhkan untuk membagi simpul menjadi dua cabang
- 'ccp_alpha': [0, 0.01] Memangkas cabang pohon yang tidak memberi peningkatan signifikan pada performa
**Linear Regression**
-  'fit_intercept': [True, False] Menentukan  perhitungan intersep (bias) atau tidak
-  'positive': [True, False] Menentukan koefisien regresi hanya positive saja
  
### Hasil Evaluasi Hyperparameter Tuning:
| Model              | MAE     | MSE     | RMSE    | R2      |
|--------------------|---------|---------|---------|---------|
| Linear Regression  | 5.3015  | 44.7317 | 6.6797  | 0.8371  |
| Random Forest      | 5.7445  | 51.7849 | 7.1772  | 0.8124  |
| Decision Tree      | 6.8242  | 71.7488 | 8.4535  | 0.7400  |

### Kesimpulan:
- Linear Regression menunjukkan performa stabil dengan tuning, namun perbaikan metrik sangat kecil (misalnya R² naik dari 0.8314 menjadi 0.8371). Hal ini mengindikasikan bahwa model ini sudah cukup optimal sejak awal.
- Random Forest menunjukkan peningkatan kecil pada MAE, namun MSE dan RMSE justru sedikit naik. Hal ini menandakan bahwa tuning memberikan efek terbatas pada peningkatan performa, namun menjaga konsistensi prediksi.
- Decision Tree mengalami peningkatan signifikan setelah tuning. R² meningkat dari 0.6433 ke 0.7400, dan error (MSE dan RMSE) berkurang cukup besar, menunjukkan bahwa pemilihan parameter seperti ccp_alpha dan max_depth efektif mengurangi overfitting.
- Secara keseluruhan, tuning efektif untuk model yang kompleks seperti Decision Tree, tetapi kurang berdampak pada model sederhana seperti Linear Regression. Random Forest menunjukkan hasil yang stabil, meskipun tidak ada peningkatan besar. Pemilihan teknik tuning dan parameter yang tepat sangat berpengaruh pada hasil akhir, terutama untuk model yang rentan terhadap overfitting seperti Decision Tree.
- Dapat disimpulkan bahwa beberapa faktor mempengaruhi terhadap tingkat prestasi siswa. Terutama waktu belajar yang menjadi faktor paling utama, di mana siswa yang belajar sekitar 5-8 jam per hari memiliki nilai ujian tertinggi sekitar 90,37. Selain itu, kesehatan mental juga menunjukkan pengaruh yang cukup signifikan, dengan peningkatan rating kesehatan mental dari 1 hingga 10 berbanding lurus dengan kenaikan nilai ujian dari 62,76 menjadi 77,73. Faktor pendukung lain seperti durasi tidur menunjukkan pola optimal pada 7-8 jam tidur dengan nilai rata-rata 71,41, sementara tidur kurang dari 5 jam menghasilkan nilai terendah (63,45). Frekuensi olahraga, meskipun menunjukkan tren positif dengan nilai tertinggi pada 6 kali olahraga per minggu (74,40), tidak memberikan pengaruh sekuat faktor lainnya. Mengisyaratkan bahwa olahraga atau pun aktivitas fisik tidak berpengaruh secara langsung terhadap performa akademik.

# Referensi:
- Brandy, J. M., Penckofer, S., Solari-Twadell, P. A., & Velsor-Friedrich, B. (2015). Factors predictive of depression in first-year college students. Journal of Psychosocial Nursing and Mental Health Services, 53(2), 38-44.
- Reuter, P. R., & Forster, B. L. (2021). Student health behavior and academic performance. PeerJ, 9, e11107.
- Wyatt, T. J., Oswalt, S. B., & Ochoa, Y. (2017). Mental Health and Academic Performance of First-Year College Students. International Journal of Higher Education, 6(3), 178-187.
- Azmi, A., Azel, F., & Voutama, A. (2023). Prediksi Churn Nasabah Bank Menggunakan Klasifikasi Random Forest dan Decision Tree dengan Evaluasi Confusion Matrix. Jurnal Ilmiah Komputer dan Informatika (KOMPUTA), Universitas Singaperbangsa Karawang. https://doi.org/10.37676/komputa.v11i3.12639
- Guliyev, H., & Yerdelen Tatoğlu, F. (2021). Customer churn analysis in banking sector: Evidence from explainable machine learning model. Journal of Applied Microeconometrics, 1(2).
- Namira, N., Slamet, I., & Susanto, I. (2024). Prediksi Nasabah Churn dengan Algoritma Decision Tree, Random Forest, dan Support Vector Machine. Proceedings of ESCAF 3rd 2024, Universitas Bina Insan Lubuklinggau.
- GeeksforGeeks. (2024). What are the advantages and disadvantages of Random Forest? https://www.geeksforgeeks.org/what-are-the-advantages-and-disadvantages-of-random-forest/
- GeeksforGeeks. (2023). Decision Tree. https://www.geeksforgeeks.org/decision-tree/
- GeeksforGeeks. (2023, March 30). ML - advantages and disadvantages of linear regression. https://www.geeksforgeeks.org/ml-advantages-and-disadvantages-of-linear-regression/ 

