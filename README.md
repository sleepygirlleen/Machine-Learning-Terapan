# Laporan Proyek Machine Learning - Sulistiani

## Domain Proyek

### Latar Belakang

Prestasi siswa dalam bidang akademik dapat dipengaruhi oleh berbagai faktor. Berdasarkan penelitian oleh Reuter dan Forster (2021), beberapa faktor yang dapat mempengaruhi diantaranya adalah durasi tidur, waktu belajar, serta tingkat aktivitas fisik. Gaya hidup yang tidak sehat berpengaruh dalam menurunkan kemampuan kognitif dan konsentrasi siswa dalam belajar.

Meskipun sebagian siswa sudah menyadari betapa pentingnya hidup sehat, akan tetapi tidak sedikit dari mereka belum menyadari sejauh mana faktor-faktor tersebut dapat mempengaruhi proses belajar. Ketidaksadaran tersebut dapat menghambat upaya mereka dalam mencapau hasil belajar yang optimal.

Selain itu, faktor kesehatan mental dapat mempengaruhi performa siswa dalam bidang akademik. Penelitian oleh Brandy, et. al (2015) menunjukkan bahwa siswa tahun pertama cenderung berisiko tinggi mengalami gangguan kesehatan mental. Penelitian lainnya yang dilakukan oleh Wyatt, et., al (2017) menemukan bahwa semakin tinggi jenjang pendidikan siswa, semakin besar pula potensi untuk melakukan percobaan bunuh diri dan depresi.

Untuk mengatasi permasalahan tersebut, penelitian ini akan  menganalisis dan memprediksi hubungan antara kualitas tidur, konsumsi makanan, aktivitas fisik, serta kesehatan mental terhadap prestasi akademik siswa. Diharapkan hasil penelitian ini dapat memberikan kontribusi dalam meningkatkan kualitas pendidikan melalui penerapan gaya hidup sehat dan memperhatikan kesehatan mental siswa. 

Referensi:
Brandy, J. M., Penckofer, S., Solari-Twadell, P. A., & Velsor-Friedrich, B. (2015). Factors predictive of depression
in first-year college students. Journal of Psychosocial Nursing and Mental Health Services, 53(2), 38-44.
Reuter, P. R., & Forster, B. L. (2021). Student health behavior and academic performance. PeerJ, 9, e11107.
Wyatt, T. J., Oswalt, S. B., & Ochoa, Y. (2017). Mental Health and Academic Performance of First-Year College Students. International Journal of Higher Education, 6(3), 178-187.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana hubungan antara faktor seperti gaya hidup dan kesehatan mental terhadap prestasi siswa secara kesseluruhan?
- Bagaimana penggunaan algoritma machine learning dalam memprediksi hubungan antara gaya hidup dan kesehatan mental terhadap prestasi siswa?

### Goals
- Mengidentifikasi hubungan antara gaya hidup dan kesehatan mental terhadap prestasi siswa
- Membangun model machine learning untuk memprediksi hubungan antara gaya hidup dan kesehatan mental terhadap prestasi siswa

### Solution statements
- Menggunakan algoritma model regresi, yaitu Random Forest, Decision Tree, dan K-Nearest Neighbors
- Melakukan teknik hyperparameter tuning untuk meningkatkan performa model

## Data Understanding
Dataset yang digunakan merupakan kumpulan data simulasi yang mengeksplorasi hubungan antara kebiasaan gaya hidup dan kinerja akademik siswa. Terdapat sejumlah 1000 data dan 15 fitur lebih di dalamnya. Dataset ini sangat cocok untuk proyek ML, analisis regresi, pengelompokan, dan data yang dibuat dengan menggunakan pola untuk praktik pendidikan. Berikut di bawah ini merupakan link dataset yang digunakan untuk penelitian: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance/data

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
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

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
### Tahapan:
1. Data Cleaning: Beberapa fitur yang kurang relevan atau tidak berkontribusi secara signifikan terhadap fokus analisis, seperti student_id, social_media_hours, netflix_hours, part_time_job, parental_education_level, extracurricular_participation, internet_quality, dan diet_quality, telah dihapus dari dataset untuk menyederhanakan pemodelan dan mengurangi potensi noise. Selain itu, metode Interquartile Range (IQR) digunakan untuk mengidentifikasi dan menghapus outlier pada fitur numerik, guna memastikan kualitas data yang lebih konsisten dan menghindari distorsi dalam hasil analisis.
2. Data Splitting:  Dataset dibagi menjadi dua bagian, yaitu data latih (training set) dan data uji (test set), dengan proporsi 80:20 menggunakan fungsi train_test_split. Fitur independen yang digunakan meliputi mental_health_rating, sleep_hours, dan diet_quality_encoded, sedangkan variabel dependen adalah exam_score. Pemisahan ini bertujuan untuk melatih model pada data latih dan menguji performanya secara objektif pada data yang belum pernah dilihat sebelumnya. Nilai random_state=42 digunakan untuk memastikan reprodusibilitas hasil.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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

Referensi:
- Azmi, A., Azel, F., & Voutama, A. (2023). Prediksi Churn Nasabah Bank Menggunakan Klasifikasi Random Forest dan Decision Tree dengan Evaluasi Confusion Matrix. Jurnal Ilmiah Komputer dan Informatika (KOMPUTA), Universitas Singaperbangsa Karawang. https://doi.org/10.37676/komputa.v11i3.12639
- Guliyev, H., & Yerdelen Tatoğlu, F. (2021). Customer churn analysis in banking sector: Evidence from explainable machine learning model. Journal of Applied Microeconometrics, 1(2).
- Namira, N., Slamet, I., & Susanto, I. (2024). Prediksi Nasabah Churn dengan Algoritma Decision Tree, Random Forest, dan Support Vector Machine. Proceedings of ESCAF 3rd 2024, Universitas Bina Insan Lubuklinggau.
- GeeksforGeeks. (2024). What are the advantages and disadvantages of Random Forest? https://www.geeksforgeeks.org/what-are-the-advantages-and-disadvantages-of-random-forest/
- GeeksforGeeks. (2023). Decision Tree. https://www.geeksforgeeks.org/decision-tree/
- GeeksforGeeks. (2023, March 30). ML - advantages and disadvantages of linear regression. https://www.geeksforgeeks.org/ml-advantages-and-disadvantages-of-linear-regression/ 


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
### Metrik Evaluasi:
Berikut merupakan ukuran metrik evaluasi yang digunakan (Navlani & Idris, 2021):
- MAE: Mean Absolute Error (MAE) adalah metrik yang mengukur rata-rata antara nilai asli dan nilai prediksi. MAE tidak mengkuadratkan selisih, sehingga lebih robust terhadap outlier. MAE menghitung selisih absolut antara nilai aktual dan prediksi, lalu dirata-ratakan. Nilai MAE yang lebih kecil menunjukkan model yang lebih akurat.
  
![image](https://github.com/user-attachments/assets/1bca2376-11f8-4306-8afe-8be1e2eabc63)

- RMSE: Root Mean Square Error (RMSE) merupakan akar kuadrat dari MSE. RMSE sering digunakan karena memiliki satuan yang sama dengan data asli, sehingga lebih mudah diinterpretasikan. RMSE mengkuadratkan selisih untuk memperbesar pengaruh error besar, menghitung rata-rata, lalu diakarkan. RMSE sering digunakan karena memiliki satuan yang sama dengan data asli.

![image](https://github.com/user-attachments/assets/eb06711a-c807-4229-b135-4a6ad98d5a8c)

- R² Score: R² Score atau yang biasa dikenal sebagai koefisien determinasi adalah model evaluasi statistik yang menilai model regresi. Ukuran prediksi ini membantu analis data untuk menjelaskan performa dari model kinerja.R² membandingkan error model dengan error rata-rata. Jika model sama buruknya dengan rata-rata, maka R² = 0; jika model sempurna, maka R² = 1; dan jika lebih buruk dari rata-rata, maka R² < 0.

![image](https://github.com/user-attachments/assets/64151105-07bd-4d62-b2d4-bb662921cf51)

### Hasil Evaluasi:
| Model                   | MAE    | MSE     | RMSE   | R²     |
|-------------------------|--------|---------|--------|--------|
| Random Forest Regressor | 5.7728 | 50.8986 | 7.1301 | 0.8035 |
| Decision Tree           | 7.4712 | 92.2860 | 9.6066 | 0.6433 |
| Linear Regression       | 5.3687 | 43.6309 | 6.6054 | 0.8314 |

### Hasil Evaluasi Hyperparameter Tuning:
                      MAE      MSE    RMSE      R2
Linear Regression  5.3015  44.7317  6.6797  0.8371
Random Forest      5.7445  51.7849  7.1772  0.8124
Decision Tree      6.8242  71.7488  8.4535  0.7400

### Insight:
- Linear Regression mencatat performa terbaik dengan R² 0.8314, mengindikasikan kemampuannya menjelaskan 83.14% variasi data target. Nilai MAE 5.3687 dan RMSE 6.6054 menunjukkan konsistensi prediksi yang tinggi, dengan 68% hasil prediksi diperkirakan berada dalam rentang ±6.6 unit dari nilai aktual. Pencapaian ini menguatkan asumsi bahwa hubungan antara fitur dan target bersifat linear pada dataset ini. 
- Random Forest menyusul di posisi kedua dengan R² 0.8074, menawarkan fleksibilitas dalam menangani pola non-linear, meski memerlukan sumber daya komputasi lebih besar.
- Decision Tree dengan R² 0.6546 memperlihatkan keterbatasan signifikan dalam akurasi, terutama disebabkan oleh kecenderungan underfitting dan sensitivitas terhadap noise data.

### Rekomendasi
- Linear Regression direkomendasikan sebagai solusi utama untuk implementasi produksi, khususnya jika interpretasi model menjadi pertimbangan penting. Optimasi tambahan dengan transformasi Box-Cox pada variabel target dapat meningkatkan stabilitas prediksi terhadap outlier.
- Untuk skenario dengan kompleksitas data tinggi, Random Forest yang telah di-tuning hyperparameter-nya (n_estimators=200, max_depth=15) mampu menjadi alternatif andal dengan potensi peningkatan akurasi 3-5%.
- Decision Tree sebaiknya dikembangkan dalam framework ensemble seperti Gradient Boosting untuk mengatasi kelemahan akurasinya. Pemantauan berkala terhadap rasio RMSE/MAE diperlukan untuk mendeteksi dini masalah outlier atau konsep drift pada data baru.
