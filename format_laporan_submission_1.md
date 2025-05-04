# Laporan Proyek Machine Learning - Sulistiani

## Domain Proyek

Prestasi siswa dalam bidang akademik dipengaruhi oleh berbagai faktor. Berdasarkan penelitian oleh Reuter dan Forster (2021), beberapa faktor yang dapat mempengaruhi diantaranya adalah durasi tidur, jenis makanan yang dikonsumsi, tingkat aktivitas fisik, serta kebiasaan mengonsumsi zat tertentu seperti alkohol, ganja, dan rokok elektrik. Gaya hidup yang tidak sehat memiliki potensi untuk menurunkan kemampuan kognitif dan konsentrasi siswa dalam belajar.

Meskipun sebagian siswa sudah menyadari betapa pentingnya hidup sehat, akan tetapi tidak sedikit dari mereka yang belum menyadari sejauh mana faktor-faktor tersebut dapat mempengaruhi proses belajar. Ketidaksadaran tersebut dapat menghambat upaya mereka dalam mencapau hasil belajar yang optimal.

Selain itu, faktor kesehatan mental dapat mempengaruhi performa siswa dalam bidang akademik. Penelitian oleh Brandy, et. al (2015) menunjukkan bahwa siswa tahun pertama cenderung berisiko tinggi mengalami gangguan kesehatan mental. Penelitian lainnya yang dilakukan oleh Wyatt, et., al (2017) menemukan bahwa semakin tinggi jenjang pendidikan siswa, semakin besar pula potensi untuk melakukan percobaan bunuh diri dan depresi.

Untuk mengatasi permasalahan tersebut, penelitian ini akan  menganalisis dan memprediksi hubungan antara kualitas tidur, konsumsi makanan, aktivitas fisik, serta kesehatan mental terhadap prestasi akademik siswa. Diharapkan hasil penelitian ini dapat memberikan kontribusi dalam meningkatkan kualitas pendidikan melalui penerapan gaya hidup sehat dan memperhatikan kesehatan mental siswa. 

Referensi:
Brandy, J. M., Penckofer, S., Solari-Twadell, P. A., & Velsor-Friedrich, B. (2015). Factors predictive of depression
in first-year college students. Journal of Psychosocial Nursing and Mental Health Services, 53(2), 38-44.
Reuter, P. R., & Forster, B. L. (2021). Student health behavior and academic performance. PeerJ, 9, e11107.
Wyatt, T. J., Oswalt, S. B., & Ochoa, Y. (2017). Mental Health and Academic Performance of First-Year College Students. International Journal of Higher Education, 6(3), 178-187.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Apakah gaya hidup memiliki pengaruh yang cukup besar terhadap performa akademik siswa?
- Bagaimana kesehatan mental dalam mempengaruhi performa akademik siswa?
- Bagaimana hubungan antara faktor seperti gaya hidup dan kesehatan mental terhadap prestasi siswa secara kesseluruhan?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

