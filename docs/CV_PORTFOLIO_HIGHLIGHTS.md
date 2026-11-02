# Portfolio & CV Highlights: IDX30 Portfolio Optimization using Adaptive Genetic Algorithm (AGA)

Dokumen ini menyediakan draf deskripsi proyek siap pakai untuk **Curriculum Vitae (CV)**, profil **LinkedIn**, dan portofolio profesional Anda, dirancang khusus untuk posisi **Data Scientist** dan **Machine Learning Engineer**.

Tersedia dalam dua bahasa: **English** (standar internasional) dan **Bahasa Indonesia**.

---

## English Versions (Recommended for Tech & International Roles)

### Project Title & Links
- **Title Options:**
  - *IDX30 Stock Portfolio Optimization using Adaptive Genetic Algorithm & Walk-Forward Validation*
  - *Dynamic Portfolio Optimization Engine: Adaptive Genetic Algorithm (AGA) with Walk-Forward Backtesting*
- **Live Demo:** [https://portfolio-optimization-aga.streamlit.app](https://portfolio-optimization-aga.streamlit.app)
- **GitHub Repository:** [https://github.com/miftahsuryan/idx30-portfolio-optimization-aga](https://github.com/miftahsuryan/idx30-portfolio-optimization-aga)

### Option A: Compact Format (3 High-Impact Bullet Points — Ideal for 1-Page Resume)
* **Developed an end-to-end portfolio optimization framework** using an **Adaptive Genetic Algorithm (AGA)** on Indonesian Blue-Chip stocks (IDX30), dynamically adjusting crossover ($P_c$) and mutation ($P_m$) rates to prevent premature convergence.
* **Engineered a 4-phase Walk-Forward Optimization (WFO) backtest** incorporating realistic market frictions (0.15% buy, 0.25% sell fees), outperforming the IDX30 benchmark with a **46.51% Total Return** (vs. 13.48%), a **2.1036 Sharpe Ratio** (vs. 0.3767), and a **-15.74% Max Drawdown**.
* **Statistically validated performance** across 80 independent runs using **Shapiro-Wilk normality tests**, **Paired t-tests**, and **Wilcoxon Signed-Rank tests**, demonstrating statistically significant objective fitness gains ($p < 0.001$) over Standard Genetic Algorithms across all 4 out-of-sample phases.

---

### Option B: Detailed Technical Format (5-6 Bullet Points — Ideal for ML/Data Science Master Resume)
* **Architected a regularized Mean-Variance optimization engine** utilizing an **Adaptive Genetic Algorithm (AGA)** with Herfindahl-Hirschman Index (HHI) diversification penalties ($\lambda_{\text{div}} = 0.05$) and strict asset allocation caps ($w_i \le 20\%$) via an iterative excess-redistribution repair mechanism.
* **Designed adaptive operator scaling ($k_1 \dots k_6$)** that tunes crossover and mutation rates based on population fitness variance, preventing local optimum stagnation and sustaining exploratory diversity across generations.
* **Built a Walk-Forward Optimization (WFO) backtesting pipeline** across 4 temporal out-of-sample phases, evaluating portfolio rebalancing, weight drift, and transaction friction (0.15% buy / 0.25% sell fees).
* **Achieved a 46.51% cumulative out-of-sample return** (50.61% annualized) and **2.1036 Sharpe Ratio**, outperforming Standard GA (33.90% return, 1.45 Sharpe), 1/N Equal Weight (38.51% return, 1.60 Sharpe), and the IDX30 Market Benchmark (13.48% return, 0.38 Sharpe).
* **Conducted rigorous statistical hypothesis testing** (Shapiro-Wilk, Paired t-test, Wilcoxon Signed-Rank, and Holm-Bonferroni correction) on 80 experimental runs, confirming superior multi-objective fitness convergence ($p < 0.001$) and robust out-of-sample generalization.
* **Deployed an institutional Quantitative Investment Terminal** using Streamlit and Plotly (Bloomberg/FactSet aesthetic), featuring 4-phase WFO equity trajectories, real-time live evolutionary simulation, and real-world Indonesia Stock Exchange (IDX) round-lot sizing calculations (`1 Lot = 100 shares`).

---

### Option C: Executive One-Liner (For Summary / Portfolio Header)
> Built an end-to-end financial portfolio optimization engine using Adaptive Genetic Algorithms (AGA) and Walk-Forward backtesting on IDX30 constituents, achieving an out-of-sample Sharpe Ratio of **2.1036** and **46.51% Total Return** with statistically verified alpha ($p < 0.001$).

---

## Versi Bahasa Indonesia (Untuk CV & Perusahaan Nasional)

### Judul & Tautan Proyek untuk CV
- **Pilihan Judul:** *Optimasi Portofolio Saham IDX30 Menggunakan Adaptive Genetic Algorithm (AGA) dan Walk-Forward Validation*
- **Live Demo:** [https://portfolio-optimization-aga.streamlit.app](https://portfolio-optimization-aga.streamlit.app)
- **GitHub Repository:** [https://github.com/miftahsuryan/idx30-portfolio-optimization-aga](https://github.com/miftahsuryan/idx30-portfolio-optimization-aga)

### Opsi A: Format Ringkas (3 Poin Utama — Cocok untuk Resume 1 Halaman)
* **Mengembangkan sistem optimasi portofolio investasi** berbasis **Adaptive Genetic Algorithm (AGA)** pada 22 saham konsisten indeks IDX30, mengadaptasi probabilitas *crossover* ($P_c$) dan mutasi ($P_m$) secara dinamis guna mengatasi konvergensi prematur.
* **Mengimplementasikan pengujian *Walk-Forward Optimization* (WFO) 4 fase** dengan memperhitungkan biaya transaksi riil (beli 0,15%, jual 0,25%) dan *rebalancing*, menghasilkan **Total Return 46,51%** (vs. Benchmark IDX30 13,48%), **Sharpe Ratio 2,1036** (vs. 0,3767), serta **Max Drawdown terkendali di -15,74%**.
* **Melakukan validasi statistik inferensial** terhadap 80 eksperimen independen menggunakan uji normalitas **Shapiro-Wilk**, **Paired t-test**, dan **Wilcoxon Signed-Rank**, membuktikan peningkatan performa fungsi objektif yang signifikan secara statistik ($p < 0,001$) dibanding Standard GA pada seluruh fase *out-of-sample*.

---

### Opsi B: Format Teknis Mendalam (5-6 Poin — Cocok untuk CV Komprehensif)
* **Merancang fungsi objektif Mean-Variance teregulasi** yang mengintegrasikan penalti diversifikasi *Herfindahl-Hirschman Index* (HHI, $\lambda_{\text{div}} = 0,05$) dan batas maksimum bobot per aset ($w_i \le 20\%$) menggunakan algoritma perbaikan (*repair mechanism*) redistribusi proporsional.
* **Menerapkan mekanisme adaptif Srinivas & Patnaik ($k_1 \dots k_6$)** untuk mengatur parameter probabilitas *crossover* dan mutasi secara otomatis sesuai sebaran fitness populasi guna menjaga keseimbangan eksplorasi dan eksploitasi.
* **Membangun pipeline *Walk-Forward Optimization* (WFO)** 2 fase validasi (*hyperparameter tuning*) dan 4 fase pengujian *out-of-sample*, memperhitungkan efek *weight drift* dan friksi biaya transaksi riil.
* **Mencatatkan kinerja portofolio *out-of-sample*** dengan **Total Return 46,51%** (Annual Return 50,61%) dan **Sharpe Ratio 2,1036**, melampaui Standard GA (Return 33,90%, Sharpe 1,45), *Equal Weight* (Return 38,51%, Sharpe 1,60), dan Indeks IDX30 (Return 13,48%, Sharpe 0,38).
* **Menguji signifikansi statistik secara ketat** melalui uji Shapiro-Wilk, Paired t-test, Wilcoxon Signed-Rank, serta koreksi Holm-Bonferroni, membuktikan bahwa keunggulan nilai fitness konvergen AGA terhadap Standard GA valid dan signifikan ($p < 0,001$).
* **Mendeploy Quantitative Investment Terminal bertaraf institusional** berbasis Streamlit & Plotly (estetika FactSet/Bloomberg), dilengkapi analisis kurva *equity* 4 fase, simulasi evolusioner real-time, serta kalkulator konversi bobot ke lot saham Bursa Efek Indonesia (BEI: **1 Lot = 100 lembar**).

---

## Technical Skills & ATS Keyword Tags

Salin daftar kata kunci berikut ke bagian **Skills** pada CV Anda untuk meningkatkan keterbacaan oleh sistem ATS (*Applicant Tracking System*):

```text
Technical Domains:
Quantitative Finance, Portfolio Optimization, Modern Portfolio Theory (Markowitz Mean-Variance), Evolutionary Computation, Genetic Algorithms (SGA & AGA), Metaheuristics, Hyperparameter Tuning, Walk-Forward Optimization (WFO), Time-Series Backtesting, Risk Management, Quantitative Analytics.

Statistical & Mathematical Methods:
Multi-Objective Optimization, Herfindahl-Hirschman Index (HHI), Sharpe Ratio, Maximum Drawdown (MDD), Parametric & Non-Parametric Hypothesis Testing, Shapiro-Wilk Test, Paired Student's t-test, Wilcoxon Signed-Rank Test, Exact Permutation / Sign-Flip Test, Bootstrap Confidence Intervals, Holm-Bonferroni Correction.

Programming & Tools:
Python, Streamlit, Plotly, NumPy, Pandas, SciPy, Matplotlib, Seaborn, yfinance, Jupyter Notebook, Git, Object-Oriented Programming (OOP), Interactive Web Dashboards.
```

---

## Ringkasan Perbandingan Performa (*Quick Reference Sheet*)

| Metrik Finansial | AGA (4x Rebalance) | Standard GA | Equal Weight (1/N) | IDX30 Index | Selisih AGA vs Benchmark |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Return (%)** | **46.49%** | 33.90% | 38.51% | 13.48% | **+33.01%** |
| **Annual Return (%)** | **50.59%** | 36.75% | 41.82% | 14.53% | **+36.06%** |
| **Annual Volatility (%)** | **21.21%** | 21.26% | 22.44% | 22.63% | **-1.42%** |
| **Sharpe Ratio** | **2.1025** | 1.4467 | 1.5961 | 0.3767 | **+1.7258 (+458%)** |
| **Max Drawdown (%)** | **-15.74%** | -17.35% | -18.78% | -17.14% | **+1.40% (Lebih Tangguh)** |
| **Total Biaya Transaksi (%)** | 0.43% | 0.50% | 0.15% | 0.38% | Biaya realistis |
| **Uji Signifikansi Fitness** | **$p < 0.001$** | Baseline | - | - | **Signifikan di semua 4 fase** |

