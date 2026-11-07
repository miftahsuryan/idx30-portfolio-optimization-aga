# Portfolio & CV Highlights: IDX30 Portfolio Optimization using Adaptive Genetic Algorithm (AGA)

Dokumen ini menyediakan draf deskripsi proyek siap pakai untuk **Curriculum Vitae (CV)**, profil **LinkedIn**, dan portofolio profesional Anda, dirancang khusus untuk posisi **Data Scientist**, **Quantitative Analyst**, dan **Machine Learning Engineer**.

Seluruh angka dan parameter di dokumen ini telah diselaraskan **100% sesuai dengan hasil skripsi asli** (`Implementasi-fix.ipynb`).

---

## English Versions (Recommended for Tech & International Roles)

### Project Title & Links
- **Title Options:**
  - *IDX30 Stock Portfolio Optimization using Adaptive Genetic Algorithm & Walk-Forward Validation*
  - *Dynamic Portfolio Optimization Engine: Adaptive Genetic Algorithm (AGA) with Walk-Forward Backtesting*
- **Live Demo:** [https://portfolio-optimization-aga.streamlit.app](https://portfolio-optimization-aga.streamlit.app)
- **GitHub Repository:** [https://github.com/miftahsuryan/idx30-portfolio-optimization-aga](https://github.com/miftahsuryan/idx30-portfolio-optimization-aga)

### Option A: Compact Format (3 High-Impact Bullet Points — Ideal for 1-Page Resume)
* **Developed an end-to-end portfolio optimization framework** using an **Adaptive Genetic Algorithm (AGA)** on 22 consistent Indonesian Blue-Chip stocks (IDX30), dynamically adjusting crossover ($P_c$) and mutation ($P_m$) rates to prevent premature convergence.
* **Engineered a 4-phase Walk-Forward Optimization (WFO) backtest** incorporating realistic market frictions (0.15% buy, 0.25% sell fees), outperforming the IDX30 benchmark with a **46.49% Total Return** (vs. 13.48%), a **2.1025 Sharpe Ratio** (vs. 0.3767), and a **-15.74% Max Drawdown**.
* **Evaluated multi-run stochastic stability** across 80 paired optimization runs (4 phases $\times$ 10 seeds under an equal 44,000 NFE budget) using **Shapiro-Wilk normality tests**, **Paired t-tests**, and **Wilcoxon Signed-Rank tests**, confirming statistically significant objective fitness improvements ($p < 0.001$) over Standard Genetic Algorithms across all 4 phases.

---

### Option B: Detailed Technical Format (5-6 Bullet Points — Ideal for ML/Data Science Master Resume)
* **Architected a regularized Mean-Variance optimization engine** utilizing an **Adaptive Genetic Algorithm (AGA)** with Herfindahl-Hirschman Index (HHI) diversification penalties ($\lambda_{\text{div}} = 0.90$) and single-asset allocation caps ($w_{\max} = 10\%$) tuned via 2-phase WFO Grid Search and enforced with an iterative proportional excess-redistribution repair operator (`max_iters = 15`).
* **Designed adaptive operator scaling ($k_1 \dots k_6$)** that tunes crossover and mutation probabilities based on population fitness distribution, preventing local optimum stagnation and sustaining exploratory diversity across generations.
* **Built a Walk-Forward Optimization (WFO) backtesting pipeline** across 4 expanding-window out-of-sample phases (2025–2026), evaluating both isolated phase performance and continuous 4× quarterly rebalancing with price drift and transaction friction (0.15% buy / 0.25% sell fees).
* **Achieved a 46.49% cumulative out-of-sample return** (50.59% annualized) and **2.1025 Sharpe Ratio**, outperforming Standard GA (33.90% return, 1.4467 Sharpe), 1/N Equal Weight (38.51% return, 1.5961 Sharpe), and the IDX30 Market Benchmark (13.48% return, 0.3767 Sharpe).
* **Conducted paired multi-run statistical evaluation** (Shapiro-Wilk normality routing to Paired Student's t-test or Wilcoxon Signed-Rank test) across 80 shared-population runs, confirming superior objective fitness convergence ($p < 0.001$ in all 4 phases) and $2\times$–$3\times$ lower variance across seeds.
* **Deployed an interactive Quantitative Investment Terminal** using Streamlit and Plotly, featuring 4-phase WFO equity trajectories, real-time live evolutionary simulation, and Indonesia Stock Exchange (IDX) round-lot sizing calculations (`1 Lot = 100 shares`).

---

### Option C: Executive One-Liner (For Summary / Portfolio Header)
> Built an end-to-end financial portfolio optimization engine using Adaptive Genetic Algorithms (AGA) and 4-phase Walk-Forward backtesting on IDX30 constituents, achieving an out-of-sample Sharpe Ratio of **2.1025** and **46.49% Total Return** (`+33.01%` over the IDX30 benchmark).

---

## Versi Bahasa Indonesia (Untuk CV & Perusahaan Nasional)

### Judul & Tautan Proyek untuk CV
- **Pilihan Judul:** *Optimasi Portofolio Saham IDX30 Menggunakan Adaptive Genetic Algorithm (AGA) dan Walk-Forward Validation*
- **Live Demo:** [https://portfolio-optimization-aga.streamlit.app](https://portfolio-optimization-aga.streamlit.app)
- **GitHub Repository:** [https://github.com/miftahsuryan/idx30-portfolio-optimization-aga](https://github.com/miftahsuryan/idx30-portfolio-optimization-aga)

### Opsi A: Format Ringkas (3 Poin Utama — Cocok untuk Resume 1 Halaman)
* **Mengembangkan sistem optimasi portofolio investasi** berbasis **Adaptive Genetic Algorithm (AGA)** pada 22 saham konsisten indeks IDX30, mengadaptasi probabilitas *crossover* ($P_c$) dan mutasi ($P_m$) secara dinamis guna mengatasi konvergensi prematur.
* **Mengimplementasikan pengujian *Walk-Forward Optimization* (WFO) 4 fase** dengan memperhitungkan biaya transaksi riil (beli 0,15%, jual 0,25%) dan *rebalancing*, menghasilkan **Total Return 46,49%** (vs. Benchmark IDX30 13,48%), **Sharpe Ratio 2,1025** (vs. 0,3767), serta **Max Drawdown terkendali di -15,74%**.
* **Melakukan evaluasi stabilitas stokastik berpasangan** terhadap 80 eksperimen (4 fase $\times$ 10 *seed* dengan populasi awal bersama dan anggaran $44.000$ NFE setara) menggunakan uji normalitas **Shapiro-Wilk**, **Paired t-test**, dan **Wilcoxon Signed-Rank**, membuktikan konvergensi nilai fitness yang signifikan lebih optimal ($p < 0,001$) dibanding Standard GA pada seluruh fase.

---

### Opsi B: Format Teknis Mendalam (5-6 Poin — Cocok untuk CV Komprehensif)
* **Merancang fungsi objektif Mean-Variance teregulasi** yang mengintegrasikan penalti diversifikasi *Herfindahl-Hirschman Index* (HHI, $\lambda_{\text{div}} = 0,90$) dan batas maksimum bobot per aset ($w_{\max} = 10\%$) hasil *Grid Search tuning* dengan mekanisme perbaikan (*repair_weights*) redistribusi kelebihan bobot proporsional iteratif (`max_iters = 15`).
* **Menerapkan mekanisme adaptif ($k_1 \dots k_6$)** untuk mengatur probabilitas *crossover* dan mutasi secara otomatis sesuai distribusi fitness populasi guna menjaga keseimbangan eksplorasi dan eksploitasi.
* **Membangun pipeline *Walk-Forward Optimization* (WFO)** 2 fase validasi (*hyperparameter tuning*) dan 4 fase pengujian *out-of-sample* (terisolasi per fase maupun kumulatif 4× *rebalancing*), memperhitungkan efek *weight drift* dan friksi biaya transaksi riil.
* **Mencatatkan kinerja portofolio *out-of-sample* kumulatif** dengan **Total Return 46,49%** (Annual Return 50,59%) dan **Sharpe Ratio 2,1025**, melampaui Standard GA (Return 33,90%, Sharpe 1,4467), *Equal Weight* (Return 38,51%, Sharpe 1,5961), dan Indeks IDX30 (Return 13,48%, Sharpe 0,3767).
* **Menguji stabilitas konvergensi multi-run secara statistik** melalui uji normalitas Shapiro-Wilk diikuti pemilihan otomatis Paired Student's t-test atau Wilcoxon Signed-Rank test pada 80 eksekusi berpasangan, membuktikan bahwa keunggulan fungsi objektif AGA konsisten dan signifikan ($p < 0,001$ di keempat fase).
* **Mendeploy aplikasi web interaktif Quantitative Investment Terminal** berbasis Streamlit & Plotly, dilengkapi analisis kurva *equity* 4 fase, simulasi evolusioner real-time, serta kalkulator konversi bobot ke lot saham Bursa Efek Indonesia (BEI: **1 Lot = 100 lembar**).

---

## Technical Skills & ATS Keyword Tags

```text
Technical Domains:
Quantitative Finance, Portfolio Optimization, Modern Portfolio Theory (Markowitz Mean-Variance), Evolutionary Computation, Genetic Algorithms (SGA & AGA), Metaheuristics, Hyperparameter Tuning, Walk-Forward Optimization (WFO), Time-Series Backtesting, Risk Management, Quantitative Analytics.

Statistical & Mathematical Methods:
Regularized Objective Optimization, Herfindahl-Hirschman Index (HHI), Sharpe Ratio, Maximum Drawdown (MDD), Parametric & Non-Parametric Hypothesis Testing, Shapiro-Wilk Normality Test, Paired Student's t-test, Wilcoxon Signed-Rank Test.

Programming & Tools:
Python, Streamlit, Plotly, NumPy, Pandas, SciPy, Matplotlib, Seaborn, yfinance, Jupyter Notebook, Git, Modular Software Engineering, Interactive Web Dashboards.
```

---

## Ringkasan Perbandingan Performa (*Quick Reference Sheet*)

| Metrik Finansial | AGA (4x Rebalance) | Standard GA (4x Rebalance) | Equal Weight (1/N) | IDX30 Index (4x Rebalance) | Selisih AGA vs Benchmark IDX30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Return (%)** | **46.4914%** | 33.8970% | 38.5142% | 13.4829% | **+33.0085%** |
| **Annual Return (%)** | **50.5937%** | 36.7545% | 41.8176% | 14.5260% | **+36.0677%** |
| **Annual Volatility (%)** | **21.2094%** | 21.2583% | 22.4410% | 22.6332% | **-1.4238%** |
| **Sharpe Ratio** | **2.1025** | 1.4467 | 1.5961 | 0.3767 | **+1.7258 (+458%)** |
| **Max Drawdown (%)** | **-15.7424%** | -17.3522% | -18.7795% | -17.1428% | **+1.4004% (Lebih Tangguh)** |
| **Total Biaya Transaksi (%)** | 0.4291% | 0.4979% | 0.1500% | 0.3775% | Biaya realistis |
| **Nilai Akhir Portofolio** | **1.4649** | 1.3390 | 1.3851 | 1.1348 | **+0.3301** |
| **Uji Signifikansi Fitness (10 Run × 4 Fase)** | **$p < 0.001$** | Baseline | - | - | **Signifikan di semua 4 fase** |
