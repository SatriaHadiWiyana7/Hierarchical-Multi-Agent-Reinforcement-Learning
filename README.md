# Proyek MARL Kontrol Lampu Lalu Lintas (SUMO & RLlib)

Proyek ini mengimplementasikan sistem **Multi-Agent Reinforcement Learning (MARL)** untuk optimasi kontrol lampu lalu lintas yang adaptif. Sistem ini menggunakan **SUMO (Simulation of Urban MObility)** sebagai lingkungan simulasi dan **Ray RLlib** dengan algoritma **PPO (Proximal Policy Optimization)** untuk melatih agen-agen.

## 🔬 Metode

* **Reinforcement Learning (RL)**: Agen belajar melalui interaksi coba-coba (*trial and error*) dengan lingkungan.
* **Multi-Agent RL (MARL)**: Setiap lampu lalu lintas adalah agen independen yang belajar berkoordinasi.
* **Proximal Policy Optimization (PPO)**: Algoritma RL yang stabil dan efisien untuk melatih *policy* agen.
* **Simulasi (SUMO)**: Menggunakan SUMO (via TraCI) sebagai lingkungan simulasi lalu lintas mikroskopis.

---

## 🚀 Panduan Cepat

### 1. Persiapan (PENTING)

**Wajib:** Pastikan Anda telah meng-install simulator **SUMO** dan menambahkannya ke `PATH` sistem Anda.
* Unduh di: [situs web resmi Eclipse SUMO](https://www.eclipse.org/sumo/)
* Verifikasi dengan mengetik `sumo` di terminal Anda.

### 2. Instalasi Proyek

```bash
# 1. Clone repositori
git clone [https://github.com/SatriaHadiWiyana7/Multi-Agent-Reinforcement-Learning-MARL-.git](https://github.com/SatriaHadiWiyana7/Multi-Agent-Reinforcement-Learning-MARL-.git)
cd Multi-Agent-Reinforcement-Learning-MARL-

# 2. Buat & aktifkan virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install semua library
pip install -r requirements.txt

# 4. Menjalankan Perintah
python train.py

# 5. Memantau Training (di terminal baru):
tensorboard --logdir=ray_results

# 6. Cetak Grafik 
python grafik.py
