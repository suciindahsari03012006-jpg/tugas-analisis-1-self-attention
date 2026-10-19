"""
========================================================================================
LABORATORIUM KECERDASAN BUATAN (MK55) — PEKAN KE-5
UNIVERSITAS ISLAM NEGERI (UIN) ALAUDDIN MAKASSAR
FAKULTAS SAINS DAN TEKNOLOGI • PROGRAM STUDI SISTEM INFORMASI
========================================================================================
Topik         : Arsitektur Transformer & Mekanisme Scaled Dot-Product Self-Attention
SubCPMK       : SubCPMK 04 (CPMK-2 / CPL02)
Tugas Terkait : Tugas Analisis 1 — Diagram Alur Transformer & Kalkulasi Matriks Q, K, V
Bobot Nilai   : 5% dari Total Nilai Semester
Dosen Pengampu: Adhy Rizaldy, M.Kom. & Muhammad Rezky, S.Pd., M.Kom.
========================================================================================

PETUNJUK PRAKTIKUM:
Skrip ini mengimplementasikan mekanisme fundamental Transformer dari prinsip matematika
pertama (first principles) menggunakan pustaka murni NumPy dan visualisasi Matplotlib:
1. Positional Encoding Sinusoidal (Vaswani et al., 2017)
2. Kalkulasi Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
3. Causal Masking (Look-Ahead Mask) untuk Lapisan Decoder
4. Mekanisme Multi-Head Self-Attention (Parallel Projections & Concatenation)
5. Studi Kasus Resolusi Ambiguitas Konteks ("animal" vs "street" pada kata ganti "it")
6. Visualisasi Matriks Bobot Perhatian (Attention Heatmap) yang disimpan otomatis ke disk.
========================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt

# Set seed untuk reproduktibilitas eksperimen matematis
np.random.seed(42)

# ======================================================================================
# BAGIAN 1: FUNGSI BANTU MATEMATIKA (SOFTMAX NUMERIK STABIL)
# ======================================================================================
def softmax(x, axis=-1):
    """
    Menghitung fungsi Softmax dengan trik stabilitas numerik (mengurangi nilai maksimum)
    untuk mencegah overflow eksponensial (np.exp) saat menangani nilai skalar besar.
    """
    # Kurangi dengan nilai max pada axis terakhir untuk mencegah inf / NaN
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ======================================================================================
# BAGIAN 2: POSITIONAL ENCODING SINUSOIDAL (VASWANI ET AL., 2017)
# ======================================================================================
def get_positional_encoding(seq_len, d_model):
    """
    Menghasilkan matriks Positional Encoding berdimensi (seq_len, d_model)
    menggunakan fungsi gelombang sinus (indeks genap 2i) dan kosinus (indeks ganjil 2i+1).

    Formula:
      PE(pos, 2i)   = sin( pos / (10000 ^ (2i / d_model)) )
      PE(pos, 2i+1) = cos( pos / (10000 ^ (2i / d_model)) )
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis] # (seq_len, 1)

    # Divisor eksponensial untuk setiap pasangan dimensi
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Terapkan sinus pada kolom indeks genap (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)
    # Terapkan kosinus pada kolom indeks ganjil (1, 3, 5, ...)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


# ======================================================================================
# BAGIAN 3: SCALED DOT-PRODUCT ATTENTION
# ======================================================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Menghitung Scaled Dot-Product Attention sesuai persamaan paper 'Attention Is All You Need':
        Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + mask ) * V

    Parameter:
      Q    : Matriks Query berukuran (..., seq_len_q, d_k)
      K    : Matriks Key berukuran (..., seq_len_k, d_k)
      V    : Matriks Value berukuran (..., seq_len_v, d_v), biasanya seq_len_k == seq_len_v
      mask : Matriks topeng biner opsional berukuran (seq_len_q, seq_len_k) untuk Decoder

    Return:
      output           : Representasi kontekstual baru berukuran (..., seq_len_q, d_v)
      attention_weights: Matriks bobot probabilitas perhatian (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Langkah 1: Matriks perkalian titik (Dot-Product) antara Q dan K transpose
    # Matriks skor berukuran (seq_len_q, seq_len_k)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2))

    # Langkah 2: Penskalaan dengan 1 / sqrt(d_k) untuk menstabilkan gradien Softmax
    scaled_scores = scores / np.sqrt(d_k)

    # Langkah 3: Terapkan masking jika ada (cth: Look-ahead mask pada Decoder)
    if mask is not None:
        # Nilai bertanda True/1 pada mask akan diisi angka sangat negatif (-1e9)
        # sehingga setelah dieksponensialkan pada Softmax nilainya menjadi persis 0.0
        scaled_scores = np.where(mask == 1, -1e9, scaled_scores)

    # Langkah 4: Hitung probabilitas Softmax di sepanjang baris Key (axis terakhir)
    attention_weights = softmax(scaled_scores, axis=-1)

    # Langkah 5: Perkalian bobot perhatian terdistribusi dengan matriks Value (V)
    output = np.matmul(attention_weights, V)

    return output, attention_weights


# ======================================================================================
# BAGIAN 4: CAUSAL / LOOK-AHEAD MASKING (DECODER MASK)
# ======================================================================================
def create_causal_mask(seq_len):
    """
    Membuat matriks segitiga atas (upper-triangular) untuk Decoder.
    Elemen di atas diagonal utama bernilai 1 (dimask / disembunyikan),
    sehingga kata pada posisi t dilarang mengintip kata pada posisi t+1, t+2, dst.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask


# ======================================================================================
# BAGIAN 5: MULTI-HEAD ATTENTION MECHANISM
# ======================================================================================
class SimpleMultiHeadAttention:
    """
    Implementasi modular Multi-Head Attention:
    1. Memproyeksikan input ke Q, K, V untuk setiap Head secara terpisah
    2. Menghitung scaled dot-product attention pada masing-masing head
    3. Menggabungkan (concatenate) luaran seluruh head
    4. Menerapkan proyeksi linier akhir menggunakan matriks bobot W_O
    """
    def __init__(self, d_model=64, num_heads=4):
        assert d_model % num_heads == 0, "d_model harus habis dibagi num_heads!"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Inisialisasi bobot proyeksi linier (He/Xavier random normal)
        self.W_q = np.random.randn(d_model, d_model) * 0.05
        self.W_k = np.random.randn(d_model, d_model) * 0.05
        self.W_v = np.random.randn(d_model, d_model) * 0.05
        self.W_o = np.random.randn(d_model, d_model) * 0.05

    def split_heads(self, x, seq_len):
        # Mengubah bentuk matriks dari (seq_len, d_model) -> (num_heads, seq_len, d_k)
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return np.transpose(x, (1, 0, 2))

    def forward(self, Q_in, K_in, V_in, mask=None):
        seq_len_q = Q_in.shape[0]
        seq_len_k = K_in.shape[0]

        # 1. Proyeksi Linier Awal
        Q = np.dot(Q_in, self.W_q)
        K = np.dot(K_in, self.W_k)
        V = np.dot(V_in, self.W_v)

        # 2. Pecah ke beberapa Head
        Q_heads = self.split_heads(Q, seq_len_q) # (num_heads, seq_len_q, d_k)
        K_heads = self.split_heads(K, seq_len_k) # (num_heads, seq_len_k, d_k)
        V_heads = self.split_heads(V, seq_len_k) # (num_heads, seq_len_k, d_k)

        # 3. Hitung Attention untuk setiap Head
        out_heads = []
        weights_heads = []
        for h in range(self.num_heads):
            out_h, w_h = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], mask=mask)
            out_heads.append(out_h)
            weights_heads.append(w_h)

        # 4. Gabungkan (Concatenate) seluruh Head: (num_heads, seq_len_q, d_k) -> (seq_len_q, d_model)
        out_concat = np.concatenate(out_heads, axis=-1)

        # 5. Proyeksi Linier Akhir dengan W_O
        output = np.dot(out_concat, self.W_o)

        return output, np.array(weights_heads)


# ======================================================================================
# BAGIAN 6: STUDI KASUS KONKRET — RESOLUSI AMBIGUITAS KATA ("IT")
# ======================================================================================
def run_coreference_case_study():
    """Studi kasus Bahasa Indonesia dengan bobot sintetis yang dapat dijelaskan.

    Catatan: ini simulasi pedagogis, bukan model bahasa yang dilatih. Bias Q/K
    dibuat eksplisit agar contoh hubungan kata dapat diamati secara konsisten.
    """
    print("\n" + "="*80)
    print("STUDI KASUS: SIMULASI SELF-ATTENTION BAHASA INDONESIA")
    print("="*80)

    tokens = ["Kucing", "itu", "tidak", "melewati", "jalan", "karena", "ia", "lelah"]
    seq_len = len(tokens)
    d_model = 16
    d_k = 16

    print(f'Kalimat Analisis: "{" ".join(tokens)}"')
    print(f"Jumlah Token    : {seq_len} token")
    print(f"Dimensi Vektor  : {d_model} fitur")
    print("Catatan         : bobot Q/K sintetis untuk demonstrasi, bukan hasil model terlatih.\n")

    embeddings = np.random.randn(seq_len, d_model) * 0.1
    W_q = np.random.randn(d_model, d_k) * 0.2
    W_k = np.random.randn(d_model, d_k) * 0.2
    W_v = np.random.randn(d_model, d_model) * 0.2
    Q = np.dot(embeddings, W_q)
    K = np.dot(embeddings, W_k)
    V = np.dot(embeddings, W_v)

    idx_kucing = tokens.index("Kucing")
    idx_jalan = tokens.index("jalan")
    idx_ia = tokens.index("ia")
    idx_lelah = tokens.index("lelah")

    # Bentuk pola contoh secara eksplisit agar relasi yang sedang didemonstrasikan
    # tampak pada heatmap. Nilai ini tidak diklaim sebagai hasil pelatihan.
    Q[idx_ia] = 0.0
    Q[idx_ia, 0] = np.sqrt(d_k)
    K[idx_kucing] = 0.0
    K[idx_kucing, 0] = 4.0
    K[idx_lelah] = 0.0
    K[idx_lelah, 0] = 1.5
    K[idx_jalan] = 0.0
    K[idx_jalan, 0] = 0.3

    out, attn_matrix = scaled_dot_product_attention(Q, K, V)
    ia_weights = attn_matrix[idx_ia]
    print("DISTRIBUSI BOBOT PERHATIAN UNTUK QUERY TOKEN: 'ia'")
    print("-" * 55)
    sorted_indices = np.argsort(ia_weights)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        bar = "#" * int(ia_weights[idx] * 40)
        print(f"{rank:2d}. Token '{tokens[idx]:<10}' : {ia_weights[idx]:.4f} ({ia_weights[idx]*100:5.1f}%) | {bar}")

    print("\nINTERPRETASI:")
    print(f"- Dalam simulasi ini, query 'ia' memberi bobot terbesar kepada 'Kucing' ({ia_weights[idx_kucing]*100:.1f}%).")
    print("- Secara bahasa, kata 'lelah' mendukung pembacaan bahwa 'ia' merujuk pada kucing.")
    print("- Ini ilustrasi pola perhatian sintetis, bukan bukti empiris kemampuan model terlatih.\n")
    return tokens, attn_matrix


# ======================================================================================
# BAGIAN 7: VISUALISASI ATTENTION HEATMAP KE GAMBAR
# ======================================================================================
def plot_attention_heatmap(tokens, attn_matrix, output_path="attention_heatmap_pekan_05.png"):
    """
    Membuat grafik heatmap korelasi matriks Self-Attention 2D dan menyimpannya ke berkas citra.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Warna peta panas: tema modern dark / viridis
    cax = ax.matshow(attn_matrix, cmap='viridis')
    fig.colorbar(cax, fraction=0.046, pad=0.04, label="Skor Bobot Perhatian (Softmax Probability)")

    # Atur label sumbu
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='left', fontsize=10, fontweight='bold')
    ax.set_yticklabels(tokens, fontsize=10, fontweight='bold')

    ax.set_xlabel("Key Tokens (K)", fontsize=11, fontweight='bold', labelpad=12)
    ax.set_ylabel("Query Tokens (Q)", fontsize=11, fontweight='bold', labelpad=12)
    ax.set_title("Visualisasi Matriks Self-Attention (Pekan 5: SubCPMK 04)\nUIN Alauddin Makassar — Sistem Informasi",
                 fontsize=12, fontweight='bold', pad=24)

    # Tampilkan teks angka pada tiap sel jika dimensi tidak terlalu padat
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            val = attn_matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=7.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Heatmap visualisasi berhasil disimpan ke: {output_path}")


# ======================================================================================
# BAGIAN 8: DEMONSTRASI CAUSAL LOOK-AHEAD MASK PADA DECODER
# ======================================================================================
def demo_causal_mask():
    print("\n" + "="*80)
    print("DEMONSTRASI: CAUSAL MASKING (LOOK-AHEAD MASK) PADA DECODER")
    print("="*80)
    sample_tokens = ["Saya", "sedang", "belajar", "Transformer", "AI"]
    n = len(sample_tokens)

    mask = create_causal_mask(n)
    print("Matriks Segitiga Atas Causal Mask (1 = Dimask / Terlarang Diintip):")
    print(mask)

    # Simulasi skor dot-product acak
    dummy_scores = np.random.randn(n, n)
    masked_scores = np.where(mask == 1, -1e9, dummy_scores)
    attn_masked = softmax(masked_scores, axis=-1)

    print("\nMatriks Bobot Perhatian Decoder Setelah Softmax:")
    print("Perhatikan bahwa nilai di atas diagonal utama menjadi 0.0000 (tidak bisa melihat masa depan)!")
    print("-" * 65)
    for i in range(n):
        row_str = " ".join([f"{attn_masked[i, j]:6.4f}" for j in range(n)])
        print(f"Query '{sample_tokens[i]:<11}': [ {row_str} ]")
    print("-" * 65)


# ======================================================================================
# BAGIAN 9: PANDUAN TUGAS ANALISIS 1 UNTUK MAHASISWA
# ======================================================================================
def print_assignment_brief():
    print("\n" + "="*80)
    print("PANDUAN TUGAS ANALISIS 1 (BOBOT 5% — SubCPMK 04 / CPMK-2 / CPL02)")
    print("="*80)
    print("""
TUGAS INDIVIDU MAHASISWA:
1. Dekonstruksi Matematis (35%):
   Hitung secara manual (pada selembar kertas / LaTeX) perkalian matriks Q * K^T
   untuk 3 kata sederhana dengan d_k = 2, lalu lakukan normalisasi Softmax.
2. Analisis Arsitektur Encoder vs Decoder (35%):
   Jelaskan mengapa model BERT hanya menggunakan blok Encoder, sedangkan model
   GPT (seperti ChatGPT) hanya menggunakan blok Decoder. Sebutkan kelebihan
   dan kelemahan masing-masing arsitektur!
3. Eksperimen Kode & Heatmap (30%):
   Ubah kalimat uji pada skrip ini menggunakan kalimat Bahasa Indonesia Anda sendiri
   (minimal 6 kata). Jalankan skrip ini untuk menghasilkan plot heatmap korelasi baru,
   lalu sertakan interpretasi Anda mengenai kata mana yang saling memperhatikan paling kuat.

PENGUMPULAN:
Format Berkas : Laporan PDF (maksimal 3 halaman) + Tautan Repositori GitHub
Portal Upload : LMS UIN Alauddin Makassar (Menu Tugas Analisis 1)
Tenggat Waktu : 1 pekan sebelum perkuliahan Pekan ke-6 dimulai.
""")
    print("="*80 + "\n")


# ======================================================================================
# EKSEKUSI UTAMA (MAIN ENTRY POINT)
# ======================================================================================
if __name__ == "__main__":
    print("Memulai Eksekusi Laboratorium Pekan 5: Transformer & Self-Attention...")

    # 1. Uji Positional Encoding
    pe = get_positional_encoding(seq_len=6, d_model=8)
    print("\nContoh Matriks Positional Encoding (seq_len=6, d_model=8):")
    print(np.round(pe, 3))

    # 2. Uji Multi-Head Attention
    mha = SimpleMultiHeadAttention(d_model=16, num_heads=2)
    dummy_x = np.random.randn(4, 16)
    out_mha, w_mha = mha.forward(dummy_x, dummy_x, dummy_x)
    print(f"\nMulti-Head Attention Test: Input {dummy_x.shape} -> Output {out_mha.shape}")
    print(f"Bentuk Matriks Bobot Multi-Head: {w_mha.shape} (heads, seq_len, seq_len)")

    # 3. Jalankan Studi Kasus Ambiguitas Kalimat
    tokens, attn_matrix = run_coreference_case_study()

    # 4. Hasilkan dan Simpan Visualisasi Heatmap
    plot_attention_heatmap(tokens, attn_matrix, "attention_heatmap_pekan_05.png")

    # 5. Demonstrasi Causal Masking Decoder
    demo_causal_mask()

    # 6. Tampilkan Panduan Tugas Analisis 1
    print_assignment_brief()
    print("Praktikum Pekan 5 selesai. Periksa keluaran dan interpretasi sebelum dikumpulkan.")
