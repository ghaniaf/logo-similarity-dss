"""
app.py v3 — DSS Analisis Kemiripan Logo Merek Dagang
Universitas Widyatama | Ghania Fazila (41122100060)
- Sidebar untuk pengaturan
- XAI berdampingan query + Rank-1
- Info kandidat ke bawah (baris per baris)
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import json
import os
from pathlib import Path
from PIL import Image
from torchvision import models
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_grad_cam import EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

BASE_DIR      = Path(__file__).parent
MODEL_PATH    = BASE_DIR / "hybrid_best.pt"
EMBEDS_PATH   = BASE_DIR / "pdki_embeddings.npy"
HSV_PATH      = BASE_DIR / "pdki_hsv.npy"
METADATA_PATH = BASE_DIR / "pdki_metadata.json"
IMAGES_DIR    = BASE_DIR / "pdki_images"

IMG_SIZE             = 224
EMBED_DIM            = 256
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)
HSV_BINS             = [16, 8, 8]
TOP_K                = 10
ALPHA_FUSION         = 1.0
SIMILARITY_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LetterboxResize:
    def __init__(self, size=224):
        self.size = size
    def __call__(self, img):
        img = img.convert("RGB")
        w, h = img.size
        scale = self.size / max(w, h)
        img = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), (0,0,0))
        canvas.paste(img, ((self.size-img.width)//2, (self.size-img.height)//2))
        return canvas

tf_eff = T.Compose([
    LetterboxResize(IMG_SIZE), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
tf_hc = T.Compose([LetterboxResize(IMG_SIZE), T.ToTensor()])

def extract_hog(tensor_img):
    img_np = (tensor_img.permute(1,2,0).numpy()*255).astype(np.uint8)
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return hog(gray, orientations=HOG_ORIENTATIONS,
               pixels_per_cell=HOG_PIXELS_PER_CELL,
               cells_per_block=HOG_CELLS_PER_BLOCK,
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)

def extract_hsv(tensor_img):
    img_np = (tensor_img.permute(1,2,0).numpy()*255).astype(np.uint8)
    hsv    = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv],[0],None,[HSV_BINS[0]],[0,180]).flatten()
    s = cv2.calcHist([hsv],[1],None,[HSV_BINS[1]],[0,256]).flatten()
    v = cv2.calcHist([hsv],[2],None,[HSV_BINS[2]],[0,256]).flatten()
    hist = np.concatenate([h,s,v])
    if hist.sum() > 0: hist = hist / hist.sum()
    return hist.astype(np.float32)


class HybridLogoModel(nn.Module):
    def __init__(self, hog_dim, embed_dim=256):
        super().__init__()
        eff = models.efficientnet_v2_s(weights=None)
        self.backbone = nn.Sequential(*list(eff.children())[:-1])
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.hog_proj = nn.Sequential(
            nn.Linear(hog_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3))
        self.embedding_layer = nn.Sequential(
            nn.Linear(1280+256,512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(512,embed_dim))
    def forward(self, img, hog_vec):
        deep  = self.pool(self.backbone(img)).flatten(1)
        hog_f = self.hog_proj(hog_vec)
        return nn.functional.normalize(
            self.embedding_layer(torch.cat([deep,hog_f],dim=1)), p=2, dim=1)

class ModelWrapperCAM(nn.Module):
    def __init__(self, model, hog_fixed, ref_emb=None):
        super().__init__()
        self.model = model; self.hog_fixed = hog_fixed
        self.ref = torch.tensor(ref_emb, dtype=torch.float32) if ref_emb is not None else None
    def forward(self, img):
        emb = self.model(img, self.hog_fixed)
        if self.ref is not None:
            return (emb * self.ref.to(emb.device)).sum(dim=1, keepdim=True)
        return emb.norm(dim=1, keepdim=True)

def resolve_image_path(meta):
    orig = Path(meta.get("path",""))
    if orig.exists(): return orig
    filename = meta.get("filename","")
    if "PDKI/" in meta.get("path",""):
        r = IMAGES_DIR / meta["path"].split("PDKI/")[-1]
        if r.exists(): return r
    if filename and IMAGES_DIR.exists():
        matches = list(IMAGES_DIR.rglob(filename))
        if matches: return matches[0]
    return None

@st.cache_resource
def load_model_and_db():
    missing = [str(f) for f in [MODEL_PATH,EMBEDS_PATH,HSV_PATH,METADATA_PATH] if not Path(f).exists()]
    if missing: raise FileNotFoundError(f"File tidak ditemukan: {', '.join(missing)}")
    _t = tf_hc(LetterboxResize(IMG_SIZE)(Image.new("RGB",(100,100))))
    hog_dim = len(extract_hog(_t))
    model = HybridLogoModel(hog_dim, EMBED_DIM).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters(): p.requires_grad_(True)
    db_embeds = np.load(EMBEDS_PATH)
    db_hsv    = np.load(HSV_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        db_meta = json.load(f)
    return model, db_embeds, db_hsv, db_meta

@torch.no_grad()
def get_embedding(img_pil, model):
    t_eff = tf_eff(img_pil).unsqueeze(0).to(DEVICE)
    t_hc  = tf_hc(img_pil)
    hog_v = torch.tensor(extract_hog(t_hc), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    hsv_v = extract_hsv(t_hc)
    emb   = model(t_eff, hog_v).cpu().numpy().flatten()
    return emb, hsv_v, t_eff, t_hc, hog_v

def retrieve(emb_q, hsv_q, db_embeds, db_hsv,
             k=TOP_K, alpha=ALPHA_FUSION, threshold=SIMILARITY_THRESHOLD):
    hybrid_scores = cosine_similarity(emb_q.reshape(1,-1), db_embeds).flatten()
    color_scores  = cosine_similarity(hsv_q.reshape(1,-1), db_hsv).flatten()
    final_scores  = alpha * hybrid_scores + (1-alpha) * color_scores
    if final_scores.max() > 0.9999:
        final_scores[final_scores.argmax()] = -1
    top_idx    = np.argsort(final_scores)[::-1][:k]
    top_scores = final_scores[top_idx]
    is_similar = top_scores >= threshold
    return top_idx, top_scores, is_similar

def generate_heatmap(img_pil, model, hog_v, t_eff, ref_emb=None):
    wrapped = ModelWrapperCAM(model, hog_v, ref_emb)
    wrapped.eval()
    for p in wrapped.parameters(): p.requires_grad_(True)
    with EigenGradCAM(model=wrapped, target_layers=[model.backbone[0][-1]]) as cam:
        grayscale = cam(input_tensor=t_eff, targets=None)[0]
    mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
    img_np = (t_eff.squeeze().permute(1,2,0).cpu().numpy()*std+mean).clip(0,1).astype(np.float32)
    img_display = np.clip(img_np*3.0,0,1) if img_np.mean()<0.2 else img_np
    overlay = show_cam_on_image(img_display, grayscale, use_rgb=True)
    return overlay, grayscale

def heatmap_to_rgb(grayscale):
    return cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255*grayscale), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)

def status_badge(status):
    s = status.lower()
    if "didaftar" in s:
        st.success(f"✅ {status}")
    elif "tolak" in s:
        st.error(f"🚫 {status}")
    elif any(x in s for x in ["pengumuman","substantif","proses"]):
        st.warning(f"⏳ {status}")
    else:
        st.info(f"ℹ️ {status}")


def main():
    st.set_page_config(
        page_title="DSS Kemiripan Logo Merek",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── SIDEBAR ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚖️ DSS Kemiripan Logo")
        st.caption("Pemeriksaan Substantif Merek | DJKI")
        st.divider()

        st.markdown("### ⚙️ Pengaturan")
        top_k        = st.slider("Jumlah kandidat (Top-K)", 3, 20, TOP_K)
        threshold    = st.slider("Threshold kemiripan", 0.0, 1.0,
                                  SIMILARITY_THRESHOLD, 0.05)
        filter_kelas = st.text_input("Filter Kelas NICE",
                                      placeholder="contoh: 25 (kosongkan = semua)")
        show_gradcam = st.checkbox("Tampilkan Grad-CAM", value=True)

        st.divider()
        st.info("Sistem ini adalah alat bantu DSS dan **bukan penentu keputusan akhir**.",
                icon="ℹ️")
        st.caption("Ghania Fazila (41122100060) | Universitas Widyatama")

    # ── HEADER ──────────────────────────────────────────────────────
    st.markdown("# Sistem Analisis Kemiripan Logo Merek Dagang")
    st.caption("Decision Support System untuk Pemeriksaan Substantif Merek | DJKI")
    st.divider()

    # ── LOAD MODEL ──────────────────────────────────────────────────
    with st.spinner("Memuat model dan database PDKI..."):
        try:
            model, db_embeds, db_hsv, db_meta = load_model_and_db()
            st.success(f"Model siap. Database: {len(db_meta):,} logo PDKI.", icon="✅")
        except FileNotFoundError as e:
            st.error(str(e))
            st.code("hybrid_best.pt\npdki_embeddings.npy\npdki_hsv.npy\npdki_metadata.json")
            st.stop()

    st.divider()

    # ── UPLOAD ──────────────────────────────────────────────────────
    col_input, col_setting = st.columns([2, 1])
    with col_input:
        st.subheader("📤 Upload Logo Query")
        uploaded = st.file_uploader(
            "Upload logo merek yang ingin dianalisis",
            type=["jpg","jpeg","png"])
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, caption="Logo Query", width=280)

    # ── ANALISIS ────────────────────────────────────────────────────
    if uploaded and st.button("🔍 Analisis Kemiripan",
                               type="primary", use_container_width=True):

        with st.spinner("Mengekstrak fitur logo..."):
            emb_q, hsv_q, t_eff, t_hc, hog_v = get_embedding(img_pil, model)

        # Filter kelas NICE
        if filter_kelas.strip():
            vidx = [i for i,m in enumerate(db_meta)
                    if str(m.get("kelas_nice","")) == filter_kelas.strip()]
            if not vidx:
                st.warning(f"Tidak ada logo kelas NICE {filter_kelas} di database.")
                st.stop()
            db_e = db_embeds[vidx]; db_h = db_hsv[vidx]
            db_m = [db_meta[i] for i in vidx]
            st.caption(f"Filter aktif: Kelas NICE {filter_kelas} ({len(vidx)} logo)")
        else:
            db_e, db_h, db_m = db_embeds, db_hsv, db_meta

        with st.spinner("Mencari kandidat mirip..."):
            top_idx, top_scores, is_similar = retrieve(
                emb_q, hsv_q, db_e, db_h, k=top_k, threshold=threshold)

        n_similar = int(is_similar.sum())
        st.divider()
        st.subheader("📊 Ringkasan Hasil")

        c1, c2, c3 = st.columns(3)
        c1.metric("Kandidat diperiksa", f"{top_k}")
        c2.metric("Berpotensi mirip", f"{n_similar}",
                  delta="Perlu diperiksa" if n_similar>0 else "Aman",
                  delta_color="inverse" if n_similar>0 else "normal")
        c3.metric("Skor tertinggi", f"{top_scores[0]:.4f}")

        if n_similar == 0:
            st.success("✅ Tidak ditemukan logo yang berpotensi mirip di atas threshold.")
        elif n_similar <= 2:
            st.warning(f"⚠️ {n_similar} logo berpotensi mirip — perlu pemeriksaan lebih lanjut.")
        else:
            st.error(f"🚨 {n_similar} logo berpotensi mirip — perlu pemeriksaan mendalam.")

        st.divider()

        # ── GRAD-CAM berdampingan query + Rank-1 ────────────────────
        if show_gradcam:
            st.subheader("🔬 Explainability — Area yang Diperhatikan Model")
            st.caption(
                "Heatmap menunjukkan area yang paling berkontribusi terhadap keputusan "
                "kemiripan sistem. **Kiri:** logo query yang dianalisis. "
                "**Kanan:** kandidat Rank-1 yang paling mirip. (EigenGradCAM)"
            )

            with st.spinner("Generating heatmap query dan kandidat Rank-1..."):
                # Heatmap QUERY
                ref_emb_r1        = db_e[top_idx[0]]
                overlay_q, gray_q = generate_heatmap(
                    img_pil, model, hog_v, t_eff, ref_emb=ref_emb_r1)

                # Heatmap KANDIDAT RANK-1
                rank1_meta = db_m[top_idx[0]]
                rank1_path = resolve_image_path(rank1_meta)
                overlay_r1 = gray_r1 = rank1_img = None
                if rank1_path:
                    try:
                        rank1_img  = Image.open(rank1_path).convert("RGB")
                        t_eff_r1   = tf_eff(rank1_img).unsqueeze(0).to(DEVICE)
                        t_hc_r1    = tf_hc(rank1_img)
                        hog_v_r1   = torch.tensor(
                            extract_hog(t_hc_r1), dtype=torch.float32
                        ).unsqueeze(0).to(DEVICE)
                        overlay_r1, gray_r1 = generate_heatmap(
                            rank1_img, model, hog_v_r1, t_eff_r1, ref_emb=emb_q)
                    except Exception as e:
                        st.caption(f"Heatmap kandidat tidak dapat di-generate: {e}")

            # Panel berdampingan
            col_q, col_div, col_r1 = st.columns([5, 0.3, 5])

            with col_q:
                st.markdown("**🔵 Logo Query**")
                q1, q2, q3 = st.columns(3)
                q1.image(img_pil, caption="Asli", use_container_width=True)
                q2.image(heatmap_to_rgb(gray_q), caption="Heatmap", use_container_width=True)
                q3.image(overlay_q, caption="Overlay", use_container_width=True)

            with col_div:
                st.markdown(
                    "<div style='border-left:2px solid #444;height:220px;margin:auto'></div>",
                    unsafe_allow_html=True)

            with col_r1:
                nama_r1 = rank1_meta.get("nama_merek","Rank-1")[:25]
                st.markdown(f"**🔴 Kandidat Rank-1: {nama_r1}**")
                if overlay_r1 is not None and rank1_img is not None:
                    r1, r2, r3 = st.columns(3)
                    r1.image(rank1_img, caption="Asli", use_container_width=True)
                    r2.image(heatmap_to_rgb(gray_r1), caption="Heatmap", use_container_width=True)
                    r3.image(overlay_r1, caption="Overlay", use_container_width=True)
                else:
                    st.warning("Gambar kandidat tidak tersedia untuk Grad-CAM.")

            st.caption(
                "💡 **Cara membaca:** Area merah/kuning = area yang paling diperhatikan "
                "model saat menilai kemiripan. Kalau area yang sama menonjol di kedua logo, "
                "itu mengindikasikan elemen visual yang menjadi dasar kemiripan."
            )
            st.divider()

        # ── HASIL RETRIEVAL (format ke bawah) ───────────────────────
        st.subheader(f"🏆 Top-{top_k} Kandidat dari Database PDKI")

        hasil_rows = []

        for rank, (idx, score, sim) in enumerate(
            zip(top_idx, top_scores, is_similar), 1):

            meta       = db_m[idx]
            img_path   = resolve_image_path(meta)
            nama_merek = meta.get("nama_merek","N/A")[:50]
            owner      = meta.get("owner_name","")
            no_perm    = meta.get("nomor_permohonan","")
            kelas      = meta.get("kelas_nice","")
            tgl        = meta.get("tanggal_permohonan","")
            status     = meta.get("status_permohonan","N/A")

            hasil_rows.append({
                "rank":rank, "score":score, "sim":sim,
                "nama_merek":nama_merek, "status":status,
                "nomor_permohonan":no_perm, "kelas_nice":kelas
            })

            with st.container():
                col_img, col_info = st.columns([1, 3])

                with col_img:
                    if img_path:
                        try:
                            st.image(Image.open(img_path).convert("RGB"),
                                     use_container_width=True)
                        except:
                            st.warning("Gambar tidak dapat dimuat")
                    else:
                        st.warning("⚠️ Gambar tidak ditemukan")

                with col_info:
                    # Baris 1: rank + skor + status mirip
                    badge = "🔴 Berpotensi mirip" if sim else "🟢 Di bawah threshold"
                    st.markdown(f"**Rank {rank}** &nbsp;|&nbsp; Skor: `{score:.4f}` &nbsp;|&nbsp; {badge}")
                    st.markdown(f"**Nama Merek:** {nama_merek}")
                    if owner:
                        st.markdown(f"**Pemilik:** {owner}")
                    if no_perm:
                        st.markdown(f"**No. Permohonan:** `{no_perm}`")
                    if kelas:
                        st.markdown(f"**Kelas NICE:** {kelas}")
                    if tgl:
                        st.markdown(f"**Tanggal Permohonan:** {tgl}")
                    status_badge(status)

                st.markdown("---")

        # ── EKSPOR ───────────────────────────────────────────────────
        st.divider()
        st.subheader("💾 Ekspor Hasil")

        lines = [
            "HASIL ANALISIS KEMIRIPAN LOGO MEREK DAGANG",
            "="*55,
            f"Query           : {uploaded.name}",
            f"Threshold       : {threshold}",
            f"Kandidat        : {top_k}",
            f"Berpotensi mirip: {n_similar}",
            f"Filter kelas    : {filter_kelas if filter_kelas.strip() else 'Semua kelas'}",
            "",
            f"{'Rank':<5} {'Skor':<8} {'Status':<22} {'No. Permohonan':<22} {'Nama Merek'}",
            "-"*80,
        ]
        for r in hasil_rows:
            lines.append(
                f"{r['rank']:<5} {r['score']:.4f}  "
                f"{r['status'].replace('(TM)','').strip():<22} "
                f"{r['nomor_permohonan']:<22} {r['nama_merek']}"
            )
        lines += [
            "", "="*55,
            "DISCLAIMER: Output sistem DSS — bukan keputusan hukum.",
            "Verifikasi oleh pemeriksa merek DJKI tetap diperlukan."
        ]

        st.download_button(
            label="📥 Download Laporan (.txt)",
            data="\n".join(lines),
            file_name=f"analisis_{Path(uploaded.name).stem}.txt",
            mime="text/plain",
            use_container_width=True)

    elif not uploaded:
        st.markdown(
            """
            """, unsafe_allow_html=True)

    st.divider()
    st.caption(
        "Ghania Fazila (41122100060) | Sistem Informasi | Universitas Widyatama | "
        "Pembimbing: Murnawan S.T., M.T., MOS.")


if __name__ == "__main__":
    main()
