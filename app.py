"""
Barcode Scanner — Streamlit App  (Snapshot Camera Edition)
----------------------------------------------------------
KEY CHANGES vs the old version
  ✓ Uses st.camera_input() — Streamlit's built-in camera snapshot.
    No WebRTC, no streaming, no threading, no lag.
  ✓ Photo is processed instantly on the server the moment it's taken.
  ✓ Barcode numbers appear in prominent styled output boxes every time.
  ✓ Green polygons + labels are drawn on the frozen captured photo.
  ✓ Much simpler code — no BarcodeProcessor class, no locks needed.

Requirements
  pip install streamlit opencv-python-headless pyzbar numpy
  Linux also needs: sudo apt-get install libzbar0
  Mac:              brew install zbar
"""

import cv2
import numpy as np
import streamlit as st
from pyzbar.pyzbar import decode

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Barcode Scanner",
    page_icon="📷",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Page heading */
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.15rem;
    }

    /* Individual barcode result card */
    .bc-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #f9fbe7 100%);
        border-left: 6px solid #43a047;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .bc-number {
        font-family: "Courier New", Courier, monospace;
        font-size: 1.35rem;
        font-weight: 800;
        color: #1b5e20;
        letter-spacing: 0.04em;
        word-break: break-all;
    }
    .bc-badge {
        display: inline-block;
        background: #43a047;
        color: #fff;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 20px;
        padding: 2px 10px;
        margin-top: 6px;
        letter-spacing: 0.05em;
    }
    .bc-index {
        font-size: 0.78rem;
        color: #888;
        margin-bottom: 3px;
    }

    /* Warning card — no barcode detected */
    .no-bc {
        background: #fff8e1;
        border-left: 6px solid #ffa000;
        border-radius: 10px;
        padding: 14px 20px;
        color: #e65100;
        font-weight: 600;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Core detection function ────────────────────────────────────────────────
def detect_and_annotate(img_bgr: np.ndarray):
    """
    Run pyzbar on *img_bgr*, draw green polygons + labels for every barcode
    found, and return (annotated_copy, list_of_dicts).

    Each dict has keys 'data' (str) and 'type' (str).
    """
    out = img_bgr.copy()
    found = []

    for obj in decode(img_bgr):
        data = obj.data.decode("utf-8", errors="replace")
        kind = obj.type
        found.append({"data": data, "type": kind})

        # ── Polygon around barcode ─────────────────────────────────────────
        if len(obj.polygon) == 4:
            pts = np.array([(p.x, p.y) for p in obj.polygon], np.int32)
        else:
            x, y, w, h = obj.rect
            pts = np.array(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], np.int32
            )
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 220, 0), 4)

        # ── Label: measure text first so the background box fits exactly ──
        label = f"{data}  [{kind}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.60
        fthick = 2
        (tw, th), bl = cv2.getTextSize(label, font, fscale, fthick)

        x, y, w, h = obj.rect
        # Place label above the barcode; clamp so it never goes off the top
        pad = 6
        lx = x
        ly = max(y - pad, th + bl + pad * 2)  # y of the text baseline

        # Green pill behind text
        cv2.rectangle(
            out,
            (lx - 2, ly - th - bl - pad),
            (lx + tw + pad * 2, ly + pad // 2),
            (0, 200, 0),
            -1,
        )
        # Black text on top
        cv2.putText(
            out,
            label,
            (lx + pad, ly - bl),
            font,
            fscale,
            (0, 0, 0),
            fthick,
            cv2.LINE_AA,
        )

    return out, found


# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">📷 Barcode Scanner</p>', unsafe_allow_html=True)
st.write(
    "Position a barcode in the camera view below, then press **Take photo**. "
    "All barcodes found are highlighted with a **green box** and their numbers "
    "are shown immediately on the right."
)

st.info(
    "💡 **Tip:** Hold your camera steady and make sure the barcode is well-lit "
    "and fills at least ¼ of the frame for best results.",
    icon=None,
)

st.divider()

# st.camera_input shows a live camera preview (no streaming lag) and returns
# the snapshot as bytes the moment the shutter button is tapped.
photo = st.camera_input(
    label="Camera  •  press **Take photo** when the barcode is centred",
    key="camera",
)

if photo is not None:
    # ── Decode bytes → OpenCV BGR array ───────────────────────────────────
    raw = np.frombuffer(photo.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the photo — please try again.")
    else:
        annotated, barcodes = detect_and_annotate(img_bgr)

        st.divider()
        img_col, result_col = st.columns([3, 2], gap="large")

        # ── Left: annotated photo ──────────────────────────────────────────
        with img_col:
            st.subheader("📌 Captured photo")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        # ── Right: barcode numbers ─────────────────────────────────────────
        with result_col:
            st.subheader("🔢 Barcode numbers")

            if barcodes:
                for i, bc in enumerate(barcodes, start=1):
                    st.markdown(
                        f"""
                        <div class="bc-card">
                            <div class="bc-index">Barcode #{i}</div>
                            <div class="bc-number">{bc['data']}</div>
                            <span class="bc-badge">{bc['type']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.divider()

                # ── Table view ─────────────────────────────────────────────
                rows = [
                    {"#": i, "Type": bc["type"], "Barcode Number": bc["data"]}
                    for i, bc in enumerate(barcodes, 1)
                ]
                st.table(rows)

                # ── CSV download ───────────────────────────────────────────
                csv_lines = ["No.,Type,Barcode Number"] + [
                    f"{r['#']},{r['Type']},{r['Barcode Number']}" for r in rows
                ]
                st.download_button(
                    label="⬇️  Download as CSV",
                    data="\n".join(csv_lines),
                    file_name="barcodes.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            else:
                st.markdown(
                    """
                    <div class="no-bc">
                        ⚠️ No barcode detected in this photo.<br><br>
                        <b>Try:</b><br>
                        • Better lighting (avoid glare)<br>
                        • Move closer to the barcode<br>
                        • Hold the camera still before tapping
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
