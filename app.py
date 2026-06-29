"""
Live Barcode Scanner — Streamlit App
-------------------------------------
- Opens your webcam right in the browser.
- Every barcode (or QR code) visible in the live feed gets a green box +
  label drawn around it in real time.
- Once everything you want is nicely boxed, click "Capture Photo" to lock
  in that frame. The decoded values are then listed in a table below
  (type + number), with a CSV download option.
"""

import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from pyzbar.pyzbar import decode
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

st.set_page_config(page_title="Live Barcode Scanner", page_icon="📷", layout="wide")

st.title("📷 Live Barcode Scanner")
st.write(
    "Point your camera at one or more barcodes. Every barcode found is "
    "outlined with a **green box**. Once everything you want is boxed, "
    "hit **Capture Photo**."
)

# ---------------------------------------------------------------------------
# Shared state between the video-processing thread (recv, runs continuously
# in the background) and the main Streamlit thread. Protected by a lock
# since both can read/write at the same time.
# ---------------------------------------------------------------------------
lock = threading.Lock()
shared_state = {"frame": None, "barcodes": []}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class BarcodeProcessor(VideoProcessorBase):
    """Finds every barcode in each frame and draws a green box around it."""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        decoded_objects = decode(img)

        detected = []
        for obj in decoded_objects:
            # The 4-point polygon hugs rotated/skewed barcodes better than
            # the plain bounding rect, so prefer it when available.
            if len(obj.polygon) == 4:
                pts = np.array([(p.x, p.y) for p in obj.polygon], dtype=np.int32)
            else:
                x, y, w, h = obj.rect
                pts = np.array(
                    [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype=np.int32
                )
            pts = pts.reshape((-1, 1, 2))

            # Green box around the barcode
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

            barcode_data = obj.data.decode("utf-8", errors="replace")
            barcode_type = obj.type
            label = f"{barcode_data} ({barcode_type})"

            x, y, w, h = obj.rect
            label_y = max(0, y - 25)
            label_w = max(w, 9 * len(label))
            cv2.rectangle(img, (x, label_y), (x + label_w, y), (0, 255, 0), -1)
            cv2.putText(
                img, label, (x + 2, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

            detected.append({"data": barcode_data, "type": barcode_type})

        with lock:
            shared_state["frame"] = img.copy()
            shared_state["barcodes"] = detected

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Session state for the captured (locked-in) results
# ---------------------------------------------------------------------------
if "captured_barcodes" not in st.session_state:
    st.session_state.captured_barcodes = []
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

# Make the capture button feel like a camera shutter sitting right under
# the live frame, rather than just another sidebar widget.
st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button[kind="primary"] {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 0.7rem 0;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

camera_choice = st.radio("Camera", ["Back camera", "Front camera"], horizontal=True)
facing_mode = "environment" if camera_choice == "Back camera" else "user"

# The key changes with facing_mode so switching the radio button forces a
# clean remount -> a fresh getUserMedia call with the new constraint, which
# is more reliable than hoping the browser hot-swaps the camera.
ctx = webrtc_streamer(
    key=f"barcode-scanner-{facing_mode}",
    video_processor_factory=BarcodeProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "facingMode": facing_mode,
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False,
    },
)

# Shutter button sits directly under the live frame — tap it once every
# barcode you want is boxed in green.
btn_col, reset_col = st.columns([4, 1])
with btn_col:
    capture_clicked = st.button(
        "📸  Tap to capture", use_container_width=True, type="primary"
    )
with reset_col:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

status_box = st.empty()

if reset_clicked:
    st.session_state.captured_barcodes = []
    st.session_state.captured_image = None

if capture_clicked:
    with lock:
        current_barcodes = list(shared_state["barcodes"])
        current_frame = (
            shared_state["frame"].copy() if shared_state["frame"] is not None else None
        )
    if not current_barcodes:
        st.warning(
            "No barcode is in view right now — hold steady over a barcode "
            "until it's boxed in green, then try again."
        )
    else:
        st.session_state.captured_barcodes = current_barcodes
        st.session_state.captured_image = current_frame
        st.success(f"Captured {len(current_barcodes)} barcode(s)!")

st.divider()

# Captured photo and its barcode list, shown side by side.
if st.session_state.captured_image is not None or st.session_state.captured_barcodes:
    img_col, list_col = st.columns([1, 1])

    with img_col:
        st.subheader("📌 Captured frame")
        if st.session_state.captured_image is not None:
            st.image(
                cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

    with list_col:
        st.subheader("✅ Barcode numbers")
        if st.session_state.captured_barcodes:
            rows = [
                {"No.": i + 1, "Type": b["type"], "Barcode Number": b["data"]}
                for i, b in enumerate(st.session_state.captured_barcodes)
            ]
            st.table(rows)

            csv_lines = ["No.,Type,Barcode Number"] + [
                f'{r["No."]},{r["Type"]},{r["Barcode Number"]}' for r in rows
            ]
            st.download_button(
                "⬇️ Download as CSV",
                data="\n".join(csv_lines),
                file_name="barcodes.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Live-updating status (must stay the LAST thing in the script). While the
# camera is streaming, this loop keeps refreshing the "barcodes currently
# in view" indicator. Any button click / stream stop triggers a Streamlit
# rerun, which naturally breaks out of this loop and re-executes everything
# above with the freshly updated state.
# ---------------------------------------------------------------------------
if ctx.state.playing:
    while True:
        with lock:
            count = len(shared_state["barcodes"])
            values = [b["data"] for b in shared_state["barcodes"]]
        if count > 0:
            status_box.success(f"✅ {count} barcode(s) in view:\n" + "\n".join(values))
        else:
            status_box.info("🔍 Scanning… no barcode in view yet")
        time.sleep(0.3)
