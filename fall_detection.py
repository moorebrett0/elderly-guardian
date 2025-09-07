#!/usr/bin/env python3
import sys, argparse, logging, os, time, threading, asyncio
os.environ.setdefault("AXELERA_FRAMEWORK", "/home/aetina/voyager-sdk")

from axelera.app.stream import create_inference_stream
from axelera.app.utils import load_yamlfile
import numpy as np
import cv2

# Optional Kasa control
try:
    from kasa import SmartPlug  # pip install python-kasa
except Exception:
    SmartPlug = None

COCO_EDGES = [
    (5,7), (7,9), (6,8), (8,10), (5,6), (5,11), (6,12), (11,12),
    (11,13), (13,15), (12,14), (14,16), (0,5), (0,6), (0,1), (0,2), (1,3), (2,4)
]

def can_show_windows():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False, "DISPLAY is not set (headless session)."
    try:
        info = cv2.getBuildInformation()
        if ("GTK" in info) or ("QT" in info) or ("WIN32" in info) or ("COCOA" in info):
            try:
                cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
                dummy = np.zeros((2,2,3), dtype=np.uint8)
                cv2.imshow("__probe__", dummy); cv2.waitKey(1); cv2.destroyWindow("__probe__")
                return True, "GUI backend available."
            except Exception as e:
                return False, f"OpenCV GUI failed: {e}"
        return False, "OpenCV built without a GUI backend."
    except Exception as e:
        return False, f"Could not inspect OpenCV build: {e}"

def to_bgr_np(image_obj, first_frame_debug=False):
    """
    Convert Voyager Image -> BGR uint8 np.ndarray.
    Prefers Image.asarray()/aspil(); falls back to raw bytes (NV12).
    Returns None if conversion fails.
    """
    arr = None
    if first_frame_debug:
        logging.info("image_obj type: %s", type(image_obj).__name__)
        try:
            attrs = [a for a in dir(image_obj) if not a.startswith("_")]
            logging.info("image_obj attrs (skim): %s", attrs[:60])
            for name in ("width", "height", "color_format", "pitch", "pixel_stride"):
                if hasattr(image_obj, name):
                    logging.info("image_obj.%s = %r", name, getattr(image_obj, name))
        except Exception:
            pass

    if isinstance(image_obj, np.ndarray):
        arr = image_obj

    if arr is None and hasattr(image_obj, "asarray"):
        try: arr = image_obj.asarray()
        except Exception: arr = None

    if arr is None and hasattr(image_obj, "aspil"):
        try:
            pil = image_obj.aspil()
            if pil is not None: arr = np.asarray(pil)
        except Exception:
            arr = None

    if arr is None:
        try: arr = np.array(image_obj)
        except Exception: arr = None

    if isinstance(arr, np.ndarray) and arr.size > 0:
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            fmt = getattr(image_obj, "color_format", "RGB")
            if isinstance(fmt, str) and fmt.upper().startswith("BGR"):
                return arr.copy()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    # Raw bytes / NV12 fallback
    try:
        width = getattr(image_obj, "width", None)
        height = getattr(image_obj, "height", None)
        if width and height:
            buf = None
            if hasattr(image_obj, "tobytes"):
                buf = image_obj.tobytes()
            elif hasattr(image_obj, "data"):
                data_attr = image_obj.data
                buf = data_attr() if callable(data_attr) else data_attr
            if isinstance(buf, (bytes, bytearray, memoryview)):
                buf = bytes(buf)
                gst_fmt = None
                if hasattr(image_obj, "get_gst_format"):
                    try: gst_fmt = image_obj.get_gst_format()
                    except Exception: gst_fmt = None
                expected_nv12 = int(width * height * 3 / 2)
                if gst_fmt == "NV12" or len(buf) == expected_nv12:
                    y_size = width * height
                    y = np.frombuffer(buf[:y_size], dtype=np.uint8).reshape((height, width))
                    uv = np.frombuffer(buf[y_size:y_size + (y_size // 2)], dtype=np.uint8).reshape((height // 2, width))
                    return cv2.cvtColorTwoPlane(y, uv, cv2.COLOR_YUV2BGR_NV12)
                # Packed 3/4 channel fallback
                px = int(getattr(image_obj, "pixel_stride", 0) or 0)
                if px in (3, 4) and len(buf) >= width * height * px:
                    arr = np.frombuffer(buf[:width * height * px], dtype=np.uint8).reshape((height, width, px))
                    if px == 3:
                        fmt = getattr(image_obj, "color_format", "RGB")
                        if isinstance(fmt, str) and fmt.upper().startswith("BGR"):
                            return arr.copy()
                        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    else:
                        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    except Exception as e:
        if first_frame_debug:
            logging.debug("Raw/NV12 fallback failed: %s", e)

    return None

def is_fallen(box, kps):
    # conservative fall heuristic using bbox aspect + vertical spread of keypoints
    x, y, w, h = box
    if h <= 1e-3 or w <= 1e-3:
        return False
    aspect = float(w) / float(h)
    ys = kps[:, 1]; vis = kps[:, 2] > 0.2
    yspread = (ys[vis].max() - ys[vis].min()) if vis.any() else 0.0
    yspread_ratio = yspread / (h + 1e-6)
    return (aspect > 1.4) and (yspread_ratio < 0.45)

def draw_pose_overlay(frame_bgr, boxes, keypoints, fallen_flags):
    h, w = frame_bgr.shape[:2]
    def draw_kp(pt, color, r=3):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_bgr, (x, y), r, color, -1, lineType=cv2.LINE_AA)
    def draw_line(p1, p2, color, thickness=2):
        x1, y1 = int(p1[0]), int(p1[1]); x2, y2 = int(p2[0]), int(p2[1])
        cv2.line(frame_bgr, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    for i in range(boxes.shape[0]):
        x, y, bw, bh = boxes[i]
        fallen = bool(fallen_flags[i])
        tl = (int(x), int(y)); br = (int(x + bw), int(y + bh))
        color = (0, 0, 255) if fallen else (0, 200, 0)
        cv2.rectangle(frame_bgr, tl, br, color, 2, lineType=cv2.LINE_AA)
        cv2.putText(frame_bgr, "FALL" if fallen else "OK", (tl[0], max(0, tl[1]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        kps = keypoints[i]; vis = kps[:, 2] > 0.2
        for j in range(17):
            if vis[j]: draw_kp(kps[j], (255, 255, 255), r=3)
        for (a, b) in COCO_EDGES:
            if vis[a] and vis[b]: draw_line(kps[a], kps[b], (255, 255, 255), thickness=2)

def trigger_kasa_async(ip: str, on_seconds: int):
    """Pulse a Kasa plug ON then OFF without blocking the main loop."""
    if not ip:
        return
    if SmartPlug is None:
        logging.error("python-kasa not installed. Run: pip install python-kasa")
        return

    async def _pulse():
        try:
            plug = SmartPlug(ip)
            await plug.update()
            await plug.turn_on()
            logging.info("Kasa plug %s -> ON", ip)
            await asyncio.sleep(max(1, on_seconds))
            await plug.turn_off()
            logging.info("Kasa plug %s -> OFF", ip)
        except Exception as e:
            logging.error("Kasa control failed: %s", e)

    threading.Thread(target=lambda: asyncio.run(_pulse()), daemon=True).start()

def main():
    ap = argparse.ArgumentParser(description="Fall detection using YOLOv8 Pose + OpenCV overlay + Kasa trigger")
    ap.add_argument("app_config_yaml", help="Path to the application config YAML file")
    ap.add_argument("--log", default="INFO")
    ap.add_argument("--pipe", default="gst", choices=["gst","torch","torch-aipu"], help="Pipeline type")
    ap.add_argument("--display", action="store_true", help="Show a live window (requires GUI)")
    ap.add_argument("--out", default="", help="Optional output video path, e.g. /home/aetina/fall_demo.mp4")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")

    # human-size filter knobs
    ap.add_argument("--min-area", type=int, default=60000, help="Min bbox area (pixels^2)")
    ap.add_argument("--max-area", type=int, default=300000, help="Max bbox area (pixels^2)")

    # pose quality knobs
    ap.add_argument("--min-conf", type=float, default=0.60, help="Min pose confidence")
    ap.add_argument("--min-kp-conf", type=float, default=0.35, help="Min per-keypoint confidence")
    ap.add_argument("--min-kp-visible", type=int, default=6, help="Min visible keypoints required")

    # Kasa trigger options
    ap.add_argument("--kasa-ip", default="", help="Kasa plug IP to pulse when a fall is detected")
    ap.add_argument("--kasa-on-seconds", type=int, default=5, help="How long to keep the plug ON")
    ap.add_argument("--kasa-cooldown", type=int, default=30, help="Seconds to wait before retriggering")
    ap.add_argument("--fall-frames", type=int, default=5, help="Consecutive fallen frames required to trigger")

    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")

    cfg = (load_yamlfile(args.app_config_yaml) or {}).get("config", {})
    data_root = cfg.get("data_root"); pipeline_path = cfg.get("pipeline_path"); source = cfg.get("source")
    if not all([data_root, pipeline_path, source]):
        logging.error("Missing required fields in app-config.yaml (data_root, pipeline_path, source)"); sys.exit(1)
    if not os.path.isfile(pipeline_path):
        logging.error("Pipeline YAML not found: %s", pipeline_path); sys.exit(1)

    logging.info("AXELERA_FRAMEWORK=%s", os.environ.get("AXELERA_FRAMEWORK"))
    logging.info("Using data_root=%s", data_root)
    logging.info("Using pipeline_path=%s", pipeline_path)
    logging.info("Using source=%s", source)

    if args.display:
        ok, reason = can_show_windows()
        if not ok:
            logging.warning("Disabling --display: %s", reason)
            args.display = False
            if not args.out:
                args.out = "/home/aetina/fall_demo.mp4"
                logging.info("Saving annotated video to %s", args.out)

    stream = create_inference_stream(network=pipeline_path, sources=[source], pipe_type=args.pipe, data_root=data_root)
    logging.info("InferenceStream started. Ctrl+C to stop.")

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frames_seen = frames_valid = frames_written = 0
    first_frame = True

    # fall trigger smoothing + cooldown
    fall_streak = 0
    last_trigger_ts = 0.0

    try:
        for fr in stream:
            img = getattr(fr, "image", None)
            meta = getattr(fr, "meta", None)

            frame_bgr = to_bgr_np(img, first_frame_debug=first_frame)
            frames_seen += 1

            if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
                if first_frame:
                    logging.warning("First frame could not be converted to BGR. "
                                    "If this persists, try a file source or a different USB device.")
                first_frame = False
                continue

            first_frame = False
            frames_valid += 1
            H, W = frame_bgr.shape[:2]

            if writer is None and args.out:
                writer = cv2.VideoWriter(args.out, fourcc, 30.0, (W, H))
                if not writer.isOpened():
                    logging.error("Failed to open video writer at %s", args.out); writer = None
                else:
                    logging.info("Writing annotated video to %s", args.out)

            # ---------- Pose extraction ----------
            pose_meta = meta.get("yolov8lpose-coco") if (meta is not None and hasattr(meta, "get")) else None
            if pose_meta is not None:
                boxes  = getattr(pose_meta, "boxes", None)        # (N,4) xywh
                kpts   = getattr(pose_meta, "keypoints", None)    # (N,17,3)
                scores = getattr(pose_meta, "scores", None)       # (N,)
            else:
                boxes = kpts = scores = None

            drew = False
            fallen_now = False

            if isinstance(boxes, np.ndarray) and isinstance(kpts, np.ndarray) and kpts.shape[1] == 17:
                N = boxes.shape[0]

                # Confidence & keypoint quality gates
                if isinstance(scores, np.ndarray) and scores.shape[0] == N:
                    conf_keep = (scores >= args.min_conf)
                else:
                    conf_keep = np.ones((N,), dtype=bool)

                vis = (kpts[:, :, 2] >= args.min_kp_conf)     # (N,17)
                vis_counts = vis.sum(axis=1)                  # (N,)
                vis_keep = (vis_counts >= args.min_kp_visible)

                # Human-size bbox area filter
                areas = boxes[:, 2] * boxes[:, 3]             # w*h
                size_keep = (areas >= args.min_area) & (areas <= args.max_area)

                keep = conf_keep & vis_keep & size_keep
                boxes  = boxes[keep]
                kpts   = kpts[keep]
                if isinstance(scores, np.ndarray) and scores.shape[0] == N:
                    scores = scores[keep]

                Nf = boxes.shape[0]
                if Nf > 0:
                    fallen_flags = [is_fallen(boxes[i], kpts[i]) for i in range(Nf)]
                    fallen_now = any(fallen_flags)
                    draw_pose_overlay(frame_bgr, boxes, kpts, fallen_flags)
                    cv2.putText(frame_bgr, f"Persons: {Nf}", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    drew = True

            if not drew:
                cv2.putText(frame_bgr, "No poses", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # ---------- Fall → Kasa trigger logic ----------
            if fallen_now:
                fall_streak += 1
            else:
                fall_streak = 0

            if args.kasa_ip:
                if fall_streak >= max(1, args.fall_frames):
                    now = time.time()
                    if (now - last_trigger_ts) >= max(1, args.kasa_cooldown):
                        last_trigger_ts = now
                        logging.info("FALL condition met (%d frames). Pulsing Kasa plug %s for %ds.",
                                     fall_streak, args.kasa_ip, args.kasa_on_seconds)
                        trigger_kasa_async(args.kasa_ip, args.kasa_on_seconds)
                        # prevent immediate retrigger on consecutive frames
                        fall_streak = 0

            # ---------- Output ----------
            if args.display:
                cv2.imshow("Fall Detection (YOLOv8 Pose)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer is not None:
                writer.write(frame_bgr); frames_written += 1

            if args.max_frames and frames_seen >= args.max_frames:
                break

    finally:
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        logging.info("Frames: seen=%d, valid=%d, written=%d", frames_seen, frames_valid, frames_written)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping application.")

