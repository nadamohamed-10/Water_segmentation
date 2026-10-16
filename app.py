"""Flask REST API + web UI for water segmentation.

    python app.py            # http://127.0.0.1:5000

POST /predict   multipart/form-data, field "file" = 12-band .tif/.tiff (or .npy shaped (12,H,W))
    query/form options:  format=json (default) | png     threshold=0..1     tta=1     include_mask=1
GET  /health    model + status          GET /samples   list of bundled sample tiles
"""
import base64
import io
import os
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from water_seg.inference import DEFAULT_WEIGHTS, InputError, WaterSegmenter, overlay, read_image, rgb_preview

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(_png(arr)).decode("ascii")


def create_app(weights_path=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 64)) * 1024 * 1024
    segmenter = WaterSegmenter(weights_path or os.environ.get("WEIGHTS", DEFAULT_WEIGHTS))
    lock = threading.Lock()  # one forward pass at a time; keeps memory flat

    def error(status, message):
        return jsonify(error=message), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return error(413, f"File is larger than the {app.config['MAX_CONTENT_LENGTH'] // 2**20} MB upload limit.")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", **segmenter.info())

    @app.get("/samples")
    def samples():
        return jsonify(samples=sorted(p.name for p in SAMPLES.glob("*.tif")))

    @app.get("/samples/<path:name>")
    def sample_file(name):
        return send_from_directory(SAMPLES, name, as_attachment=True)

    @app.post("/predict")
    def predict():
        up = request.files.get("file")
        if up is None or not up.filename:
            return error(400, 'No file received. Send multipart/form-data with the image in the field "file".')
        try:
            thr = request.values.get("threshold")
            thr = None if thr in (None, "") else float(thr)
            if thr is not None and not 0.0 <= thr <= 1.0:
                raise ValueError
        except ValueError:
            return error(400, "threshold must be a number between 0 and 1.")
        tta = request.values.get("tta", "0").lower() in ("1", "true", "yes", "on")

        try:
            img = read_image(up.read(), up.filename)
        except InputError as exc:
            return error(422, str(exc))

        with lock:
            res = segmenter.predict(img, thr, tta)
        mask = res["mask"]

        if request.values.get("format", "json").lower() == "png":  # raw binary mask image: 0 = land, 255 = water
            resp = send_file(io.BytesIO(_png(mask * 255)), mimetype="image/png", download_name="mask.png")
            resp.headers["X-Water-Fraction"] = f"{mask.mean():.6f}"
            return resp

        rgb = rgb_preview(img)
        body = dict(
            filename=up.filename, height=int(mask.shape[0]), width=int(mask.shape[1]),
            threshold=res["threshold"], tta=tta,
            water_pixels=int(mask.sum()), water_fraction=float(mask.mean()),
            inference_ms=round(res["ms"], 1),
            mask_png_base64=_b64(mask * 255),          # binary mask: 0 / 255
            overlay_png_base64=_b64(overlay(rgb, mask)),
            rgb_png_base64=_b64(rgb),
            probability_png_base64=_b64((res["prob"] * 255).astype(np.uint8)),
        )
        if request.values.get("include_mask", "0").lower() in ("1", "true", "yes"):
            body["mask"] = mask.tolist()               # 0/1 nested list, rows x cols
        return jsonify(body)

    return app


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
