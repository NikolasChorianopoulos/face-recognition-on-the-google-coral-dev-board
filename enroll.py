""" Script to create faces database (final database is on data/face_db.json). 
    The "identities": [] should be empty at first and will fill up with vectors.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from typing import List

import cv2
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_identity_dirs(source_dir: str) -> List[str]:
    if not os.path.isdir(source_dir):
        return []
    names = []
    for entry in os.listdir(source_dir):
        path = os.path.join(source_dir, entry)
        if os.path.isdir(path) and not entry.startswith("."):
            names.append(entry)
    return sorted(names, key=str.lower)


def list_images(identity_dir: str) -> List[str]:
    images = []
    for entry in os.listdir(identity_dir):
        full_path = os.path.join(identity_dir, entry)
        ext = os.path.splitext(entry)[1].lower()
        if os.path.isfile(full_path) and ext in VALID_EXTS:
            images.append(full_path)
    return sorted(images)


def map_bbox_to_original(obj, frame_shape, det_input_size):
    height, width = frame_shape[:2]
    scale_x = width / det_input_size[0]
    scale_y = height / det_input_size[1]
    bbox = obj.bbox.scale(scale_x, scale_y)
    return int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)


def expand_and_clip_bbox(x0, y0, x1, y1, width, height, margin_ratio):
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)
    x0 = max(0, x0 - margin_x)
    y0 = max(0, y0 - margin_y)
    x1 = min(width, x1 + margin_x)
    y1 = min(height, y1 + margin_y)
    return x0, y0, x1, y1


def extract_embedding_from_face(face_bgr, emb_interpreter, emb_input_size):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, emb_input_size)
    run_inference(emb_interpreter, face_rgb.tobytes())

    out_detail = emb_interpreter.get_output_details()[0]
    raw = emb_interpreter.get_tensor(out_detail["index"])[0]
    raw = raw.reshape(-1)

    scale, zero_point = out_detail.get("quantization", (0.0, 0))
    if scale and scale > 0:
        vec = (raw.astype(np.float32) - float(zero_point)) * float(scale)
    else:
        vec = raw.astype(np.float32)
    return l2_normalize(vec)


def extract_embedding_from_image(
    image_path,
    det_interpreter,
    det_input_size,
    emb_interpreter,
    emb_input_size,
    det_threshold,
    margin_ratio,
    min_face_size,
):
    image = cv2.imread(image_path)
    if image is None:
        return None, "unreadable image"

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    det_input = cv2.resize(rgb, det_input_size)
    run_inference(det_interpreter, det_input.tobytes())
    objs = get_objects(det_interpreter, det_threshold)

    if len(objs) != 1:
        return None, f"expected 1 face, found {len(objs)}"

    x0, y0, x1, y1 = map_bbox_to_original(objs[0], image.shape, det_input_size)
    x0, y0, x1, y1 = expand_and_clip_bbox(x0, y0, x1, y1, image.shape[1], image.shape[0], margin_ratio)

    if (x1 - x0) < min_face_size or (y1 - y0) < min_face_size:
        return None, "face too small"

    face = image[y0:y1, x0:x1]
    if face.size == 0:
        return None, "empty face crop"

    embedding = extract_embedding_from_face(face, emb_interpreter, emb_input_size)
    return embedding, None


def compute_mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    stacked = np.stack(embeddings, axis=0)
    mean_vec = np.mean(stacked, axis=0)
    return l2_normalize(mean_vec)


def build_parser():
    parser = argparse.ArgumentParser(description="Build/update face database from enrollment folders.")
    parser.add_argument(
        "--db_path",
        default="data/face_db.json",
        help="Path to face database JSON file.",
    )
    parser.add_argument(
        "--source_dir",
        default=None,
        help="Enrollment source directory. Overrides DB setting when provided.",
    )
    parser.add_argument(
        "--detector_model",
        default="models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
        help="Detector model path.",
    )
    parser.add_argument(
        "--embedder_model",
        default="models/mobilefacenet_edgetpu.tflite",
        help="Embedder model path.",
    )
    parser.add_argument(
        "--det_threshold",
        type=float,
        default=0.5,
        help="Minimum detector score to accept a face for enrollment.",
    )
    parser.add_argument(
        "--min_face_size",
        type=int,
        default=40,
        help="Reject crops smaller than this pixel width/height.",
    )
    parser.add_argument(
        "--margin_ratio",
        type=float,
        default=0.20,
        help="Extra bbox margin ratio before cropping face.",
    )
    parser.add_argument(
        "--min_valid_samples",
        type=int,
        default=None,
        help="Minimum valid samples per identity. Defaults to DB setting.",
    )
    parser.add_argument(
        "--prune_missing",
        action="store_true",
        help="Remove identities from DB if no corresponding folder exists in source_dir.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if not os.path.isfile(args.db_path):
        raise FileNotFoundError(f"DB file not found: {args.db_path}")

    db = read_json(args.db_path)
    source_dir = args.source_dir or db.get("enrollment", {}).get("source_dir", "data/enroll")
    min_valid_samples = args.min_valid_samples
    if min_valid_samples is None:
        min_valid_samples = int(db.get("enrollment", {}).get("min_valid_samples_per_identity", 8))

    update_policy = db.get("enrollment", {}).get("update_policy", "overwrite")
    if update_policy != "overwrite":
        print(f"Warning: unsupported update_policy '{update_policy}', using overwrite.")

    print(f"Loading detector: {args.detector_model}")
    det_interpreter = make_interpreter(args.detector_model)
    det_interpreter.allocate_tensors()
    det_input_size = input_size(det_interpreter)

    print(f"Loading embedder: {args.embedder_model}")
    emb_interpreter = make_interpreter(args.embedder_model)
    emb_interpreter.allocate_tensors()
    emb_input_size = input_size(emb_interpreter)
    emb_out_shape = emb_interpreter.get_output_details()[0]["shape"]
    model_embedding_dim = int(np.prod(emb_out_shape[1:]))

    identity_names = list_identity_dirs(source_dir)
    if not identity_names:
        print(f"No identity folders found in: {source_dir}")
        return

    existing = {rec.get("name"): rec for rec in db.get("identities", []) if rec.get("name")}
    built = 0
    skipped = 0

    for name in identity_names:
        identity_dir = os.path.join(source_dir, name)
        image_paths = list_images(identity_dir)
        if not image_paths:
            print(f"[SKIP] {name}: no images found")
            skipped += 1
            continue

        valid_embeddings = []
        rejected = 0

        for image_path in image_paths:
            embedding, _reason = extract_embedding_from_image(
                image_path=image_path,
                det_interpreter=det_interpreter,
                det_input_size=det_input_size,
                emb_interpreter=emb_interpreter,
                emb_input_size=emb_input_size,
                det_threshold=args.det_threshold,
                margin_ratio=args.margin_ratio,
                min_face_size=args.min_face_size,
            )
            if embedding is None:
                rejected += 1
                continue
            valid_embeddings.append(embedding)

        if len(valid_embeddings) < min_valid_samples:
            print(
                f"[SKIP] {name}: valid={len(valid_embeddings)} rejected={rejected} "
                f"(need at least {min_valid_samples})"
            )
            skipped += 1
            continue

        mean_embedding = compute_mean_embedding(valid_embeddings)

        existing[name] = {
            "name": name,
            "num_samples": len(valid_embeddings),
            "mean_embedding": [float(x) for x in mean_embedding.tolist()],
            "source_folder": os.path.join(source_dir, name),
            "updated_at": now_utc_iso(),
        }
        print(f"[OK] {name}: valid={len(valid_embeddings)} rejected={rejected}")
        built += 1

    if args.prune_missing:
        keep_names = set(identity_names)
        existing = {k: v for k, v in existing.items() if k in keep_names}

    db.setdefault("embedder", {})
    db["embedder"]["model_path"] = args.embedder_model
    db["embedder"]["embedding_dim"] = model_embedding_dim
    db["embedder"]["input_shape"] = [1, int(emb_input_size[1]), int(emb_input_size[0]), 3]
    db["embedder"]["output_shape"] = [1, int(db["embedder"]["embedding_dim"])]
    db["embedder"]["distance_metric"] = "euclidean"
    db["embedder"]["l2_normalized"] = True

    db.setdefault("enrollment", {})
    db["enrollment"]["source_dir"] = source_dir
    db["enrollment"]["update_policy"] = "overwrite"
    db["enrollment"]["min_valid_samples_per_identity"] = int(min_valid_samples)

    db["identities"] = sorted(existing.values(), key=lambda rec: rec["name"].lower())
    write_json(args.db_path, db)

    print(f"Done. Built/updated: {built}, skipped: {skipped}, total identities in DB: {len(db['identities'])}")


if __name__ == "__main__":
    main()
