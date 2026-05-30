"""A script that runs face detection and recognition on camera frames using OpenCV and a known people database (see enroll.py). """

import argparse
import cv2
import json
import os
import time
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    models_dir = 'models'
    detector_model = 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'
    embedder_model = 'mobilefacenet_edgetpu.tflite'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_model', help='.tflite detector model path',
                        default=os.path.join(models_dir,detector_model))
    parser.add_argument('--embedder_model', help='.tflite embedder model path',
                        default=os.path.join(models_dir,embedder_model))
    parser.add_argument('--top_k', type=int, default=3,
                        help='maximum number of faces to display')
    
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    
    parser.add_argument('--threshold', type=float, default=0.1,                        
                        help='classifier score threshold')
    
    parser.add_argument('--db_path', default='data/face_db.json',
                        help='Path to face database JSON file')
    
    parser.add_argument('--recognition_threshold', type=float, default=1.00,
                        help='Optional override for recognition Euclidean threshold')
    
    parser.add_argument('--embed_margin_ratio', type=float, default=0.10,
                        help='Extra bbox margin ratio used before embedding crop')
    
    parser.add_argument('--recognize_every_n_frames', type=int, default=3,
                        help='Run recognition once every N frames')
    
    parser.add_argument('--box_smoothing_alpha', type=float, default=0.60,
                        help='Smoothing factor for bbox coordinates (0..1)')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Start the display in fullscreen mode')
    args = parser.parse_args()
    

    # Load Databse
    with open(args.db_path, 'r', encoding='utf-8') as f:
        face_db = json.load(f)

    embedder_meta = face_db.get('embedder', {})
    db_identities = face_db.get('identities', [])
    embedding_dim = int(embedder_meta.get('embedding_dim', 0))
    if embedding_dim <= 0:
        raise ValueError('Invalid or missing embedder.embedding_dim in DB')
    if not isinstance(db_identities, list) or not db_identities:
        raise ValueError('DB has no identities. Run enroll.py first.')

    identity_names = []
    identity_vectors = []
    for rec in db_identities:
        name = rec.get('name')
        mean_embedding = rec.get('mean_embedding')
        if not name or not isinstance(mean_embedding, list):
            continue
        if len(mean_embedding) != embedding_dim:
            raise ValueError(f'Identity "{name}" has embedding size {len(mean_embedding)} != {embedding_dim}')
        identity_names.append(name)
        identity_vectors.append(mean_embedding)

    if not identity_names:
        raise ValueError('No valid identities found in DB')

    identity_matrix = np.asarray(identity_vectors, dtype=np.float32)
    norms = np.linalg.norm(identity_matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    identity_matrix = identity_matrix / norms

    db_threshold = face_db.get('thresholds', {}).get('recognition_distance_max')
    recognition_threshold = args.recognition_threshold if args.recognition_threshold is not None else db_threshold
    if recognition_threshold is None:
        recognition_threshold = 0.9
        print('Warning: No recognition threshold set. Falling back to 0.9')
    recognition_threshold = float(recognition_threshold)

    print(f'Loaded {len(identity_names)} identities from {args.db_path} (dim={embedding_dim}, thr={recognition_threshold})')


    # Load Models and allocate tensors
    print('Loading detector {} '.format(args.detector_model))
    det_interpreter = make_interpreter(args.detector_model)
    det_interpreter.allocate_tensors()
    det_input_size = input_size(det_interpreter)

    print('Loading embedder {} '.format(args.embedder_model))
    emb_interpreter = make_interpreter(args.embedder_model)
    emb_interpreter.allocate_tensors()
    emb_input_size = input_size(emb_interpreter)

    # Open camera window 
    cap = cv2.VideoCapture(args.camera_idx)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    frame_count = 0
    last_display_labels = []
    last_display_boxes = []
    smoothed_fps = None

    while cap.isOpened():
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        frame_count += 1
        do_recognition = (frame_count % args.recognize_every_n_frames == 0) or not last_display_labels

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, det_input_size)
        run_inference(det_interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(det_interpreter, args.threshold)[:args.top_k]

        display_boxes = []
        display_labels = []
        display_scores = []
        height, width, _ = cv2_im.shape
        scale_x, scale_y = width / det_input_size[0], height / det_input_size[1]

        for idx, obj in enumerate(objs):
            bbox = obj.bbox.scale(scale_x, scale_y)
            current_box = bbox_to_tuple(bbox, width, height, args.embed_margin_ratio)
            if idx < len(last_display_boxes):
                current_box = smooth_box(last_display_boxes[idx], current_box, args.box_smoothing_alpha)
            display_boxes.append(current_box)
            display_scores.append(obj.score)

            if do_recognition:
                face_embedding = extract_face_embedding_from_box(
                    cv2_im,
                    current_box,
                    emb_interpreter,
                    emb_input_size,
                )

                if face_embedding is not None:
                    best_name, best_distance = match_identity(face_embedding, identity_names, identity_matrix)
                    if best_distance <= recognition_threshold:
                        display_labels.append('{} {:.2f}'.format(best_name, best_distance))
                    else:
                        display_labels.append('Unknown {:.2f}'.format(best_distance))
                else:
                    display_labels.append('Unknown')
            elif idx < len(last_display_labels):
                display_labels.append(last_display_labels[idx])
            else:
                display_labels.append('Face')

        last_display_boxes = display_boxes
        last_display_labels = display_labels

        cv2_im = append_objs_to_img(cv2_im, display_boxes, display_labels, display_scores)
        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        if smoothed_fps is None:
            smoothed_fps = fps
        else:
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * fps
        cv2_im = append_perf_overlay(cv2_im, smoothed_fps, frame_ms)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, display_boxes, display_labels, display_scores):
    frame_h, _frame_w = cv2_im.shape[:2]
    for idx, box in enumerate(display_boxes):
        x0, y0, x1, y1 = box

        percent = int(100 * display_scores[idx]) if idx < len(display_scores) else 100
        identity_text = display_labels[idx] if idx < len(display_labels) else 'Face'
        percent_text = '{}%'.format(percent)
        box_color = (0, 255, 0) if not identity_text.startswith('Unknown') else (0, 0, 255)
        top_text_y = max(20, y0 - 10)
        bottom_text_y = min(frame_h - 10, y1 + 25)

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), box_color, 2)
        cv2_im = cv2.putText(cv2_im, identity_text, (x0, top_text_y),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2_im = cv2.putText(cv2_im, percent_text, (x0, bottom_text_y),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


def append_perf_overlay(cv2_im, fps, frame_ms):
    text = 'FPS: {:.1f}  MS: {:.1f}'.format(fps, frame_ms)
    text_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
    frame_h, frame_w = cv2_im.shape[:2]
    x = max(0, frame_w - text_w - 12)
    y = max(text_h + 8, 8 + text_h)

    cv2.putText(cv2_im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness)

    title_line = 'Face recognition on the TPU'
    credit_line = 'Created by MSc Students: N. Chorianopoulos, S. Mandalos, N. Pandi - 2026'
    static_scale = text_scale
    credit_scale = static_scale
    static_thickness = 1
    (_title_w, _title_h), _title_base = cv2.getTextSize(
        title_line, cv2.FONT_HERSHEY_SIMPLEX, static_scale, static_thickness
    )
    (_credit_w, _credit_h), _credit_base = cv2.getTextSize(
        credit_line, cv2.FONT_HERSHEY_SIMPLEX, credit_scale, static_thickness
    )
    static_x = 10
    title_y = 16
    credit_y = frame_h - 10
    cv2.putText(
        cv2_im,
        title_line,
        (static_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        static_scale,
            (220, 220, 220),
        static_thickness,
    )
    cv2.putText(
        cv2_im,
        credit_line,
        (static_x, credit_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        credit_scale,
        (220, 220, 220),
        static_thickness,
    )
    return cv2_im


def bbox_to_tuple(bbox, width, height, margin_ratio):
    x0 = max(0, int(bbox.xmin))
    y0 = max(0, int(bbox.ymin))
    x1 = min(width, int(bbox.xmax))
    y1 = min(height, int(bbox.ymax))

    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)
    x0 = max(0, x0 - margin_x)
    y0 = max(0, y0 - margin_y)
    x1 = min(width, x1 + margin_x)
    y1 = min(height, y1 + margin_y)
    return x0, y0, x1, y1


def smooth_box(previous_box, current_box, alpha):
    if previous_box is None:
        return current_box
    return tuple(int(round(alpha * current + (1.0 - alpha) * previous))
                 for previous, current in zip(previous_box, current_box))


def extract_face_embedding_from_box(cv2_im, box, emb_interpreter, emb_input_size):
    x0, y0, x1, y1 = box

    face = cv2_im[y0:y1, x0:x1]
    if face.size == 0:
        return None

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, emb_input_size)
    run_inference(emb_interpreter, face_rgb.tobytes())

    out_detail = emb_interpreter.get_output_details()[0]
    raw = emb_interpreter.get_tensor(out_detail['index'])[0].reshape(-1)

    scale, zero_point = out_detail.get('quantization', (0.0, 0))
    if scale and scale > 0:
        vec = (raw.astype(np.float32) - float(zero_point)) * float(scale)
    else:
        vec = raw.astype(np.float32)

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def match_identity(face_embedding, identity_names, identity_matrix):
    diff = identity_matrix - face_embedding
    distances = np.linalg.norm(diff, axis=1)
    best_idx = int(np.argmin(distances))
    return identity_names[best_idx], float(distances[best_idx])

if __name__ == '__main__':
    main()
