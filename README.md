# Face Recognition and Detection on the Google Coral Dev Board

This program runs face detection and face recognition on a Google Coral Dev Board with an Edge TPU, written in Python. It uses the coral camera for live video at 30 fps and two pretrained models, found in `models/`:

- [SSD MobileNetv2-SSD:](https://github.com/saunack/MobileNetv2-SSD) A face detector returning bounding box coordinates
- MobileFaceNet: An embedder for turning each detected face into a 128 dimensional vector

This project was developed collaboratively by myself and two colleagues as part of our Master's program in Electronics and Information Processing at the [University of Patras](https://physics.upatras.gr/en/).

<img width="1754" height="1169" alt="DF272186-5A3B-4EAE-8E45-89D1A3DEBDCC" src="https://github.com/user-attachments/assets/ed0a037c-7adb-4823-845b-80578f7f261a" />

###### _The image was taken from a projector screen, hence the blurines_

The workflow is simple:

1. Clone the repository onto your Coral Dev Board or onto a Linux machine that can copy files to the board.
2. Put enrollment photos into folders under `data/enroll/`.
3. Run `enroll.py` once to build the face database in `data/face_db.json`.
4. Run `detect.py` to start live recognition from the camera.

The database is updated automatically when you rerun `enroll.py`, so you only need to run it again if you add new people, add new photos, or want to rebuild the database.

## Requirements

You need a Coral Dev Board with the Edge TPU software stack available, plus Python dependencies for:

- [OpenCV](https://github.com/opencv/opencv)
- [NumPy](https://github.com/numpy/numpy)
- [PyCoral](https://github.com/google-coral/pycoral) / Edge TPU runtime

The repository already includes the `.tflite` models in `models/`, so you do not need to download them separately.

## Repository Layout

```text
.
├── detect.py
├── enroll.py
├── data/
│   ├── enroll/
│   └── face_db.json
└── models/
    ├── mobilefacenet_edgetpu.tflite
    └── ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite
```

## Setup

### 1. Clone the repository

Copy and run:

```bash
git clone https://github.com/NikolasChorianopoulos/face-recognision-on-the-google-coral-dev-board
cd face-recognision-on-the-google-coral-dev-board
```

### 2. Prepare enrollment folders

Inside `data/enroll/`, we have to create one folder per person you want in the database.

First create the known person folders on your PC, add the pictures inside each folder, and then copy each folder on the coral isnide `data/enroll/` using mdt.

```bash
mdt push /path/to/local/folder /home/mendel/destination_name
```

You can do this from a Linux PC directly. Windows and macOS may also work depending on the terminal you use, but Linux is usually the easiest path.

Example `data/enroll/` structure:

```text
data/enroll/
├── alice/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── bob/
│   ├── 01.jpg
│   └── 02.jpg
└── carol/
    ├── 01.jpg
    └── 02.jpg
```

### 3. Use good enrollment images

For best results, each image should:

- Contain one clear face
- Be reasonably well lit
- Show the face close enough to the camera for the detector to find it reliably
- Avoid heavy blur, strong shadows, and extreme angles
- Only jpeg images tested, other formats may also work fine

Square images are helpful, but they are not required. The scripts crop and resize the face automatically before embedding.

If an image has no detectable face, more than one face, or a face crop that is too small, the enrollment script will skip it.

### 4. Build the database

Run enrollment after cloning the repo and after adding or changing people:

```bash
python3 enroll.py
```

What this does:

- Scans every folder under `data/enroll/`
- Detects faces in each image
- Builds an embedding for each valid face
- Averages the embeddings per person
- Writes the result to `data/face_db.json`

Important notes:

- This script should normally be run once to create the database, and again only if you add new people or new images.
- `data/face_db.json` is updated automatically.
- By default, each identity must have at least 15 valid samples, matching the value stored in `data/face_db.json`.

### 5. Run live recognition

Once the database is built, start the camera recognition script:

```bash
python3 detect.py
```

The script will open the camera, detect faces, and try to match them against the identities stored in `data/face_db.json`.

Press `q` in the camera window to quit if you wave a keyboard connected on the coral, else ctrl+c on the terminal to exit.

## Command-Line Arguments

### detect.py

| Argument                     |                                                         Default | Description                                                                                                                                                                 |
| ---------------------------- | --------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--detector_model`           | [SSD MobileNet V2](https://github.com/NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board/blob/main/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite) | Path to the face detector model.                                                                                                                                            |
| `--embedder_model`           |[MobileFaceNet](https://github.com/NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board/blob/main/models/mobilefacenet_edgetpu.tflite) | Path to the face embedder model.                                                                                                                                            |
| `--top_k`                    |                                                             `3` | Maximum number of detected faces to display and process each frame.                                                                                                         |
| `--camera_idx`               |                                                             `0` | Camera device index to open. Usb camera usually is index 1 and coral index 0. Use `v4l2-ctl --list-devices` to find out. You should see something like this: `/dev/video0`. |
| `--threshold`                |                                                           `0.1` | Minimum detector score required to accept a face detection.                                                                                                                 |
| `--db_path`                  |                                             `data/face_db.json` | Path to the face database JSON file.                                                                                                                                        |
| `--recognition_threshold`    |                                                          `1.00` | Maximum Euclidean distance allowed for a successful identity match. Lower is stricter, higher is more permissive.                                                           |
| `--embed_margin_ratio`       |                                                          `0.10` | Extra margin added around each detected face before embedding.                                                                                                              |
| `--recognize_every_n_frames` |                                                             `3` | Runs recognition only every N frames to reduce load.                                                                                                                        |
| `--box_smoothing_alpha`      |                                                          `0.60` | Smoothing factor for bounding box coordinates. Higher values follow the current box more closely.                                                                           |
| `--fullscreen`               |                                                             off | Starts the display window in fullscreen mode.                                                                                                                               |

Example:

```bash
python3 detect.py --camera_idx 0 --top_k 5 --fullscreen
```

### enroll.py

| Argument              |                                                         Default | Description                                                                            |
| --------------------- | --------------------------------------------------------------: | -------------------------------------------------------------------------------------- |
| `--db_path`           |                                             `data/face_db.json` | Path to the face database JSON file that will be created or updated.                   |
| `--source_dir`        |                                                   `data/enroll` | Enrollment folder containing one subfolder per identity.                               |
| `--detector_model`    | [SSD MobileNet V2](https://github.com/NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board/blob/main/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite) | Path to the face detector model.                                                       |
| `--embedder_model`    | [MobileFaceNet](https://github.com/NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board/blob/main/models/mobilefacenet_edgetpu.tflite) | Path to the face embedder model.                                                       |
| `--det_threshold`     |                                                           `0.5` | Minimum detector score required to accept a face during enrollment.                    |
| `--min_face_size`     |                                                            `40` | Minimum width/height in pixels for a detected face crop to be accepted.                |
| `--margin_ratio`      |                                                          `0.20` | Extra margin added around the detected face crop before embedding.                     |
| `--min_valid_samples` |                                  value from `data/face_db.json` | Minimum number of valid images required per identity.                                  |
| `--prune_missing`     |                                                             off | Remove identities from the database if their folder no longer exists in `data/enroll`. |

Example:

```bash
python3 enroll.py --prune_missing
```

## How Enrollment Works

`enroll.py` is designed to build a database from folders of face images.

For each person folder:

1. The script loads every supported image file in that folder.
2. It runs face detection on the image.
3. The image is accepted only if exactly one face is found.
4. The detected face is cropped, expanded slightly by the margin setting, and resized for the embedder.
5. The face embedding is computed and normalized.
6. All valid embeddings for that person are averaged into a single identity vector.

If the folder does not contain enough valid images, that person is skipped.

## Practical Tips

- Use multiple images per person, not just one.
- Vary the angle slightly, but keep the face clearly visible.
- Avoid very dark, very bright, or blurry photos.
- Keep faces large enough in the frame for the detector to find them.
- If recognition is too permissive or too strict, adjust `--recognition_threshold` in `detect.py`.

## Troubleshooting

- If `enroll.py` says it found zero faces, check that the images are clear, well lit, and contain only one face.
- If a person is skipped, they probably do not have enough valid samples yet.
- If recognition feels unstable, try better lighting, closer framing, or a lower `--recognition_threshold` value.
- The project can still make mistakes because face recognition is sensitive to lighting, pixel quality, blur, camera angle, and how close the face is to the camera.

## Notes

- In`data/face_db.json`, "identities": [] starts as an empty list and is filled by `enroll.py`.
- The detector and embedder models in `models/` are already configured for the Edge TPU.
- The default database currently uses Euclidean distance for recognition.

## Known issues

Mismatched or incorrect recognitions can occur. Common causes include poor or uneven lighting, low-resolution or noisy camera sensors, insufficient or low-quality enrollment images, incorrect database entries, or an inappropriate recognition threshold. To reduce mismatches:

- Improve illumination and reduce backlighting.
- Rebuild the database with more and higher-quality images per identity.
- Lower or raise `--recognition_threshold` in `detect.py` to tune strictness.
- Use a higher-resolution, well-focused camera and ensure faces occupy a reasonable portion of the frame.
- Confirm `data/face_db.json` contains the expected identities and embeddings.

## License

See `LICENSE` for project licensing details.
