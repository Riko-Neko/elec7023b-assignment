# Name: Li Ziqi
# Assignment Three
# Run: uv run --with tensorflow==2.20.0 --with pillow --python 3.13 assin3_detection.py

import argparse
import io
import json
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

MODEL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/"
    "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
)
LABELS = ("https://raw.githubusercontent.com/tensorflow/models/master/"
          "research/object_detection/data/mscoco_label_map.pbtxt")
SAMPLES = ("https://raw.githubusercontent.com/ultralytics/ultralytics/main/"
           "ultralytics/assets/")
THRESHOLDS = (0.3, 0.7)


def download(url):
    with urlopen(url, timeout=120) as response:
        return response.read()


def load_detector():
    graph_def = tf.compat.v1.GraphDef()
    with tarfile.open(fileobj=io.BytesIO(download(MODEL))) as archive:
        graph_file = next(m for m in archive if m.name.endswith("/frozen_inference_graph.pb"))
        graph_def.ParseFromString(archive.extractfile(graph_file).read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph


def detect(session, image, labels):
    outputs = ["detection_boxes", "detection_scores", "detection_classes", "num_detections"]
    boxes, scores, classes, count = session.run(
        [name + ":0" for name in outputs],
        feed_dict={"image_tensor:0": np.asarray(image)[None]},
    )
    width, height = image.size
    records = []
    for box, score, class_id in zip(boxes[0], scores[0], classes[0]):
        if len(records) >= int(count[0]) or score < min(THRESHOLDS):
            break  # The model returns candidates in descending score order.
        top, left, bottom, right = np.clip(box, 0, 1).astype(float) * [height, width, height, width]
        w, h = right - left, bottom - top
        records.append({
            "ID": len(records) + 1,
            "ClassID": int(class_id), "Class": labels[int(class_id)],
            "Confidence": float(score),
            "Left": left, "Top": top, "Right": right, "Bottom": bottom,
            "Width": w, "Height": h, "Area": w * h,
            "Center": [(left + right) / 2, (top + bottom) / 2],
        })
    return records


def annotate(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=max(17, min(image.size) // 30))
    for d in detections:
        left, top, right, bottom = (d[key] for key in ("Left", "Top", "Right", "Bottom"))
        color = "#0775ad"
        draw.rectangle((left, top, right, bottom), outline=color, width=4)
        caption = f'{d["ID"]}. {d["Class"]} {d["Confidence"]:.2f}'
        text_width = draw.textlength(caption, font=font)
        text_height = font.size + 7
        x = max(0, min(left, image.width - text_width - 10))
        y = max(0, top - text_height)
        draw.rectangle((x, y, x + text_width + 10, y + text_height), fill=color)
        draw.text((x + 5, y + 2), caption, fill="white", font=font)
    return result


def main():
    parser = argparse.ArgumentParser(description="Detect objects at confidence thresholds 0.3 and 0.7.")
    parser.add_argument("images", nargs="*", help="Two image paths (default: bus.jpg and zidane.jpg).")
    parser.add_argument("--output-prefix", help="Save annotated JPEGs with this filename prefix.")
    args = parser.parse_args()
    if args.images and len(args.images) != 2:
        parser.error("Provide exactly two image paths.")

    label_text = download(LABELS).decode()
    labels = {}
    for item in re.findall(r"item\s*\{([^}]+)\}", label_text):
        class_id = int(re.search(r"\bid:\s*(\d+)", item)[1])
        labels[class_id] = re.search(r"display_name:\s*[\"']([^\"']+)", item)[1]
    graph = load_detector()
    results = []
    with tf.compat.v1.Session(graph=graph) as session:
        for index, filename in enumerate(args.images or ("bus.jpg", "zidane.jpg"), start=1):
            source = Path(filename).read_bytes() if args.images else download(SAMPLES + filename)
            with Image.open(io.BytesIO(source)) as raw_image:
                image = raw_image.convert("RGB")
            records = detect(session, image, labels)
            for threshold in THRESHOLDS:
                kept = [record for record in records if record["Confidence"] >= threshold]
                results.append({
                    "Image": index, "File": filename, "Size": image.size,
                    "Threshold": threshold, "Detections": kept,
                })
                if args.output_prefix:
                    annotate(image, kept).save(
                        f"{args.output_prefix}_{index}_{threshold:.1f}.jpg", quality=94
                    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
