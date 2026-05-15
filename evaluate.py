import json
import csv
import os
import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as nnf
import faiss
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipForImageTextRetrieval,
    BlipProcessor,
)

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

QUERY_DIR = ROOT / "images"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
CAPTION_DIR = MODEL_DIR / "gallery_captions"

DETECTOR_FILE = MODEL_DIR / "best.pt"
CLIP_BASE_MODEL = "openai/clip-vit-base-patch32"
CLIP_CHECKPOINT = MODEL_DIR / "full_clip_finetuned.pt"
VECTOR_INDEX_FILE = MODEL_DIR / "index_C_b07.bin"
INDEX_META_FILE = MODEL_DIR / "metadata.json"

BLIP_MODEL_NAME = "Salesforce/blip-itm-large-coco"

NUM_CANDIDATES = 40
RERANK_LIMIT = 20
REPORT_AT = [5, 10, 15]
BLIP_BATCH = 16

DET_CONF = 0.25
DET_IOU = 0.45
SEARCH_EF = 128


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def transfer_to_device(batch: dict, net: torch.nn.Module, device: torch.device):
    """Move tensors to device using the model parameter dtype."""
    dtype = next(net.parameters()).dtype
    moved = {}

    for key, value in batch.items():
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device)

    return moved


def parse_clip_output(net: torch.nn.Module, out):
    """Extract image embedding from any CLIP output format."""
    if torch.is_tensor(out):
        return out

    if getattr(out, "image_embeds", None) is not None:
        return out.image_embeds

    if getattr(out, "pooler_output", None) is not None:
        pooled = out.pooler_output
        projector = getattr(net, "visual_projection", None)

        if (
            projector is not None
            and hasattr(projector, "in_features")
            and pooled.shape[-1] == projector.in_features
        ):
            return projector(pooled)

        return pooled

    if getattr(out, "last_hidden_state", None) is not None:
        cls_vec = out.last_hidden_state[:, 0]
        projector = getattr(net, "visual_projection", None)

        if projector is not None:
            return projector(cls_vec)

        return cls_vec

    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item):
                return item

    raise RuntimeError(f"Unsupported CLIP output type: {type(out).__name__}")


def unique_keep_order(values: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    ordered = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return ordered


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def score_recall(rank_list: list, positives: set, k: int) -> float:
    found = len([x for x in rank_list[:k] if x in positives])
    return found / len(positives) if positives else 0.0


def score_ndcg(rank_list: list, positives: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, candidate in enumerate(rank_list[:k])
        if candidate in positives
    )

    ideal = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(positives), k))
    )

    return dcg / ideal if ideal > 0 else 0.0


def score_ap(rank_list: list, positives: set, k: int) -> float:
    hits = 0
    total = 0.0

    for idx, candidate in enumerate(rank_list[:k]):
        if candidate in positives:
            hits += 1
            total += hits / (idx + 1)

    denom = min(len(positives), k)
    return total / denom if denom > 0 else 0.0


def summarize_metrics(rank_list: list, positives: set) -> dict:
    output = {}

    for cutoff in REPORT_AT:
        output[f"Recall@{cutoff}"] = score_recall(rank_list, positives, cutoff)
        output[f"NDCG@{cutoff}"] = score_ndcg(rank_list, positives, cutoff)
        output[f"mAP@{cutoff}"] = score_ap(rank_list, positives, cutoff)

    return output


# ---------------------------------------------------------------------
# ID handling
# ---------------------------------------------------------------------
def filename_to_product_id(stem: str) -> str:
    """
    Convert:
        id_00000001_02_1_front
    into:
        id_00000001_02
    """
    tokens = stem.split("_")

    if len(tokens) >= 3:
        return f"{tokens[0]}_{tokens[1]}_{tokens[2]}"

    if len(tokens) >= 2:
        return f"{tokens[0]}_{tokens[1]}"

    return stem


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def initialize_models(device: torch.device):
    precision = torch.float16 if device.type == "cuda" else torch.float32

    print("Loading detector...")
    detector = YOLO(str(DETECTOR_FILE))

    print("Loading CLIP encoder...")
    base_model_name = CLIP_BASE_MODEL

    clip_net = CLIPModel.from_pretrained(
        base_model_name,
        torch_dtype=precision,
        local_files_only=False,
    ).to(device)

    state_dict = torch.load(
        CLIP_CHECKPOINT,
        map_location=device,
    )

    clip_net.load_state_dict(state_dict)
    clip_net.eval()

    clip_proc = CLIPProcessor.from_pretrained(
        base_model_name,
        local_files_only=False,
    )

    print("Loading BLIP-ITM reranker...")
    rerank_proc = None
    rerank_net = None

    try:
        rerank_proc = BlipProcessor.from_pretrained(
            BLIP_MODEL_NAME,
            local_files_only=False,
        )

        rerank_net = BlipForImageTextRetrieval.from_pretrained(
            BLIP_MODEL_NAME,
            local_files_only=False,
            torch_dtype=precision,
            use_safetensors=True,
        ).to(device).eval()

    except Exception as ex:
        print(f"[BLIP disabled] {ex}")

    return (
        detector,
        clip_net,
        clip_proc,
        rerank_net,
        rerank_proc,
    )


# ---------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------
def open_vector_index():
    print("Loading vector index...")
    index = faiss.read_index(str(VECTOR_INDEX_FILE))
    index.hnsw.efSearch = SEARCH_EF

    with open(INDEX_META_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    stems = metadata.get("stems", [])

    item_ids = metadata.get("item_ids")
    if not item_ids:
        item_ids = [filename_to_product_id(s) for s in stems]

    return index, stems, item_ids


# ---------------------------------------------------------------------
# Caption loading
# ---------------------------------------------------------------------
def read_gallery_captions(stem_names: list[str]) -> dict[str, str]:
    text_map = {}
    missing_count = 0

    for stem in stem_names:
        path = CAPTION_DIR / f"{stem}.json"

        if path.exists():
            with open(path, encoding="utf-8") as f:
                text_map[stem] = json.load(f).get("caption", "")
        else:
            text_map[stem] = ""
            missing_count += 1

    if missing_count:
        print(
            f"[captions] Missing {missing_count}/{len(stem_names)} "
            "captions; using empty strings."
        )
    else:
        print(f"[captions] Loaded all {len(stem_names)} captions.")

    return text_map

# ---------------------------------------------------------------------
# Detection and cropping
# ---------------------------------------------------------------------
def detect_primary_object(
    detector: YOLO,
    image: Image.Image,
    conf_threshold: float = DET_CONF,
    iou_threshold: float = DET_IOU,
    return_raw_boxes: bool = False,
):
    """
    Run YOLO and return the highest-confidence crop.
    Falls back to None if no valid detection is found.
    """
    predictions = detector.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    boxes = predictions[0].boxes

    if boxes is None or len(boxes) == 0:
        if return_raw_boxes:
            return None, boxes
        return None

    confidences = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confidences))

    x1, y1, x2, y2 = (
        boxes.xyxy[best_idx]
        .cpu()
        .numpy()
        .astype(int)
    )

    width, height = image.size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        if return_raw_boxes:
            return None, boxes
        return None

    cropped = image.crop((x1, y1, x2, y2))

    if return_raw_boxes:
        return cropped, boxes

    return cropped


# ---------------------------------------------------------------------
# CLIP feature extraction
# ---------------------------------------------------------------------
@torch.inference_mode()
def encode_query_image(
    image: Image.Image,
    clip_net: CLIPModel,
    clip_proc: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    """
    Generate a normalized CLIP embedding as float32 NumPy array.
    """
    batch = clip_proc(images=image, return_tensors="pt")
    batch = transfer_to_device(batch, clip_net, device)

    raw = clip_net.get_image_features(**batch)
    vec = parse_clip_output(clip_net, raw).float()

    vec = nnf.normalize(vec, p=2, dim=-1)

    return (
        vec.squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ---------------------------------------------------------------------
# FAISS search
# ---------------------------------------------------------------------
def retrieve_candidates(
    query_vector: np.ndarray,
    vector_index: faiss.Index,
    gallery_stems: list[str],
) -> list[str]:
    """
    Search the FAISS index and return candidate stems.
    """
    distances, indices = vector_index.search(
        query_vector[np.newaxis, :],
        NUM_CANDIDATES,
    )

    results = [
        gallery_stems[idx]
        for idx in indices[0]
        if idx != -1
    ]

    return results


# ---------------------------------------------------------------------
# BLIP-ITM scoring
# ---------------------------------------------------------------------
@torch.inference_mode()
def compute_itm_probabilities(
    query_image: Image.Image,
    candidate_captions: list[str],
    rerank_net: BlipForImageTextRetrieval,
    rerank_proc: BlipProcessor,
    device: torch.device,
) -> np.ndarray:
    """
    Compute BLIP-ITM match probabilities.
    Returns zeros if BLIP is unavailable.
    """
    if rerank_net is None or rerank_proc is None:
        return np.zeros(len(candidate_captions), dtype=np.float32)

    scores = []

    for start in range(0, len(candidate_captions), BLIP_BATCH):
        text_batch = candidate_captions[start : start + BLIP_BATCH]
        image_batch = [query_image] * len(text_batch)

        batch = rerank_proc(
            images=image_batch,
            text=text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        batch = transfer_to_device(batch, rerank_net, device)

        outputs = rerank_net(
            **batch,
            use_itm_head=True,
        )

        probs = nnf.softmax(
            outputs.itm_score.float(),
            dim=1,
        )[:, 1]

        scores.extend(
            probs.cpu().numpy().tolist()
        )

    return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------
def reorder_candidates(
    query_image: Image.Image,
    candidate_stems: list[str],
    caption_lookup: dict[str, str],
    rerank_net: BlipForImageTextRetrieval,
    rerank_proc: BlipProcessor,
    device: torch.device,
) -> list[str]:
    """
    Re-rank the top subset using BLIP-ITM and append the remaining
    candidates in original FAISS order.
    """
    head = candidate_stems[:RERANK_LIMIT]
    tail = candidate_stems[RERANK_LIMIT:]

    head_captions = [
        caption_lookup.get(stem, "")
        for stem in head
    ]

    match_scores = compute_itm_probabilities(
        query_image,
        head_captions,
        rerank_net,
        rerank_proc,
        device,
    )

    ranking = np.argsort(-match_scores)

    reranked_head = [
        head[idx]
        for idx in ranking
    ]

    return reranked_head + tail


# ---------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------
def run_full_evaluation(
    query_files: list[Path],
    detector: YOLO,
    clip_net: CLIPModel,
    clip_proc: CLIPProcessor,
    vector_index: faiss.Index,
    gallery_stems: list[str],
    gallery_item_ids: list[str],
    caption_lookup: dict[str, str],
    rerank_net: BlipForImageTextRetrieval,
    rerank_proc: BlipProcessor,
    device: torch.device,
    verbose_debug: bool = False,
):
    """
    Full pipeline:
        query image
        -> YOLO crop (or full-image fallback)
        -> CLIP embedding
        -> FAISS retrieval
        -> BLIP-ITM reranking
        -> metric computation
    """
    stem_to_item = dict(
        zip(gallery_stems, gallery_item_ids)
    )

    metric_names = [
        f"{metric}@{k}"
        for metric in ["Recall", "NDCG", "mAP"]
        for k in REPORT_AT
    ]

    per_query_rows = []
    all_metric_dicts = []

    for file_path in tqdm(query_files, desc="Evaluating"):
        query_stem = file_path.stem
        query_item_id = filename_to_product_id(query_stem)

        if not query_item_id:
            continue

        original_image = Image.open(file_path).convert("RGB")

        cropped_image, detected_boxes = detect_primary_object(
            detector,
            original_image,
            return_raw_boxes=True,
        )

        image_for_search = original_image
        was_detected = False

        if (
            cropped_image is not None
            and detected_boxes is not None
            and len(detected_boxes) > 0
        ):
            image_for_search = cropped_image
            was_detected = True

        if verbose_debug and not was_detected:
            print(
                f"[debug] No detection for {file_path.name}; "
                "using full image."
            )

        embedding = encode_query_image(
            image_for_search,
            clip_net,
            clip_proc,
            device,
        )

        ranked_stems = retrieve_candidates(
            embedding,
            vector_index,
            gallery_stems,
        )

        # Remove exact same image if present.
        ranked_stems = [
            stem
            for stem in ranked_stems
            if stem != query_stem
        ]

        ranked_stems = reorder_candidates(
            image_for_search,
            ranked_stems,
            caption_lookup,
            rerank_net,
            rerank_proc,
            device,
        )

        relevant_stems = {
            stem
            for stem, item_id in zip(
                gallery_stems,
                gallery_item_ids,
            )
            if item_id == query_item_id
            and stem != query_stem
        }

        metrics = summarize_metrics(
            ranked_stems,
            relevant_stems,
        )

        all_metric_dicts.append(metrics)

        row = {
            "image": file_path.name,
            "query_item_id": query_item_id,
            "relevant_in_gallery": len(relevant_stems),
            "detected": was_detected,
            **metrics,
        }

        per_query_rows.append(row)

    # Aggregate means
    if all_metric_dicts:
        aggregate = {
            name: float(
                np.mean([
                    metric_dict[name]
                    for metric_dict in all_metric_dicts
                ])
            )
            for name in metric_names
        }
    else:
        aggregate = {}

    return per_query_rows, aggregate

# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------
def program_entry():
    parser = argparse.ArgumentParser(
        description="Evaluate the fashion retrieval pipeline"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional diagnostic messages",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Evaluate only a single query image filename",
    )
    parser.add_argument(
        "--conf",
        type=float,
        help="Override YOLO confidence threshold",
    )
    parser.add_argument(
        "--topk",
        type=int,
        help="Override FAISS retrieval depth",
    )

    args = parser.parse_args()

    # Optional runtime overrides
    global DET_CONF, NUM_CANDIDATES

    if args.conf is not None:
        DET_CONF = args.conf

    if args.topk is not None:
        NUM_CANDIDATES = args.topk

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Select device
    compute_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {compute_device}\n")

    # Gather query images
    query_files = sorted(
        path
        for path in QUERY_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if args.image:
        query_files = [
            path
            for path in query_files
            if path.name == args.image
        ]

    if not query_files:
        print(f"No query images found in {QUERY_DIR}")
        sys.exit(0)

    print(f"Found {len(query_files)} query image(s).\n")

    # Load models and index
    (
        detector,
        clip_net,
        clip_proc,
        rerank_net,
        rerank_proc,
    ) = initialize_models(compute_device)

    (
        vector_index,
        gallery_stems,
        gallery_item_ids,
    ) = open_vector_index()

    print("Loading gallery captions...")
    caption_lookup = read_gallery_captions(gallery_stems)

    print(
        "\nRunning evaluation "
        "(YOLO -> CLIP -> FAISS -> BLIP-ITM -> metrics)..."
    )

    rows, aggregate = run_full_evaluation(
        query_files=query_files,
        detector=detector,
        clip_net=clip_net,
        clip_proc=clip_proc,
        vector_index=vector_index,
        gallery_stems=gallery_stems,
        gallery_item_ids=gallery_item_ids,
        caption_lookup=caption_lookup,
        rerank_net=rerank_net,
        rerank_proc=rerank_proc,
        device=compute_device,
        verbose_debug=args.debug,
    )

    # Metric column names
    metric_columns = [
        f"{metric}@{k}"
        for metric in ["Recall", "NDCG", "mAP"]
        for k in REPORT_AT
    ]

    # -----------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------
    print("\n" + "═" * 70)

    header = (
        f"{'Image':<30}  "
        f"{'item_id':<14}"
        + "".join(
            f"{column:>12}"
            for column in metric_columns
        )
    )
    print(header)
    print("─" * 70)

    for row in rows:
        values = "".join(
            f"{row[column]:>12.4f}"
            for column in metric_columns
        )

        print(
            f"{row['image']:<30}  "
            f"{row['query_item_id']:<14}"
            f"{values}"
        )

    print("─" * 70)

    if aggregate:
        mean_values = "".join(
            f"{aggregate[column]:>12.4f}"
            for column in metric_columns
        )

        print(
            f"{'MEAN':<30}  "
            f"{'':14}"
            f"{mean_values}"
        )

    print("═" * 70 + "\n")

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------
    csv_file = RESULT_DIR / "per_query_results.csv"

    csv_fields = [
        "image",
        "query_item_id",
        "relevant_in_gallery",
        "detected",
        *metric_columns,
    ]

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )
        writer.writeheader()
        writer.writerows(rows)

        if aggregate:
            writer.writerow(
                {
                    "image": "MEAN",
                    "query_item_id": "",
                    "relevant_in_gallery": "",
                    "detected": "",
                    **aggregate,
                }
            )

    print(f"Per-query results saved to {csv_file}")

    # -----------------------------------------------------------------
    # Save aggregate JSON
    # -----------------------------------------------------------------
    json_file = RESULT_DIR / "aggregate_metrics.json"

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "n_queries": len(rows),
                **(aggregate or {}),
            },
            f,
            indent=2,
        )

    print(f"Aggregate metrics saved to {json_file}")


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    program_entry()