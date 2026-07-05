import json
from pathlib import Path


GT_JSON = "data/thumos14_tad_gcd/opentad/thumos14_tad_gcd_test_new.json"
PRED_JSON = "/data/lcl/checkpoints/opentad_tadgcd/test_old_detector_on_test_new_i3d/gpu1_id0/result_detection.json"


def temporal_iou(seg1, seg2):
    s1, e1 = float(seg1[0]), float(seg1[1])
    s2, e2 = float(seg2[0]), float(seg2[1])

    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)

    if union <= 0:
        return 0.0
    return inter / union


def load_gt(path):
    with open(path, "r") as f:
        data = json.load(f)

    gt_by_video = {}

    for vid, info in data["database"].items():
        anns = info.get("annotations", [])
        gt_by_video[vid] = []
        for ann in anns:
            gt_by_video[vid].append({
                "segment": ann["segment"],
                "label": ann.get("label", None),
            })

    return gt_by_video


def load_pred(path):
    with open(path, "r") as f:
        data = json.load(f)

    results = data["results"]
    pred_by_video = {}

    for vid, preds in results.items():
        clean_preds = []
        for p in preds:
            if "segment" not in p:
                continue
            clean_preds.append({
                "segment": p["segment"],
                "score": float(p.get("score", 0.0)),
                "label": p.get("label", None),
            })

        clean_preds = sorted(clean_preds, key=lambda x: x["score"], reverse=True)
        pred_by_video[vid] = clean_preds

    return pred_by_video


def evaluate_recall(gt_by_video, pred_by_video, tiou_thresholds, topk_list):
    total_gt = sum(len(v) for v in gt_by_video.values())
    total_pred = sum(len(v) for v in pred_by_video.values())

    print("GT videos:", len(gt_by_video))
    print("Pred videos:", len(pred_by_video))
    print("Total GT instances:", total_gt)
    print("Total predictions:", total_pred)
    print()

    for topk in topk_list:
        print(f"===== Top-{topk} predictions per video =====")
        for th in tiou_thresholds:
            hit = 0

            for vid, gts in gt_by_video.items():
                preds = pred_by_video.get(vid, [])[:topk]
                pred_segments = [p["segment"] for p in preds]

                for gt in gts:
                    max_iou = 0.0
                    for pred_seg in pred_segments:
                        max_iou = max(max_iou, temporal_iou(pred_seg, gt["segment"]))

                    if max_iou >= th:
                        hit += 1

            recall = hit / max(total_gt, 1)
            print(f"tIoU={th:.1f}  Recall={recall * 100:.2f}%")
        print()


def main():
    gt_by_video = load_gt(GT_JSON)
    pred_by_video = load_pred(PRED_JSON)

    evaluate_recall(
        gt_by_video=gt_by_video,
        pred_by_video=pred_by_video,
        tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
        topk_list=[50, 100, 200, 500, 1000, 2000],
    )


if __name__ == "__main__":
    main()