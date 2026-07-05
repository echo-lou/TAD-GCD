_base_ = [
    "../../configs/_base_/datasets/thumos-14/features_i3d_pad.py",
    "../../configs/_base_/models/tridet.py",
]

# ------------------------------------------------------------
# THUMOS14-TAD-GCD old-class TriDet baseline
# train: train_labeled, old 10 classes only
# val/test: test_old, old 10 classes only
# feature: OpenTAD / ActionFormer I3D, stride=4, dim=2048
# ------------------------------------------------------------

feature_root = "/data/lcl/action/thumos14/thumos/i3d_features/i3d_actionformer_stride4_thumos/"

old_ann_train = "data/thumos14_tad_gcd/opentad/thumos14_tad_gcd_train_labeled.json"
old_ann_test = "data/thumos14_tad_gcd/opentad/thumos14_tad_gcd_test_old.json"
old_class_map = "data/thumos14_tad_gcd/opentad/category_idx_old.txt"

model = dict(
    projection=dict(
        in_channels=2048,
        input_noise=0.0005,
    ),
    rpn_head=dict(
        num_classes=10,
    ),
)

dataset = dict(
    train=dict(
        ann_file=old_ann_train,
        subset_name="training",
        class_map=old_class_map,
        data_path=feature_root,
        block_list=feature_root + "missing_files.txt",
    ),
    val=dict(
        ann_file=old_ann_test,
        subset_name="validation",
        class_map=old_class_map,
        data_path=feature_root,
        block_list=feature_root + "missing_files.txt",
    ),
    test=dict(
        ann_file=old_ann_test,
        subset_name="validation",
        class_map=old_class_map,
        data_path=feature_root,
        block_list=feature_root + "missing_files.txt",
    ),
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=old_ann_test,
)

# smoke test first: only 2 epochs
scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=1,
    max_epoch=2,
)

workflow = dict(
    logging_interval=5,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=1,
)

work_dir = "exps/tad_gcd/tridet_thumos14_old_i3d_smoke"

# ------------------------------------------------------------
# Solver: required by tools/train.py
# ------------------------------------------------------------
solver = dict(
    train=dict(
        batch_size=2,
        num_workers=2,
    ),
    val=dict(
        batch_size=1,
        num_workers=2,
    ),
    test=dict(
        batch_size=1,
        num_workers=2,
    ),
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
    ),
    clip_grad_norm=1.0,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
)

# ------------------------------------------------------------
# Inference / post-processing: adapted from configs/causaltad/thumos_i3d.py
# ------------------------------------------------------------
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=False,
)
