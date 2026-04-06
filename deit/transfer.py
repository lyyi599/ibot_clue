import argparse
from pathlib import Path

import torch


DROP_PREFIXES = (
    "head.",
    "head_dist.",
    "pre_logits.",
)

DROP_KEYS = {
    "head.weight",
    "head.bias",
    "head_dist.weight",
    "head_dist.bias",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DeiT/ViT backbone checkpoints into an iBOT-compatible checkpoint."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="deit/checkpoint_teacher.pth",
        help="Input checkpoint path. Supports official DeiT checkpoints and bare state_dict files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="deit/checkpoint_ibot_compatible.pth",
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--input-key",
        default="auto",
        help='Checkpoint key to read from. Use "auto" to detect among model/state_dict/teacher/student.',
    )
    parser.add_argument(
        "--keep-head",
        action="store_true",
        help="Keep classification head keys in the bare model branch.",
    )
    return parser.parse_args()


def looks_like_state_dict(value):
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(k, str) for k in value.keys())


def pick_state_dict(checkpoint, input_key):
    if input_key != "auto":
        if input_key == "":
            return checkpoint, "<root>"
        if input_key not in checkpoint:
            raise KeyError(f'Input key "{input_key}" not found in checkpoint.')
        return checkpoint[input_key], input_key

    if looks_like_state_dict(checkpoint):
        for key in ("model", "state_dict", "teacher", "student", "module"):
            value = checkpoint.get(key)
            if looks_like_state_dict(value):
                return value, key
        return checkpoint, "<root>"

    raise TypeError("Unsupported checkpoint structure.")


def strip_prefix(key):
    prefixes = ("module.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def clean_state_dict(state_dict, keep_head=False):
    cleaned = {}
    removed = []
    has_distillation_token = "dist_token" in state_dict

    for key, value in state_dict.items():
        bare_key = strip_prefix(key)

        if bare_key == "dist_token":
            removed.append(bare_key)
            continue

        if not keep_head and (bare_key in DROP_KEYS or bare_key.startswith(DROP_PREFIXES)):
            removed.append(bare_key)
            continue

        cleaned[bare_key] = value

    if has_distillation_token and "pos_embed" in cleaned:
        pos_embed = cleaned["pos_embed"]
        if pos_embed.ndim != 3 or pos_embed.shape[1] < 3:
            raise ValueError("Unsupported distilled checkpoint: invalid pos_embed shape.")
        # Distilled DeiT inserts a distillation token between cls and patch tokens.
        cleaned["pos_embed"] = torch.cat((pos_embed[:, :1], pos_embed[:, 2:]), dim=1)

    return cleaned, removed


def add_prefix(state_dict, prefix):
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(input_path, map_location="cpu")
    source_state_dict, source_key = pick_state_dict(checkpoint, args.input_key)
    backbone_state_dict, removed_keys = clean_state_dict(
        source_state_dict, keep_head=args.keep_head
    )

    if "cls_token" not in backbone_state_dict or "pos_embed" not in backbone_state_dict:
        raise ValueError("The selected checkpoint does not look like a ViT/DeiT backbone state_dict.")

    teacher_state_dict = add_prefix(backbone_state_dict, "backbone.")
    student_state_dict = add_prefix(backbone_state_dict, "module.backbone.")

    converted = {
        "teacher": teacher_state_dict,
        "student": student_state_dict,
        "state_dict": teacher_state_dict,
        "model": backbone_state_dict,
        "meta": {
            "source_path": str(input_path),
            "source_key": source_key,
            "num_tensors": len(backbone_state_dict),
            "removed_keys": removed_keys,
        },
    }

    torch.save(converted, output_path)

    print(f"Saved converted checkpoint to: {output_path}")
    print(f"Source key: {source_key}")
    print(f"Backbone tensors: {len(backbone_state_dict)}")
    print(f"Teacher sample keys: {list(teacher_state_dict.keys())[:5]}")
    print(f"Student sample keys: {list(student_state_dict.keys())[:5]}")
    if removed_keys:
        print(f"Removed keys: {removed_keys}")


if __name__ == "__main__":
    main()
