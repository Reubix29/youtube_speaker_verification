"""Compare the speakers in two YouTube videos with ReDimNet (IDRnD, VoxCeleb2).

ReDimNet-B6 is among the strongest openly downloadable speaker-verification
models (~0.4% EER on VoxCeleb1-O vs ~0.8% for ECAPA-TDNN). The decision
threshold is not calibrated for YouTube audio: score a few known same/different
pairs and adjust --threshold.

Example:
    python compare_redimnet.py URL_A URL_B --start-a 10 --end-a 70
"""

import torch

from youtube_speaker_verification.speaker_common import build_parser, resolve_device, run


def main():
    p = build_parser(__doc__, default_threshold=0.4)
    p.add_argument("--hub-repo", default="IDRnD/ReDimNet", help="torch.hub repo")
    p.add_argument("--model-name", default="b6", help="ReDimNet size: b0..b6, M, S")
    p.add_argument("--train-type", default="ft_lm", help="'ptn' (pretrained) or 'ft_lm' (large-margin fine-tuned)")
    p.add_argument("--dataset", default="vox2", help="training set, e.g. 'vox2' or 'vb2+vox2+cnc'")
    args = p.parse_args()

    device = resolve_device(args.device)
    model = torch.hub.load(args.hub_repo, "ReDimNet", model_name=args.model_name,
                           train_type=args.train_type, dataset=args.dataset, trust_repo=True)
    model.to(device).eval()

    run(args, model,
        f"ReDimNet-{args.model_name} ({args.hub_repo}, {args.train_type}, {args.dataset})")


if __name__ == "__main__":
    main()
