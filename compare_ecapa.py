"""Compare the speakers in two YouTube videos with ECAPA-TDNN (SpeechBrain, VoxCeleb).

Example:
    python compare_ecapa.py URL_A URL_B --start-a 10 --end-a 70 --start-b 1:00 --end-b 2:00
"""

from pathlib import Path

from speechbrain.inference.speaker import EncoderClassifier

from youtube_speaker_verification.speaker_common import build_parser, resolve_device, run

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def main():
    args = build_parser(__doc__, default_threshold=0.25).parse_args()
    device = resolve_device(args.device)
    model = EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        savedir=str(Path(__file__).resolve().parent / "pretrained_models" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )
    model.eval()

    def embed(wavs):
        return model.encode_batch(wavs).squeeze(1)  # (B, 1, 192) -> (B, 192)

    run(args, embed, f"ECAPA-TDNN ({MODEL_SOURCE})")


if __name__ == "__main__":
    main()
