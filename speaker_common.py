"""Shared CLI, embedding aggregation and scoring for the compare_* scripts."""

import argparse

import torch
import torch.nn.functional as F

from youtube_speaker_verification.youtube_audio import DEFAULT_CACHE, chunk, download_audio, load_wav


def build_parser(description, default_threshold):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("url_a", help="YouTube URL for speaker A")
    p.add_argument("url_b", help="YouTube URL for speaker B")
    p.add_argument("--start-a", help="start time in video A (seconds or HH:MM:SS)")
    p.add_argument("--end-a", help="end time in video A")
    p.add_argument("--start-b", help="start time in video B")
    p.add_argument("--end-b", help="end time in video B")
    p.add_argument("--threshold", type=float, default=default_threshold,
                   help=f"cosine similarity above which speakers are judged the same (default {default_threshold})")
    p.add_argument("--win", type=float, default=6.0, help="chunk length in seconds")
    p.add_argument("--hop", type=float, default=3.0, help="chunk hop in seconds")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    return p


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@torch.no_grad()
def embed_chunks(embed_fn, chunks, batch_size, device):
    """Run `embed_fn` ((B, T) tensor -> (B, D) tensor) over chunks; return L2-normalised (N, D)."""
    out = []
    for i in range(0, len(chunks), batch_size):
        out.append(embed_fn(chunks[i:i + batch_size].to(device)).float().cpu())
    return F.normalize(torch.cat(out), dim=1)


def mean_offdiag(m):
    n = m.shape[0]
    if n < 2:
        return float("nan")
    return ((m.sum() - m.diagonal().sum()) / (n * (n - 1))).item()


def run(args, embed_fn, model_name):
    """Download both videos, embed them with `embed_fn` and print the comparison."""
    device = resolve_device(args.device)
    embs = {}
    for label, url, start, end in [("A", args.url_a, args.start_a, args.end_a),
                                   ("B", args.url_b, args.start_b, args.end_b)]:
        path = download_audio(url, args.cache_dir, start, end)
        wav = load_wav(path)
        chunks = chunk(wav, args.win, args.hop)
        print(f"[{label}] {path.name}: {wav.shape[1] / 16000:.1f} s -> {len(chunks)} chunks")
        embs[label] = embed_chunks(embed_fn, chunks, args.batch_size, device)

    a, b = embs["A"], embs["B"]
    spk_a = F.normalize(a.mean(0), dim=0)
    spk_b = F.normalize(b.mean(0), dim=0)
    score = torch.dot(spk_a, spk_b).item()
    cross = a @ b.T

    print(f"\nModel: {model_name}")
    print(f"Cosine similarity (mean embeddings): {score:.4f}")
    print(f"Cosine distance:                     {1 - score:.4f}")
    print(f"Chunk-pair similarity A vs B:        mean {cross.mean():.4f}, std {cross.std():.4f}")
    print(f"Within-A consistency:                {mean_offdiag(a @ a.T):.4f}")
    print(f"Within-B consistency:                {mean_offdiag(b @ b.T):.4f}")
    # The verdict uses the chunk-pair mean: it is on the same scale as single-utterance
    # trials, which published thresholds are tuned on. Averaging embeddings first
    # inflates the cosine, so the mean-embedding score would need a higher threshold.
    verdict = "SAME speaker" if cross.mean() >= args.threshold else "DIFFERENT speakers"
    print(f"Verdict (chunk-pair mean vs threshold {args.threshold}): {verdict}")
    print("\nNote: low within-video consistency suggests multiple speakers or music in that clip; "
          "use --start/--end to isolate one speaker.")
    return score
