# YouTube Speaker Verification

Give this tool links to two YouTube videos and it tells you whether the person speaking in each one is likely to be **the same person** or **two different people**.

It does this with AI "speaker verification" models. These models turn a voice into a numerical fingerprint (an *embedding*) and then measure how similar the two fingerprints are. Two models are included:

| Script | Model | Notes |
|---|---|---|
| `compare_ecapa.py` | **ECAPA-TDNN** (SpeechBrain) | The widely used standard model. |
| `compare_redimnet.py` | **ReDimNet-B6** (IDRnD) | Newer and more accurate: roughly half the error rate of ECAPA-TDNN on the standard VoxCeleb benchmark. |

---

## 1. One-time installation

Do these steps once. It takes about 15–30 minutes, mostly waiting for downloads.

### Step 1: Install Miniconda

Miniconda is a free program that installs Python and the other software this tool needs, without interfering with anything else on your computer.

1. Go to <https://www.anaconda.com/download/success> and scroll to **Miniconda Installers**.
2. Download the installer for your system (Windows, macOS or Linux) and run it. Accept the default options.

### Step 2: Open a terminal

The terminal is a window where you type commands.

- **Windows:** open the Start menu and search for **Anaconda Prompt (miniconda3)**. Use this window, not the normal Command Prompt.
- **macOS:** open **Terminal** (press Cmd+Space and type "Terminal").
- **Linux:** open your Terminal application.

To run a command, type it (or copy and paste it) and press **Enter**.

### Step 3: Download this tool

1. On this project's GitHub page, click the green **Code** button, then **Download ZIP**.
2. Unzip it somewhere easy to find, such as your Documents folder. This gives you a folder called `youtube_speaker_verification` (it may end in `-main`).

### Step 4: Go to the folder in the terminal

Type `cd ` (with a space after it), then drag the unzipped folder from your file explorer into the terminal window. The folder's location will appear. Press **Enter**.

For example, it may look like this:

```
cd C:\Users\yourname\Documents\youtube_speaker_verification
```

### Step 5: Install the software

Copy and paste these three commands, one at a time, pressing Enter after each. If it asks `Proceed ([y]/n)?`, type `y` and press Enter.

```
conda create -n spkver -c conda-forge python=3.12 ffmpeg
```
```
conda activate spkver
```
```
pip install -r requirements.txt
```

The last command downloads a few gigabytes, so it can take a while. When the terminal is ready for typing again, installation is done.

---

## 2. Using the tool

Every time you open a new terminal to use the tool, first go to the folder (Step 4) and run:

```
conda activate spkver
```

You'll know it worked when `(spkver)` appears at the start of the line.

### Compare two videos

Replace the two links with the YouTube videos you want to compare:

```
python compare_redimnet.py "https://www.youtube.com/watch?v=VIDEO_A" "https://www.youtube.com/watch?v=VIDEO_B"
```

To use the ECAPA-TDNN model instead, write `compare_ecapa.py` in place of `compare_redimnet.py`.

Put each link inside quotation marks `" "`. The first run takes longer because it downloads the AI model.

### Choose which part of each video to use (recommended)

Most videos contain more than one voice: interviewers, audiences, music, intros. For reliable results, pick a section where **only the person you care about is speaking**, ideally 30 seconds to a few minutes long.

Use `--start-a` and `--end-a` for the first video, and `--start-b` and `--end-b` for the second. Times can be in seconds (`90`), minutes:seconds (`1:30`) or hours:minutes:seconds (`1:02:30`).

```
python compare_redimnet.py "LINK_A" "LINK_B" --start-a 1:00 --end-a 3:00 --start-b 0:30 --end-b 2:30
```

This compares minutes 1–3 of video A with 0:30–2:30 of video B.

### Just download the audio

To save a video's audio as a WAV file without comparing anything:

```
python youtube_audio.py "LINK" --start 0:30 --end 2:00
```

The file is saved in the `audio_cache` folder.

---

## 3. Understanding the results

You'll see output like this:

```
Model: ReDimNet-b6 (IDRnD/ReDimNet, ft_lm, vox2)
Cosine similarity (mean embeddings): 0.2390
Cosine distance:                     0.7610
Chunk-pair similarity A vs B:        mean 0.1931, std 0.0578
Within-A consistency:                0.8431
Within-B consistency:                0.7642
Verdict (chunk-pair mean vs threshold 0.4): DIFFERENT speakers
```

The tool cuts each recording into short 6-second pieces ("chunks") and compares every piece from video A with every piece from video B.

- **Verdict:** the tool's conclusion, *SAME speaker* or *DIFFERENT speakers*. It is based on the *chunk-pair mean*.
- **Chunk-pair similarity mean:** the average similarity between pieces of A and pieces of B. **Higher means more alike.** As a rough guide, the same person usually scores around 0.6–0.9 and different people usually score below 0.3.
- **Cosine similarity (mean embeddings):** a second similarity score made from one fingerprint per video. It is usually higher than the chunk-pair mean, so compare it only with other scores of the same kind.
- **Cosine distance:** one minus the similarity; higher means more different.
- **Within-A / Within-B consistency:** how similar each video is *to itself*. One clear speaker typically gives about 0.7 or more. A low value means that clip probably contains several voices, music or noise, so trim it with `--start`/`--end` and try again.

### Where the thresholds come from

Each script decides "same" or "different" by checking whether the chunk-pair mean is above a fixed threshold. You can change it with `--threshold`.

- **ECAPA-TDNN: 0.25.** This is the default in SpeechBrain's own verification function (`SpeakerRecognition.verify_batch`) for this model. SpeechBrain chose it as a working point for the model trained on the VoxCeleb dataset, which contains short clips of celebrity interviews taken from YouTube. That threshold applies to comparing one short clip with another. This is why the tool compares 6-second chunks with each other and does not compare one averaged fingerprint per video. Averaging pushes scores upwards, so 0.25 would be too lenient.
- **ReDimNet-B6: 0.4.** ReDimNet's authors do not publish a recommended threshold. 0.4 is a rough value chosen for this tool: ReDimNet's scores are spread differently from ECAPA-TDNN's, and in our tests different people scored about 0.2 and the same person about 0.7–0.8. 0.4 falls in that gap. It has **not** been calibrated on a large dataset.

In both cases a threshold is a trade-off. Raising it makes the tool less likely to wrongly say "same" but more likely to miss a genuine match. The best value depends on your recordings, so if the result matters, calibrate it on pairs where you already know the answer.

### Important caveats

- **This is a statistical estimate, not proof.** Recording quality, microphones, background noise, emotion, illness and age differences between recordings all affect the score. Don't treat a single result as conclusive.
- **Results near the threshold are uncertain.** Treat them as "can't tell" rather than a firm answer.
- **The thresholds are rough defaults** (see above). For serious use, test with a few pairs you *know* are the same or different people, and adjust with `--threshold`, for example `--threshold 0.35`.
- **A voice clone can look like a match.** These models measure how alike two voices *sound*. A convincing AI copy of someone's voice can score as "same speaker". Telling real speech from synthetic speech needs a separate deepfake-detection model.
- Only download and analyse content you're allowed to, and respect YouTube's terms of service and people's privacy.

---

## 4. Experiment: does the Aviation Files narrator sound like Chris Williamson?

We used the tool to test whether the narrator of an aviation documentary sounds like podcaster Chris Williamson. Each comparison used 2–3 minute sections (listed in the table) in which a single person speaks throughout. All numbers below are the chunk-pair similarity (higher means more alike).

**Videos tested:**

- **A:** *MH370: What If It Never Crashed* (Aviation Files): <https://youtu.be/29yWY63Tvs8>
- **B:** *Q&A: Becoming A Dad, Favourite Peptides & Red Rising* (Chris Williamson): <https://youtu.be/8Oj3NxSLP1U>

**Reference videos.** We used these to see what the scores look like when we know the answer:

- *Steve Jobs' 2005 Stanford Commencement Address* (Stanford): <https://www.youtube.com/watch?v=UF8uR6Z6KLc>
- *Randy Pausch's Last Lecture: Achieving Your Childhood Dreams* (Carnegie Mellon University): <https://www.youtube.com/watch?v=ji5_MqicxSo>

| Comparison | Sections used | ECAPA-TDNN | ReDimNet-B6 | Verdict |
|---|---|---|---|---|
| A vs B | A 3:00–6:00, B 10:00–13:00 | 0.60 | 0.70 | Same |
| A vs B | A 12:00–15:00, B 1:00:00–1:03:00 | 0.60 | 0.70 | Same |
| B vs B (Chris vs himself) | 10:00–13:00 vs 1:00:00–1:03:00 | 0.75 | 0.81 | Same |
| Steve Jobs vs himself | 1:00–3:00 vs 6:40–8:40 | 0.80 | 0.83 | Same |
| **Steve Jobs vs Randy Pausch (don't match)** | Jobs 1:00–3:00, Pausch 10:00–12:00 | 0.21 | 0.19 | Different |

**Result.** Both models judge the Aviation Files narrator to be the same speaker as Chris Williamson, and the result holds across different parts of both videos. The narrator scores far above the known non-matching pair (Steve Jobs vs Randy Pausch). It scores somewhat below Chris compared with himself (about 0.10 lower on both models). That gap could come from a different microphone, room or audio processing, or from a voice that closely imitates his without being a recording of him.

**Interpretation.** The narrator's voice is a very close match to Chris Williamson's. However, this kind of model cannot tell whether that voice is really him or an AI-generated copy of it. Based on this video from Chris' YouTube channel, the aviation video was made using Archer, an ElevenLabs voice AI. These results strongly indicate that Chris's voice was used to train Archer.

To reproduce the first row:

```bash
python compare_redimnet.py "https://youtu.be/29yWY63Tvs8" "https://youtu.be/8Oj3NxSLP1U" --start-a 3:00 --end-a 6:00 --start-b 10:00 --end-b 13:00
```

---

## 5. Troubleshooting

| Problem | What to do |
|---|---|
| `conda: command not found` / `'conda' is not recognized` | On Windows, make sure you're using **Anaconda Prompt**. On macOS/Linux, close the terminal and open a new one after installing Miniconda. |
| `No module named ...` | You forgot `conda activate spkver`. Run it and try again. |
| `ffmpeg not found` | Run `conda activate spkver` then `conda install -c conda-forge ffmpeg`. |
| Download errors from YouTube (e.g. "Sign in to confirm") | YouTube changes often. Update the downloader with `pip install -U yt-dlp`. Age-restricted or private videos may not work. |
| It's very slow | It runs on the computer's processor. An NVIDIA graphics card is used automatically if one is set up, which is much faster. Shorter sections with `--start`/`--end` also help. |
| `can't open file ...` | You're not in the project folder. Repeat Step 4. |

---
