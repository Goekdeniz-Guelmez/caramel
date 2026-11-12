# training caramel on a MacBook Pro (M4 Pro, 24 GB)

Three runs, smallest first. Do them in order: each one checks the setup the next one depends on.

| run | starts from | data | steps × frames/step | rough time |
| --- | --- | --- | --- | --- |
| 1. proof of concept | `caramel-1` | 10 h of Seamless Interaction | 300 × 6k | ~30 min |
| 2. fine-tune | `caramel-1` | Seamless Interaction + 20% Libri-Light | 2,000 × 48k | ~1 day |
| 3. pretrain from scratch | random | Libri-Light `small`, then Seamless Interaction | 2,000 + 4,000 × 48k | ~2.5 days |

**The times are estimates, not measurements.** They assume about 1,300 training frames/s: roughly 2.3 GFLOP per frame against about 3 TFLOPS of sustained bf16 throughput on the 20-core M4 Pro GPU. `train.py` prints the real `frames/s` every 10 steps. Read it after the first few minutes and recompute: `hours = steps × frames_per_step / frames_s / 3600`.

For scale, the original `caramel-1` recipe (9k + 16k + 22k steps × 48k frames) would take about 3 weeks on this machine. That's why run 3 is a scaled-down version, and why fine-tuning (run 2) is where a laptop actually gets you somewhere.

## memory

24 GB is unified memory shared with macOS, and by default the GPU can only use about two thirds to three quarters of it. Training a 413M-parameter model has a fixed cost of about 9 GB:

| | GB |
| --- | --- |
| float32 weights | 1.7 |
| AdamW moments | 3.3 |
| gradients + accumulation buffer | 3.3 |
| bf16 weight copies for the matmuls | 0.8 |

That leaves about 6-8 GB for activations, which grow with `--fpb` (frames per micro-batch). Rough estimate: 3 GB per 1,000 frames, mostly the attention matrices of the 1500-frame windows. So use **`--fpb 1500`** and get the batch from `--accum`: `--fpb 1500 --accum 32` is the original recipe's 48k frames per step.

If you see swap in Activity Monitor (memory pressure goes yellow or red), training slows down a lot. Close other apps first. If that's not enough, you can let the GPU wire more memory until the next reboot, leaving ~4 GB for macOS:

```bash
sudo sysctl iogpu.wired_limit_mb=20000
```

Generation is light: the bf16 model is 0.8 GB, and with `--bits 8` it's 0.4 GB.

## setup

```bash
./setup.sh ./caramel-1
source .venv/bin/activate
python generate.py --model ./caramel-1 --seconds 20 --realtime --bits 8   # sanity check: should print well under 80 ms/frame
```

## data

The token stores are small: one hour of conversation is 1.4 MB of tokens and one hour of one-person speech is 0.7 MB. The raw audio is the problem. Seamless Interaction is ~27 TB with its video, and even the audio alone is about 3 TB. So download it one shard at a time, tokenize it, then delete it. `prepare.py` appends to an existing store, so every shard lands in the same `data/stereo`.

Seamless Interaction is laid out as `{naturalistic,improvised}/{train,dev,test}/BBBB/AAAA.tar` (batch / archive). Each tar holds one 48 kHz wav per participant, named `V.._S.._I.._P...wav`; the two participants of one interaction share the `V_S_I` part.

```bash
seamless_shard () {   # $1 = path inside the dataset, e.g. naturalistic/train/0000, $2 = output store
  hf download facebook/seamless-interaction --repo-type dataset --include "$1/*" --local-dir raw/seamless
  for t in raw/seamless/$1/*.tar; do tar -xf "$t" -C raw/seamless/$1 --wildcards '*.wav' && rm "$t"; done
  python prepare.py "raw/seamless/$1/**/*.wav" --pair_regex '(V\d+_S\d+_I\d+)_P\d+' --out "$2"
  rm -rf "raw/seamless/$1"
}

seamless_shard naturalistic/dev/0000 data/dev      # held-out conversations for --dev
seamless_shard naturalistic/train/0000 data/stereo
seamless_shard naturalistic/train/0001 data/stereo # ...repeat for more batches, improvised/train/* too
```

**Check the first shard's output.** Each conversation should print one line. `skip ...: 1 tracks` means the two participants of an interaction landed in different archives or batches. If you see a lot of those, pass two batch folders to one `prepare.py` call so the pairs meet. I couldn't inspect the tars from where this was written, so the archive layout above comes from the dataset's documentation.

One-person speech, for the fine-tune mix and for pretraining. Libri-Light `small` is 577 h in a 35 GB tar and fits on a laptop. `medium` (5,193 h) is a 321 GB tar, so only use it if you have the disk.

```bash
mkdir -p raw && curl -L https://dl.fbaipublicfiles.com/librilight/data/small.tar | tar -x -C raw/
python prepare.py "raw/small/**/*.flac" --out data/mono && rm -rf raw/small
```

`prepare.py` prints how many times faster than real time it encodes. Use that to estimate how long the rest of the shards will take.

## 1. proof of concept (~30 min)

Fine-tune `caramel-1` on the first 10 hours of `data/stereo` with a small batch. The aim is not a better model. It's to see the whole loop work on this machine: data loads, the loss starts near the released model's level instead of ~7.6 (random), it moves, `best/` gets written, and the result generates audio. It's also where you read your real `frames/s`.

```bash
python train.py --init ./caramel-1 --data data/stereo --train_hours 10 \
    --fpb 1500 --accum 4 --steps 300 --warmup 30 --lr 3e-5 \
    --dev data/dev --eval_every 100 --save_best --out runs/poc
python generate.py --model runs/poc/best --seconds 20 --out poc.wav
```

10 hours is 450k frames, so 300 steps × 6k frames is about 4 passes over it. The training loss will drop below the dev loss: that's overfitting, and it's expected here.

## 2. fine-tune caramel-1 on Seamless Interaction (~1 day)

This is the useful run. It continues the released model on face-to-face conversations, with 20% one-person speech mixed in as in the original stage 2. It uses a fresh optimizer and a peak LR well below the original's 1.5e-4, so the model adapts without forgetting.

```bash
python train.py --init ./caramel-1 --data data/stereo data/mono --weights 0.8 0.2 \
    --fpb 1500 --accum 32 --steps 2000 --warmup 100 --lr 5e-5 \
    --dev data/dev --eval_every 250 --patience 2 --save_best --out runs/ft
python generate.py --model runs/ft/best --seconds 30 --bits 8 --out ft.wav
```

2,000 steps × 48k frames is about 2,100 hours of conversation. That's roughly half a pass over the full Seamless Interaction training set, or several passes over however many shards you've prepared. `--patience 2` stops the run once the dev loss has risen at two evals in a row, and `best/` always holds the lowest-dev-loss weights.

To stop and continue later, run with `--stop_at` and resume with `--resume runs/ft/ckpt`. Pass the same `--steps` so the LR schedule continues.

Seamless Interaction is CC-BY-NC 4.0, so weights fine-tuned on it are for non-commercial use.

## 3. pretrain from scratch (~2.5 days)

The original recipe at a laptop's scale. Stage 1 learns to speak from one-person speech, then stage 2 learns to converse. It won't reach `caramel-1`, which saw ~33B tokens, but it's the real pipeline end to end.

```bash
python train.py --data data/mono --fpb 1500 --accum 32 --steps 6000 --stop_at 2000 --out runs/s1
python train.py --data data/stereo data/mono --weights 0.8 0.2 --fpb 1500 --accum 32 --steps 6000 \
    --resume runs/s1/ckpt --dev data/dev --eval_every 500 --save_best --out runs/s2
python generate.py --model runs/s2/best --seconds 30 --out scratch.wav
```

Both stages share one LR curve over 6,000 steps, as stages 1 and 2 did in the original. Stage 1 alone is ~2,000 × 48k = 96M frames, about 3.7 passes over Libri-Light `small`.

## your own (synthetic) conversations

`prepare.py` takes conversations in three forms. Two-person data always becomes a width-16 store and one-person data a width-8 store. Keep them in separate `--out` folders and mix them with `--weights`.

1. **Stereo files** (`--stereo`): one conversation per file, speaker A on the left channel and speaker B on the right, time-aligned. Silence on a channel means that speaker isn't talking.
2. **One file per speaker** (`--pair_regex`): the two tracks of a conversation share a name part, captured by the regex's first group. For example `conv001_A.wav` and `conv001_B.wav` with `--pair_regex '(.+)_[AB]\.wav'`.
3. **A JSONL manifest** (`--manifest`): the natural output of a TTS pipeline. One conversation per line, made of separate utterances with a speaker and a start time in seconds:

```json
{"id": "conv001", "segments": [
  {"speaker": "A", "audio": "conv001/000_A.wav", "start": 0.00},
  {"speaker": "B", "audio": "conv001/001_B.wav", "start": 2.35},
  {"speaker": "A", "audio": "conv001/002_A.wav", "start": 2.90},
  {"speaker": "B", "audio": "conv001/003_B_mhm.wav", "start": 3.40}]}
```

`prepare.py` places every utterance on its speaker's track at `start` and leaves silence everywhere else. Relative paths are relative to the manifest. Any sample rate works; it's resampled to 24 kHz.

```bash
python prepare.py synthetic/*.jsonl --manifest --out data/synthetic
python train.py --init ./caramel-1 --data data/synthetic data/stereo --weights 0.5 0.5 ...
```

**Timing is what caramel learns from this, so make it realistic.** Real conversations have about 9% overlap, a median gap between turns of about 100 ms, and backchannels ("mm-hm") on the listener's track while the other person talks (see the table in the README). Synthetic dialogues where turns never overlap and always have a clean gap teach the model to wait its turn and talk like a TTS system. Put the overlaps and backchannels into the start times. Also keep conversations at least 60 s long: training windows are 60-120 s, and anything shorter is padded and wastes compute. `--min_seconds` drops clips shorter than its value (default 10 s).
