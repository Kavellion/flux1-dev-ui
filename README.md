# FLUX.1 [dev] UI

**Status: PC-proven.** Verified on a real PC (RTX 4070 12 GB, 16 GB system
RAM) on Oct 30, 2026: first picture (a lighthouse under storm clouds), a
10-image batch at 50 steps, and repeated 28-step runs — all on the default
Q4_K_S GGUF path with CPU offload, no OOM, no silent exit. **28 steps is
the model's own reference setting and the recommended standard**; 50 steps
mostly pays time (about 2.1 s/step at 1024x1024 on the test PC).

A simple, standalone Gradio app for Black Forest Labs' **FLUX.1 [dev]**
(`black-forest-labs/FLUX.1-dev`) — the 12B guidance-distilled model. One
model, one page: prompt, negative prompt, size, steps/guidance, a prompt
queue, history, Stop, Star, and Unload. Same shape as the rest of the
Kavellion UI family (Kandinsky, Sana, Z-Image, SD 3.5, PixArt).

> **License split, read first.** The *model* is under the
> **FLUX.1 [dev] Non-Commercial License** and is **gated** on Hugging Face.
> The *UI code* in this package is MIT (Alan Carmack). Outputs and use must
> follow the model license.

## Before the first run — the gated-model steps

1. Sign in to Hugging Face and open `black-forest-labs/FLUX.1-dev`.
   Accept the FLUX.1 [dev] Non-Commercial License there.
2. Create a Hugging Face **read** token (Settings → Access Tokens).
3. Start the app (below), paste the token into the app's token box, then
   Generate. The token is used for that session's downloads only — the app
   never writes it to settings.txt, a status line, or any file. Leaving the
   box empty instead uses a Hugging Face login already saved on the PC, if
   one exists.

Python is **included** in this package (`python_embeded/`, Python 3.12,
ComfyUI-portable style). There is nothing to install first and no system
Python is touched or needed. Python 3.12 is required because Diffusers 0.42
dropped Python 3.10.

## Install and run (Windows, NVIDIA GPU)

1. Unzip anywhere, e.g. `C:\AI\flux1-dev`.
2. Double-click `install.bat` once. It sets up pip in the bundled Python,
   installs PyTorch (CUDA cu130 build), then the pinned packages in
   `requirements.txt`, all into the bundled Python only.
3. Double-click `run.bat`. A browser page opens on 127.0.0.1.
4. First Generate downloads the model (see sizes below) into the Hugging
   Face cache on your user profile and loads it once; later generations
   reuse the loaded model.

Pictures land in `images_out/` next to `run.bat` — one subfolder per batch
name, reused across runs, frames numbered continuously
(`000001.png`, `000002.png`, …), with a `settings.txt` block appended per
run (date, time, frame range, model, precision, GGUF file, offload, size,
steps, guidance, true CFG, prompt, negative prompt, expanded prompts,
seeds). The token is never recorded.

## How it fits on a 12 GB card — the loader

FLUX.1 [dev] in full BF16 is about **33.9 GB** of weights (transformer
23.80 GB, T5-XXL text encoder 9.53 GB, CLIP 0.25 GB, VAE 0.34 GB). That
fits neither a 12 GB card nor 16 GB of system RAM, so this app does
**not** default to BF16.

| Precision choice | Transformer file | Total weight bytes | Notes |
|---|---|---|---|
| **GGUF Q4_K_S (default)** | `flux1-dev-Q4_K_S.gguf`, 6.81 GB | ≈16.9 GB split by offload | Recommended for 12 GB / 16 GB |
| GGUF Q5_K_S | `flux1-dev-Q5_K_S.gguf`, 8.29 GB | ≈18.4 GB split by offload | Better detail, tighter |
| GGUF Q6_K | `flux1-dev-Q6_K.gguf`, 9.86 GB | ≈19.9 GB split by offload | Tightest 12 GB fit; bigger cards preferred |
| Full BF16 | repo shards, 23.80 GB | ≈33.9 GB | 24 GB+ cards only — expected to OOM on 12 GB / 16 GB |

The transformer loads from the GGUF repo `city96/FLUX.1-dev-gguf` through
`FluxTransformer2DModel.from_single_file` with a `GGUFQuantizationConfig`
(the pattern in the Diffusers GGUF documentation); encoders, VAE,
tokenizers, and scheduler load from the gated base repo with the pipeline —
which is why the license + token steps are needed even though the GGUF
file itself is ungated. Q8_0 (12.71 GB) is deliberately **not** offered:
the file alone exceeds a 12 GB card before activations and VAE decode.

**CPU offload** defaults ON (`enable_model_cpu_offload`), so only one
stage's weights sit on the GPU at a time: while denoising, the encoders
park in RAM (~10.1 GB — the same T5 parking the SD 3.5 UI has already run
on this PC class); while encoding, the transformer parks in RAM instead.
VAE tiling is enabled as a further decode-time saver. Memory is freed
after every image inside the generate loop, not only at unload.

If a chosen GGUF load fails, the app falls back to the Q4_K_S file with
offload forced ON and says so in the status line. It never falls back to
full BF16 by itself — on a 12 GB / 16 GB PC that would be an
out-of-memory trap, not a fallback. First-load is the RAM-tightest
moment (all components assemble once before offload stages them); if
Windows is low on memory, close other apps for the first load.

First download on the default path is ≈17 GB (6.81 GB GGUF transformer +
9.53 GB T5 + 0.25 GB CLIP + 0.34 GB VAE + configs/tokenizers).

## Defaults and controls

- Steps **28**, guidance **3.5** — the model card's own values (and the
  real `FluxPipeline` defaults in Diffusers 0.42.0).
- **True CFG** slider, default **1.0 (off)**. FLUX.1 [dev] is
  guidance-distilled, so the **negative prompt does nothing until True
  CFG is above 1**. Raise it (try 3–4) to make the negative genuinely
  steer; it costs an extra transformer pass per step.
- Size presets 1024×1024 / 768×1024 / 1024×768 / 768×768 or custom
  (rounded to a multiple of 16, the pipeline's own granularity).
- Seed box (-1 = fresh random per image; a fixed seed walks S, S+1, …),
  1–10 images per prompt, a prompt queue (one prompt per line; the queue
  beats the prompt box), single-level dynamic prompts `{a|b|c}` in both
  the prompt and the negative, Random prompt, session History (last 10),
  Stop, Star batch (copies the last batch into `images_out/favorites/`),
  and Unload models.
- Settings (`settings.txt`) records everything above per run — never the
  token.

## Not in v1

- **No LoRA slots.** FLUX LoRAs exist in PEFT format and `FluxPipeline`
  can load them against the BF16 transformer, but LoRA-on-GGUF under CPU
  offload is not a verified combination in this build — v1 leaves the
  slot out rather than shipping one that might silently do nothing.
- No img2img/inpaint, no ControlNet, no animation, no upscaling.

## Verification status

- Pipeline facts verified against the Diffusers **0.42.0 wheel source**
  (`FluxPipeline.__call__` defaults and `check_inputs`: non-multiple-of-16
  sizes only warn and resize; a both-prompt-and-embeds call raises) and
  against the live Hugging Face API (repo file lists and byte sizes for
  the base repo and the GGUF repo, gating = auto, license = the FLUX.1
  [dev] Non-Commercial License in `LICENSE.md`).
- Stub tests mirror those real defaults and that validation; the bundle
  passed a fresh-folder cold test (zip integrity, entry parity with the
  sibling Z-Image Base bundle layout, CRLF bats with no parentheses in
  echo lines, `python_embeded/python.exe` and `app.py` present, extracted
  `app.py` compiles).
- **PC-proven Oct 30, 2026** (RTX 4070 12 GB, 16 GB RAM): clean install
  from this bundle, first picture on the Q4_K_S path (status line named
  `flux1-dev-Q4_K_S.gguf`), a 10-image batch at 50 steps in one session,
  and five further images at 28 steps. First-load memory assembly — the
  tight moment on 16 GB — passed.
- Two honest notes from that testing. (1) FLUX's small CLIP encoder caps
  at 77 tokens, so long prompts print a truncation warning; that warning
  is only about the CLIP side — the T5 encoder still hears the full
  prompt. (2) For long unattended runs, minimize the app's terminal
  window: a focused terminal can catch a stray keystroke event and close
  the app (observed once, on a run whose terminal sat open; every image
  already generated was safely saved).

## Example output

![Porcelain portrait](examples/example-porcelain-1.png)

*The family's classic porcelain-cyborg test prompt through FLUX.1 [dev] —
note how photographic its read is next to the other UIs. 768x1024,
Q4_K_S, 28 steps.*

![Porcelain portrait 2](examples/example-porcelain-2.jpg)

*Same prompt, another seed from the first morning's runs.*
