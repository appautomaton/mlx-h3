# MiniMax-H3 architecture

Researched 2026-08-04. Facts come from repository primary sources (`config.json`,
`LICENSE`, source files) wherever possible; vendor claims are labelled as such.

## Naming

**H3 is not M3.** Two MiniMax releases from 2026 that are constantly conflated online:

| | What it is | When |
|---|---|---|
| **MiniMax-H3** (Hailuo 3.0) | audio-video generation model | API 2026-07-31, weights **2026-08-03** |
| MiniMax-M3 | text/agentic LLM, 428B MoE, 1M context | 2026-06-01 |

This project only concerns H3.

## Three stages, only the middle is open

| Stage | Role | Open |
|---|---|---|
| H3-Context-IR | multimodal input → shot-level structured IR | No, API only |
| **H3-Base** | 768p joint audio-video generation | **Yes** |
| H3-Regenerate-2K | 768p → 2K | No, API only; promised later |

Two common misreadings:

**Context-IR expands, it does not compress.** IR = Context Intermediate Representation.
In the official reproducible sample, a one-line prompt becomes a long description carrying
shot breakdowns, timestamps (`At 00:04.500`) and three separate fields for imagery,
soundscape and score. Reported `usage` is `prompt_tokens: 5650, total_tokens: 8565`.

MiniMax states Context-IR is "critical to the quality of the final output". Feeding H3-Base
a plain-language prompt locally will visibly underperform the official demos — not because
the model is worse, but because this stage is missing. The substitute is to build your own
against `docs/VIDEO_PROMPT_WRITING_GUIDE_{base,ref}_en.md`.

**Regenerate-2K is not super-resolution.** From the README:

> instead of using a conventional dedicated super-resolution module, we use the H3 base model
> to regenerate its own low-resolution result through an in-context manner

The 768p result plus the original multimodal context are fed back into H3 itself. This lets
it recover small text and fine detail that conventional SR would have to hallucinate.

Implication: this stage runs on the already-open H3-Base weights; only the outer orchestration
is withheld. Hence the wording "not **yet** open-sourced, we will release it once it is ready",
which differs from how Context-IR is described.

## H3-Base

From `transformer/config.json` (measured, not paraphrased):

```
MiniMaxH3Transformer3DModel
  num_layers            50        + num_refiner_layers 2
  hidden_size           5376
  num_attention_heads   56        attention_head_dim 128
  ffn_dim               14336
  in_channels           24        <- video latent
  audio_in_channels     32        <- audio latent
  patch_size            [1, 2, 2]
  text_dim              5120
  rope_theta            10000.0   rope_freq_dim 16
```

33B dense (not MoE), roughly 13B of which sits in AdaLN branches.

### Four properties that shape any implementation

1. **Joint denoising.** One transformer runs full self-attention over a *single packed
   sequence* holding text conditioning, conditioning image/video/audio rows, target audio
   rows and target video rows simultaneously. No separate vocoder, no audio post-pass.

2. **Two schedulers, one forward.** Video and audio latents step down different sigma
   schedules (`scheduler` shift=12.0, `audio_scheduler` shift=3.0), but the transformer is
   called exactly once per step.

3. **Guidance is distilled into the weights.** No guider, no `negative_prompt`, no
   `guidance_scale`; strictly one forward pass per step.

4. **The text encoder is only used to half its depth.** The conditioner is Qwen3-VL-32B, but
   H3 reads the *unnormalized* hidden state after decoder layer 50, not the last one, and the
   lm_head is unused. The model has 64 layers — see `01-weights-and-starting-point.md`.

### Components

| Component | Spec |
|---|---|
| H3-Encoder | Qwen3-VL-32B, 64 layers, hidden 5120, 64 heads / 8 KV heads (GQA), 27-layer vision tower |
| H3-VisualVAE | f16t4d24 — 16x spatial, 4x temporal compression, 24 latent channels, ViT decoder |
| H3-AudioVAE | 32 kHz stereo, L/R independent, compressed to 40 Hz latents. DAC + BigVGAN lineage |

## Output constraints

- 4–15 seconds (diffusers docs say 5–15), 24 fps
- `num_frames` rounds up to the next `17n + 5`
- Short edge defaults to 768; `height`/`width` must be multiples of 32
- 32 kHz stereo audio
- Aspect ratios 21:9, 16:9, 4:3, 1:1, 3:4, 9:16
- 11 languages with stable support

## Two variants

| | Conditioning input | Weights |
|---|---|---|
| **FL2VA** | 0–2 keyframes (first / last / both) | `transformer/` |
| **Ref2VA** | up to 9 images + 3 videos (2–15 s each) + 3 audio clips, 12 files max | `transformer_ref/` |

**Structurally identical; they differ only in input packing.** Everything else — text encoder,
both VAEs, tokenizer, schedulers — is shared.

Ref2VA's `references` list is order-sensitive: order determines the `<Picture 1>` / `<Audio 1>` /
`<Video 1>` labels in the prompt presentation and advances the shared audio/video rotary clock.
Reordering the same references is a different request.

## Sources

- https://huggingface.co/MiniMaxAI/MiniMax-H3
- https://github.com/huggingface/diffusers/pull/14355 — best-documented integration
