# Prompting H3-Base

How to write the text this runtime sends to the encoder. Sourced from the released
Context-IR sample, MiniMax's published field vocabulary, and the community brief
specification; hosted-service claims are labelled as such.

## The API writes the brief. Here you write it

H3 ships as three stages and [only the middle one is open](architecture.md). Stage one,
H3-Context-IR, takes a one-line request and expands it into a structured brief: shot
breakdowns, timestamps, and separate fields for imagery, soundscape, and score. The
official sample reports `prompt_tokens: 5650` for a single-sentence input.

That stage is API-only. `mlx-h3` loads H3-Base, so the text encoder sees exactly what you
pass and nothing rewrites it first. **The brief is the interface.** A one-line prompt is
not wrong, it is just the un-expanded input that the hosted product would have expanded
for you.

This is also why hosted prompting guides transfer imperfectly. They describe a pipeline
whose first stage rewrites your text and whose last stage upscales the result.

## Two brief shapes

| | Fields | When |
|---|---|---|
| **Text-only** | 3 | no media attached |
| **Reference** | 6 | any image, video, or audio attached |

**The media present decides the shape, not what the text claims.** Attaching one image
moves you to the reference brief for the whole request. The two are not mixed.

Text-only, in order: `integrated_multimodal_description`, `overall_soundscape`,
`non_diegetic_music`.

Reference, in order: `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`.

Each is written as the field name, a colon, then its content.

## Rules shared by both shapes

### Shots and timing

Open with `[Shot 1]` and no timestamp. Later shots carry one: `[Shot N] At MM:SS.mmm`,
strictly increasing, all inside the target duration. Ordinary cuts read as "cuts to",
"transitions to", "switches to"; name a cross-dissolve, fade, or wipe only when you want
one.

One primary change per beat, ending in something a viewer could point at. Budget about
four seconds for a prop change or a hand-off.

### Camera

**H3 drifts and reframes on its own unless told otherwise.** This is the single behaviour
most worth pinning.

| | |
|---|---|
| Push | Zoom In/Out, Push In/Pull Out |
| Lateral | Pan Left/Right, Truck Left/Right |
| Vertical | Tilt Up/Down, Pedestal Up/Down |
| Path | Arc Shot, Tracking Shot, POV |
| Hold | Static Shot |
| Texture | Shake Slightly/Strongly, Roll Clockwise/Counterclockwise |

For a locked frame, say the frame never moves *and* refuse the alternatives — no pan, no
push-in, no reframing. There is no negative-prompt field, so refusals are plain sentences
in the body, and they earn their place mainly against the two things the model adds
unprompted: camera movement and on-screen text.

### Speech

Every speaker gets a stable ID — `(S1)`, `(S2)` — assigned in the order voices actually
occur and reused at every later event. Establish age, gender, timbre, and accent outside
the tag on first appearance.

Speech goes inside `<d>[Language] words.</d>`, with punctuation preserved verbatim. Eleven
languages are tagged: Arabic, Chinese, English, French, German, Italian, Japanese, Korean,
Portuguese, Russian, Spanish. Use `[unclear]` for unintelligible spans.

Voiceover is marked as off-screen, followed by a statement that the lips stay closed.
Dialogue crossing a cut takes `<scenetrans>` at both points plus a note on audio
continuity; truncated speech takes `<cutoff>`. When speech ends, say the lips close.

### On-screen text

Type the words that must be readable, in quotes, rather than describing them. Name the
typographic treatment (condensed, all-caps, serif) and the placement (centred, lower
third).

## The text-only brief

`integrated_multimodal_description` carries the whole timeline. Open it with the style and
initial composition — cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor,
vintage film — taken from the request's own wording.

`overall_soundscape` is one paragraph, one to four sentences: ambience, physical action,
and non-verbal human sound. Wind, rain, traffic, footsteps, fabric, impacts, breathing,
laughter, panting. **Not** dialogue, singing, or music a character can hear — those live in
the timeline. `N/A` only for requested silence.

`non_diegetic_music` is one to three sentences of audience-only score: instrumentation,
tempo, rhythm, dynamics. No mood words. Music from a radio or a live performance is
diegetic and belongs in the timeline instead.

## The reference brief

### Labels

References are named by input position, numbered independently per category:
`<Picture N>`, `<Video N>`, `<Audio N>`. `<Subject N>` is different — it denotes reusable
visible content in the *target*, not a source file.

One subject may draw on several assets, and one asset may supply several subjects. The
same video can be `<Video 1>` and `<Audio 2>` at once.

Create a standalone `<Picture N>` only when the image anchors a frame — first, last,
keyframe, edited keyframe, or composition. If it only defines a character, scene, costume,
or style, cite it inside the `<Subject N>` that uses it.

Ordering is semantic. [The rotary clock advances on reference
order](architecture.md#two-variants), so the same references in a different order are a
different request.

### Task-type prefix

`summary` opens with a bracketed prefix joined by ` + `, drawn from: keyframe completion,
reference generation, video editing, video continuation, audio reuse, audio reference.

**Presence of a file does not create a task type.** A reference video supplying only camera
movement, cuts, or rhythm is reference generation — not editing, not continuation.

### Retention markers

`retention_analysis` gives one line per label, each with exactly one marker and a short
justification. Two fixed vocabularies:

| Visible content | Audio |
|---|---|
| `fully_preserved` | `fully_copy` |
| `partially_preserved` | `partially_copy` |
| `attribute_transfer` | `reference` |
| `weak_reference` | `weak_reference` |

Marker choice stays inside the role already declared for that label. New actions or
backgrounds in the target are not losses of fidelity. Speaker IDs do not appear here.

### Body

`detailed_description` is the main body, written shot by shot in playback order, with
reference labels inserted where their roles take effect. Establish style in one or two
sentences before `[Shot 1]`. Generation tasks normally run 350–500 English words; editing
scales with the source instead.

Frame anchors develop in one of three directions:

| Anchor | Path |
|---|---|
| First frame | begin there, write the motion that *leaves* it |
| First and last | interpolate between them, land exactly on the last at its stated time |
| Last frame only | infer a plausible earlier state, converge onto the image in the final shot |

**Name garments explicitly, even for referenced subjects** — wardrobe drifts across
generations otherwise. Translate emotion into observable behaviour: where the eyes go, what
the hands do, what stays still.

## What this runtime actually produces

`align_frame_count` rounds up to the next `n % 17 == 5`, so a requested duration is a
request, not a result:

| Asked | Frames | Actual |
|---|---|---|
| 4 s | 107 | 4.458 s |
| 8 s | 192 | 8.000 s |
| 10 s | 243 | 10.125 s |
| 15 s | 362 | 15.083 s |

362 frames is the ceiling; above it the pipeline raises rather than truncating. The
default is 56 frames — already aligned, 2.333 s.

**Hosted 2K figures do not apply here.** Upscaling is H3-Regenerate-2K, the third stage,
which is not open. This runtime is H3-Base at 768p, capped at `MAX_PIXELS = 768 * 1344`,
about 1.0 MP. Defaults are 864×480.

Published duration ranges disagree — MiniMax says 4–15 s, diffusers says 5–15 s, hosted
guides repeat one or the other. The frame rule above is what this runtime enforces.

## Passing the brief

A brief long enough to be worth writing is long enough to fight your shell. Pass exactly
one of the positional prompt or `--prompt-file`; supplying both, or neither, is an error.
The file is read as UTF-8 with the trailing newline stripped.

```sh
uv run mlx-h3 --prompt-file brief.txt --frames 243
```

Keep briefs outside the repository. Filenames containing `.prompt.` or `.case.`, and any
file carrying a field name in its assigned form, are rejected by
`dev/check_public_tree.py`.

## Sources

- https://huggingface.co/MiniMaxAI/MiniMax-H3 — released fields and language list
- https://fal.ai/learn/devs/minimax-h3-prompting-guide — hosted-pipeline guide; its
  resolution and duration ceilings describe the full three-stage product
- https://gist.github.com/Naxdy/43b7422a1e4a79fb8b0489c6c39eaace — brief specification
