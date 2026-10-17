# Demo video script (target 3:00)

**The product is the navigation layer**: say or type what you want, and a small model
does it on any website, from the pixels on screen. The Acme Utilities portal is only
the stage we demo it on.

- **Narration** is what we say out loud; **On screen** is what we show while saying it.
- Times are targets; at a calm pace the narration below runs about 2:50.
- `[TODO]` marks numbers to update from `docs/results.md` before recording (final
  SFT/GRPO numbers land overnight). Don't say a number the results file doesn't show.

---

## 1. Hook (0:00–0:20)

**On screen:** someone squinting at a cluttered account page; cursor hunting for a
tiny icon. Then the team, on camera, briefly.

**Narration:**
> Every website makes you learn where its buttons are. For people with low vision,
> limited mobility, or just no patience for a confusing portal, that's the whole
> problem. We're team sherpas: Patrick, Kyle and Charlie. We built GUI assist.

## 2. What it is (0:20–0:40)

**On screen:** title card: "Say what you want. It clicks for you. Any website, from
pixels alone."

**Narration:**
> GUI assist is a navigation layer for any app. You say what you want, and it does it,
> by looking at the screen the way you do. No API, no integration, no access to the
> page's code. If you can see it, it can click it.

## 3. Live demo (0:40–1:35)

**On screen:** the console beside the Acme portal (`--headed`). Push-to-talk.

**Narration (beat 1, voice request):**
> Here's a utility customer portal we built as a test app. I'll just ask.
> *(speaking)* "Download my September statement."

**On screen:** the transcript appears, then each step streams in: Lightning's plan in
plain words, the screenshot with the click overlay on the download icon, and the
green "verified by sandbox state" banner.

**Narration (beat 2, why that's hard):**
> That download button is an icon with no label in the page code. A DOM-based agent
> can't find it. Ours found it from the pixels.

**On screen:** second request: "Which month did I use the most power? Open it." The
click lands on the July bar of a canvas chart.

**Narration (beat 3, canvas):**
> And this chart is drawn on a canvas. There's no element to select at all, just
> pixels. It picks July, the peak month, and opens it.

**On screen:** split screen of the same request ("Make the text bigger, I can't read
this.") with the base grounder vs the fine-tuned grounder. Base clicks "M"; tuned
clicks "L" and the text grows.

**Narration (beat 4, fine-tuning matters):**
> Fine-tuning is what makes this work. On the same request, the base model kept
> clicking the wrong size. Ours gets it right.

## 4. How it works (1:35–2:25)

**On screen:** architecture diagram: microphone → Parakeet → Lightning (planner) →
grounder (2B VLM) → Playwright click → screenshot → back to Lightning.

**Narration:**
> Under the hood, speech goes to NVIDIA Parakeet, which transcribes it in about a
> tenth of a second.
>
> Nemotron 3.5 Lightning is the planner. It reads a description of the screen,
> decides the next step, and calls tools like "click the download icon in the
> September row." Lightning never sees the image.
>
> The click target goes to our grounding model: a two-billion-parameter vision model
> we fine-tuned to answer one question, "where on this screenshot is that?", in under
> a second.
>
> Playwright performs the click, takes a new screenshot, and the loop repeats until
> the task is done.

**On screen:** NeMo Gym logos; the data pipeline (OS-Atlas → cleaning → Lightning
rewrites → SFT → GRPO); a results table.

**Narration:**
> We trained on open GUI data from OS-Atlas. Lightning rewrote raw accessibility labels
> into the way people actually ask for things. We trained with LoRA in TRL, then ran
> reinforcement learning with rewards from NVIDIA NeMo Gym, all on a single H200
> through Brev.
>
> Along the way we found that part of the standard ScreenSpot benchmark was captured
> on the same machines as our training data, so we excluded those rows and report on
> a clean test set.
>
> On ScreenSpot-v2-clean, the base model scores 86.3%; our tuned model scores
> `[TODO final]`. On our held-out dev set, a five-minute pilot already moved accuracy
> from 76 to 81%, and its answers got twice as short, so every click is faster.

## 5. Why it matters (2:25–2:50)

**On screen:** quick cuts: a bank site, a government form, an old intranet page, a
canvas dashboard. Overlay: "~33B parameters total. One GPU."

**Narration:**
> Most software was never built with accessibility in mind, and most of it will never
> be rewritten. GUI assist doesn't need it to be. It works on the app you already
> have: customer self-service without the support call, and real access for people
> who struggle with a mouse and a crowded screen. And it's small: the whole system
> fits on one GPU, so it can run privately, on your own hardware.

## 6. Close (2:50–3:00)

**On screen:** logo; repo URL `github.com/pballou/nvidia-hack-26`; Hugging Face model
`patrickballou/gui-assist-grounder`.

**Narration:**
> Say what you want. GUI assist does the rest. Thanks for watching.

---

## Recording notes

- **Run the demo for real.** Use `--agent real --headed` with the tuned grounder;
  the mock agent is for rehearsal only. If a take fails, re-run rather than cut
  around it; "minimal cuts" is in the organizers' guidance.
- **Pre-check the three demo tasks** (`download-september-statement`,
  `max-usage-month`, `text-size-large`) on the final checkpoint right before
  recording. The base-vs-tuned split screen needs the base grounder to actually miss,
  as it did in the Saturday run; if it no longer does, drop beat 4 and say the dev
  and SSv2 numbers instead.
- **Show the click overlay large.** It's the clearest evidence the model works from
  pixels.
- **Claims to keep honest:**
  - "Under a second" for grounding is the measured p50 (0.73 s). Whole-task time is
    much longer (about a minute at p50), so don't imply tasks finish instantly.
  - Only say "runs locally on a laptop" if the Mac-local check in `docs/local-mac.md`
    passed for the full stack on recording day; otherwise keep "fits on one GPU".
  - The "86.3%" base number is Qwen3.5-2B. If the team switches the base model (the
    Qwen3-VL comparator scored 90.4% zero-shot), update both numbers together.
