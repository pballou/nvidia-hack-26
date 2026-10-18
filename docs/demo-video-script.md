# Demo video script (target 3:00)

**The product is the navigation layer**: say or type what you want, and a small model
does it on any website, from the pixels on screen. The Acme Utilities portal is only
the stage we demo it on.

- **Narration** is what we say out loud; **On screen** is what we show while saying it.
- Times are targets; at a calm pace the narration below runs about 2:50.
- Numbers below are final (FINAL = SFT on Lightning-rewritten data + GRPO round 1; see
  `docs/results.md`). Don't say a number the results file doesn't show.

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

**On screen:** the console's run view: the browser runs headless, and each step's
screenshot of the Acme portal streams in with its click overlay. Push-to-talk.

**Narration (beat 1, voice request):**
> Here's a utility customer portal we built as a test app. I'll just ask.
> *(speaking)* "Download my September statement."

**On screen:** the transcript appears, then each step streams in: Lightning's plan in
plain words, the screenshot with the click overlay on the download icon, and the
green "verified by sandbox state" banner.

**Narration (beat 2, why that's hard):**
> That download button is an icon with no label in the page code, so a screen reader
> can't name it. Ours found it from the pixels.

**On screen (optional beat, clarify; cut first if long):** say "Pay something toward my
bill." Before the browser opens, the console asks "How much would you like to pay toward
your bill?"

**Narration (optional beat):**
> If a request is unclear, it asks before it touches anything.

**On screen:** second request: "Pay fifty dollars toward my bill." The plan streams in
(*Pay bill → Other amount → 50 → Continue*), the portal's own dialog shows **Amount
$50.00**, the agent clicks *Confirm payment*, and the verified banner appears.

**Narration (beat 3, money):**
> This one moves money, so it has to be exact. It picks "Other amount", enters fifty,
> and the portal's own dialog shows fifty dollars before it confirms. The check mark
> comes from the portal's state, not from the agent's word.

**On screen:** the same request ("Make the text bigger, I can't read this.") twice, using
the console's Grounder toggle: first `base`, which clicks "M"; then `grounder` (our
fine-tuned model), which clicks "L" and the text grows. Backup if the live base take
doesn't miss: cut to `docs/assets/base_vs_tuned_text_size.png` (real overlays: base
clicks M, tuned clicks L).

**Narration (beat 4, fine-tuning matters):**
> Here's where fine-tuning matters. On the same request, the base model kept
> clicking the wrong size. Ours gets it right.

## 4. How it works (1:35–2:25)

**On screen:** architecture diagram: microphone → Parakeet → Lightning (planner) →
grounder (2B VLM) → Playwright click → screenshot → back to Lightning.

**Narration:**
> Under the hood, speech goes to NVIDIA Parakeet, which transcribes it right here on
> this Mac in under a tenth of a second.
>
> Nemotron 3.5 Lightning is the planner. It reads a description of the screen,
> decides the next step, and calls tools like "click the download icon in the
> September row." Lightning never sees the image.
>
> The click target goes to our grounding model: a two-billion-parameter vision model
> we fine-tuned to answer one question, "where on this screenshot is that?", in a
> second or two on this Mac.
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
> 90.35%, and on icons it goes from 78 to 85%. On our held-out dev set, accuracy went
> from 76 to 87%.

## 5. Why it matters (2:25–2:50)

**On screen:** quick cuts: a bank site, a government form, an old intranet page, a
canvas dashboard. Overlay: "32.6B parameters total (model-card counts). Running on this
Mac."

**Narration:**
> Most software was never built with accessibility in mind, and most of it will never
> be rewritten. GUI assist doesn't need it to be. It works on the app you already
> have: customer self-service without the support call, and real access for people
> who struggle with a mouse and a crowded screen. And it's small: everything in this
> demo ran on this Mac, so it can run privately, on your own hardware.

## 6. Close (2:50–3:00)

**On screen:** logo; repo URL `github.com/pballou/nvidia-hack-26`; Hugging Face model
`patrickballou/gui-assist-grounder`.

**Narration:**
> Say what you want. GUI assist does the rest. Thanks for watching.

---

## Recording notes

- **Run the demo for real, fully local and headless.** Use `--agent real` with the
  FINAL grounder, per `docs/local-mac.md` "Fully local mode for the video"
  (`--vlm-url http://127.0.0.1:8011/v1 --summary-model base`). **Never `--headed`:**
  headed Chromium renders different pixels, and pay-50 fell from 8/8 headless to 1/4
  headed. Console defaults: thinking off (no 25–45 s thinking pauses; the Gym-measured
  config), clarify on. The mock agent is for rehearsal only. If a take fails, re-run
  rather than cut around it; "minimal cuts" is in the organizers' guidance.
- **Pre-check the demo tasks** (`download-september-statement`, `pay-50`,
  `text-size-large`; FINAL passed each 3/3 in Gym) right before recording. Spares, also
  3/3: `turn-on-autopay`, `report-outage-riverside`. Avoid `high-contrast` and
  `message-support-meter` (FINAL 0/3) and the canvas `max-usage-month` (1/3). For
  pay-50, check in Logs that the observation after "Pay bill" names "Full balance
  (selected)" and the Other-amount input.
- **Beat 4 is a live toggle:** `base` and `grounder` are served from the same
  vllm-metal instance, so flip the console's Grounder toggle. Base missed "L" 0/3 in
  Gym; if it happens to hit on the take, use the cutaway image
  `docs/assets/base_vs_tuned_text_size.png`. Switch back to `grounder` afterwards.
- **The clarify beat is optional.** Only the question is measured ("Pay something
  toward my bill" asks in the acceptance set); if you answer it on camera and go on to
  pay, dry-run that follow-up first.
- **Show the click overlay large.** It's the clearest evidence the model works from
  pixels.
- **Claims to keep honest:**
  - "A second or two" for grounding is the Mac: local p50 1.86 s (0.73 s on the
    H200). Whole-task time is much longer (pay-50 about a minute, text-size ~25 s), so
    don't imply tasks finish instantly.
  - "Runs on this Mac" is true for everything shown: Lightning (official BF16) on
    vllm-metal, Parakeet, and the grounder (base + FINAL LoRA in one vllm-metal
    instance; Gate G 87.0% local vs 87.0% H200 on 200 dev rows) on the M4 Max, 32.6B by
    model-card counts. Training ran on the H200; don't say it was trained on the Mac.
  - The "86.3%" base is Qwen3.5-2B (86.25%, 1,073/1,244). Ours is 90.35%: say
    "ninety point three five", not "ninety point four". Don't claim that tuning raised
    end-to-end task success (agent success is within noise), "ADA/EAA compliant", or
    that 20–40% deflection is measured.
