A continuous multichannel mask is essentially a stack of 2D spatial maps, where each pixel contains continuous numerical values instead of just binary (0/1) labels.

---


❯ so what do you think about moving forward? seems like classification is decent enough to keep as it seits near/within floor/ceiling from tornet model but is now setup specifically for future/early detection, but not sure about regression because I was thinking about maybe dropping and focusing on classification because seems like you said promising but kind of exhausted current stuff so maybe like you said it would benefit from combining with HRRR data we have, because dropping it fully doesn't seem like right call because seems like some of that stuff might have or could potentially help the classification head right?

Agreed on both counts, and your worry about dropping the regression branch is backed by the numbers.

Keep the decoder, but as a training signal, not as a product. The runs already contain the ablation. The control (plain L1, no detector term) head scored 0.721 / 0.741. The detector-config head scored 0.739 / 0.757. The 12ch head 0.747 / 0.753. So the detector-aware drawing loss lifts the head by about 0.016 at both horizons, in both configs where it is present. That is the shared encoder learning from "draw a future the detector recognises", and it is the whole reason the head is above the floor instead of below it. Dropping the decoder would likely give the control head back. What to drop is the indirect route as a predictor: stop scoring and iterating the drawn forecast, keep its MAE and judge numbers as a one-line health check.

Two things get simpler once the head is the target:

- The checkpoint rule is right as is. Selecting on validation head AUC was only misaligned for the route we are retiring. No route-aware rule needed.
- The loss design freezes at fold_b_detector. No more anchor, channel-scope or transformer variants.

Where the HRRR combination has to happen. HRRR starts 2014-07-30, so half of the fold B test year has no environment data. The meta model only pays on the full dataset, which is where Experiment 7 got +0.054. That makes the next step:

1. Small code: a score-only mode and an export of the head's logits and pooled features per sample, the same table shape the meta model already reads.
2. Fine-tune the detector config on the full dataset on a pod. Rough estimate: the full training set is about six times fold B, so around 3.5 to 4 hours plus scoring, roughly $12 to $15 at the usual rate. One run, one seed.
3. Feed its head outputs into the Experiment 7 meta model next to the HRRR table and score 2021 to 2022 pre-onset cases. That is the question "does a backbone tuned for onset plus environment beat either alone", and it runs in minutes on cached tables.

If that combination moves the needle, longer sequences from the temporal dataset are the next radar-side lever. If it does not, we have learned that the remaining signal is environmental and the radar side is done.

I have not written any of this into the docs yet. Say go and I will add it as the Step 3 plan and build the export and score-only pieces, no pod until you have seen the estimate confirmed.
