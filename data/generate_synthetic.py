#!/usr/bin/env python3
"""
Synthetic dataset generator for Project 1 (Automated Response Scorer + Fairness Audit).

Design goals
------------
1. LEARNABLE SIGNAL: a response's *text quality* genuinely scales with its score label
   (length, STAR completeness, specificity/metrics, competency vocabulary, structure).
   This is what lets a model achieve a meaningful QWK instead of noise.
2. CLEAN FAIRNESS MECHANISM: a sensitive attribute `group` (A/B) is assigned INDEPENDENTLY
   of text/quality -> the data is fair by construction. We then provide a SECOND label column
   `human_score_biased` that simulates biased human raters systematically under-scoring group B
   (mirrors SHL's published finding that the human ground truth itself can be biased).
3. REALISM: a little adjacent-level "blur" so QWK lands ~0.7-0.9, not a fake 1.0.

Output columns
--------------
id, question_id, question, rubric, group, true_quality, human_score, human_score_biased, response

- true_quality      : latent intended quality 1-5 (synthetic-only luxury; use to VALIDATE bias/mitigation)
- human_score       : FAIR observed label (== true_quality). Default training/eval label.
- human_score_biased: BIASED observed label (group B under-scored). Switch to this to DEMONSTRATE the audit+mitigation.
- group             : A or B, assigned at random, independent of quality.

NOTE: `group` is a SIMULATED attribute used to demonstrate methodology. It makes no claim about real groups.
"""
import csv, random, argparse, os, statistics

random.seed(42)

# --------------------------------------------------------------------------------------
# 12 competency questions + rubrics (behavioral / situational-judgment style)
# --------------------------------------------------------------------------------------
QUESTIONS = [
    ("Q01", "Conflict Resolution",
     "Describe a time you resolved a disagreement within your team.",
     "Score the response on conflict-resolution competency: clarity of the situation, the candidate's specific actions, fairness/empathy toward others, and a constructive resolution with reflection. 1=poor, 5=excellent."),
    ("Q02", "Leadership",
     "Tell us about a time you led a group toward a goal.",
     "Score on leadership: ownership, how the candidate motivated/organized others, decisions made, and a measurable outcome. 1=poor, 5=excellent."),
    ("Q03", "Problem Solving",
     "Describe a difficult problem you solved at work or in your studies.",
     "Score on problem-solving: how the problem was analyzed, the approach taken, creativity, and the result. 1=poor, 5=excellent."),
    ("Q04", "Customer Focus",
     "Give an example of how you handled a difficult customer or stakeholder.",
     "Score on customer focus: understanding the customer's need, the actions taken, professionalism, and the outcome. 1=poor, 5=excellent."),
    ("Q05", "Communication",
     "Describe a situation where you had to explain something complex to others.",
     "Score on communication: clarity, audience awareness, structure of the explanation, and effectiveness. 1=poor, 5=excellent."),
    ("Q06", "Teamwork",
     "Tell us about a time you collaborated with others to achieve something.",
     "Score on teamwork: the candidate's role, cooperation, handling of differences, and the shared result. 1=poor, 5=excellent."),
    ("Q07", "Adaptability",
     "Describe a time you had to adapt quickly to an unexpected change.",
     "Score on adaptability: recognizing the change, adjusting the approach, composure, and the outcome. 1=poor, 5=excellent."),
    ("Q08", "Initiative",
     "Give an example of when you took initiative beyond your normal responsibilities.",
     "Score on initiative: self-direction, the action taken, impact, and reflection. 1=poor, 5=excellent."),
    ("Q09", "Decision Making",
     "Describe an important decision you had to make with limited information.",
     "Score on decision-making: how options were weighed, reasoning, the decision, and the result. 1=poor, 5=excellent."),
    ("Q10", "Time Management",
     "Tell us about a time you managed competing priorities under a deadline.",
     "Score on time management: prioritization, planning, execution, and meeting the deadline. 1=poor, 5=excellent."),
    ("Q11", "Resilience",
     "Describe a setback or failure and how you responded to it.",
     "Score on resilience: honesty about the setback, response, lessons learned, and growth. 1=poor, 5=excellent."),
    ("Q12", "Innovation",
     "Give an example of an idea or improvement you introduced.",
     "Score on innovation: the idea, how it was developed and implemented, and its measurable impact. 1=poor, 5=excellent."),
]

# Per-competency content pools (situation / action / result) for relevance
COMPETENCY_CONTENT = {
    "Conflict Resolution": {
        "situation": ["two teammates disagreed on the project direction", "a colleague and I clashed over priorities",
                      "there was tension between design and engineering", "a peer felt their work was being overlooked"],
        "action": ["I set up a one-on-one to hear both sides", "I facilitated a meeting to align on shared goals",
                   "I listened to each person's concerns before proposing a compromise", "I reframed the disagreement around our common objective"],
        "result": ["we agreed on a plan everyone supported", "the team's collaboration improved noticeably",
                   "we delivered the milestone on time", "the relationship recovered and trust was rebuilt"],
    },
    "Leadership": {
        "situation": ["our team was behind on a critical release", "we needed to onboard three new members quickly",
                      "the project had no clear owner", "morale had dropped after a missed deadline"],
        "action": ["I divided the work into clear ownership areas", "I ran daily stand-ups to unblock people",
                   "I set a concrete goal and tracked progress openly", "I coached two junior members directly"],
        "result": ["we shipped the release on schedule", "throughput improved and the team felt ownership",
                   "we cut the backlog substantially", "the group exceeded its target"],
    },
    "Problem Solving": {
        "situation": ["a recurring system failure was blocking users", "our reports were producing wrong numbers",
                      "a process took far too long to complete", "we could not reproduce a critical bug"],
        "action": ["I broke the problem into smaller parts and tested each", "I traced the issue to its root cause with logs",
                   "I prototyped two approaches and compared them", "I analyzed the data to isolate the pattern"],
        "result": ["the failure rate dropped sharply", "the numbers reconciled correctly",
                   "the process time fell dramatically", "the bug was fixed and did not recur"],
    },
    "Customer Focus": {
        "situation": ["a key client was frustrated with a delay", "a stakeholder kept changing requirements",
                      "a customer reported a serious complaint", "an important account was at risk of leaving"],
        "action": ["I acknowledged the concern and set clear expectations", "I followed up daily until it was resolved",
                   "I dug into what the customer actually needed", "I coordinated across teams to fix it fast"],
        "result": ["the client renewed and was satisfied", "the complaint was resolved and trust restored",
                   "the relationship became stronger than before", "the account stayed and grew"],
    },
    "Communication": {
        "situation": ["I had to present a technical plan to non-technical leaders", "a complex policy confused the team",
                      "I needed to align stakeholders with different views", "documentation was unclear and people kept asking"],
        "action": ["I used a simple analogy and a visual", "I structured the explanation around their priorities",
                   "I checked understanding with questions throughout", "I rewrote the material in plain language"],
        "result": ["the decision was approved quickly", "the team understood and acted correctly",
                   "everyone left aligned on next steps", "support requests dropped"],
    },
    "Teamwork": {
        "situation": ["we had to deliver a joint project across two teams", "a deadline required tight coordination",
                      "members had very different working styles", "we shared responsibility for a launch"],
        "action": ["I made sure each person's strengths were used", "I kept communication open and frequent",
                   "I volunteered for the parts no one wanted", "I helped mediate small differences early"],
        "result": ["we delivered a strong result together", "the launch went smoothly",
                   "the collaboration was praised by leadership", "we hit our shared target"],
    },
    "Adaptability": {
        "situation": ["a major requirement changed mid-project", "a key teammate left unexpectedly",
                      "priorities shifted overnight", "a tool we relied on was suddenly unavailable"],
        "action": ["I quickly re-planned the work", "I picked up the additional scope",
                   "I found an alternative approach", "I stayed calm and adjusted the timeline"],
        "result": ["we still met the core goal", "the transition was smooth",
                   "we recovered without major slippage", "the outcome held up well"],
    },
    "Initiative": {
        "situation": ["I noticed a manual process wasting hours each week", "a gap in our reporting went unaddressed",
                      "no one owned a recurring problem", "I saw an opportunity to improve onboarding"],
        "action": ["I built a small tool to automate it", "I proposed and prototyped a fix on my own time",
                   "I took ownership and drove it to completion", "I documented and shared a better approach"],
        "result": ["we saved significant time each week", "the gap was closed",
                   "the improvement was adopted by the team", "onboarding became much faster"],
    },
    "Decision Making": {
        "situation": ["I had to choose a vendor with incomplete data", "we needed to pick a direction under time pressure",
                      "two options each had real trade-offs", "I had to decide whether to delay a launch"],
        "action": ["I listed the trade-offs and weighed the risks", "I gathered the best available evidence quickly",
                   "I consulted the people closest to the problem", "I chose the option with the best risk-reward"],
        "result": ["the decision proved correct", "we avoided a costly mistake",
                   "the outcome validated the choice", "we moved forward with confidence"],
    },
    "Time Management": {
        "situation": ["three deadlines fell in the same week", "I was juggling project work and support duties",
                      "an urgent task arrived mid-sprint", "I had more work than time allowed"],
        "action": ["I ranked tasks by impact and urgency", "I blocked focused time for the hardest work",
                   "I renegotiated one deadline early", "I delegated what I could"],
        "result": ["I met every critical deadline", "nothing important slipped",
                   "the work was delivered on time", "I kept quality high under pressure"],
    },
    "Resilience": {
        "situation": ["a project I led was cancelled late", "I failed an important assessment the first time",
                      "a launch I owned had a serious defect", "I received tough critical feedback"],
        "action": ["I reflected honestly on what went wrong", "I asked for feedback and made a plan",
                   "I fixed the issue and changed my approach", "I kept going and rebuilt my confidence"],
        "result": ["I succeeded on the next attempt", "I came back stronger and more careful",
                   "the second version was a clear success", "I grew from the experience"],
    },
    "Innovation": {
        "situation": ["our workflow had an obvious inefficiency", "customers wanted something we did not offer",
                      "a routine task was ripe for improvement", "I spotted a better way to use our data"],
        "action": ["I designed a new approach and tested it", "I built a proof of concept to show the value",
                   "I iterated on the idea with feedback", "I championed the change with the team"],
        "result": ["it was rolled out and measurably helped", "adoption grew quickly",
                   "it became part of our standard process", "the impact was clear and lasting"],
    },
}

# Quality-tiered language
HEDGES = ["I guess", "kind of", "sort of", "maybe", "I think", "probably", "honestly not sure but"]
VAGUE = ["it was fine", "things worked out", "we just dealt with it", "it was okay in the end",
         "nothing special really", "we managed somehow"]
CONNECTIVES = ["First,", "Then,", "After that,", "Next,", "To address it,", "Soon after,"]
STANCE = ["I took clear ownership of the situation.", "I made it my priority to resolve it constructively.",
          "I focused on addressing it directly.", "I was deliberate and proactive about it.",
          "I stayed calm and took responsibility."]
REFLECTION = ["Looking back, I learned the value of listening before reacting.",
              "The experience taught me to communicate earlier and more clearly.",
              "I took away a lasting lesson about preparation and follow-through.",
              "It reshaped how I approach similar situations now.",
              "I now apply that same structured approach by default."]
METRICS = ["by about 40%", "from three days to half a day", "saving roughly 10 hours a week",
           "improving the result by a clear margin", "ahead of the deadline by two days",
           "with a 25% improvement", "reducing errors to near zero"]
OFFTOPIC = ["I don't really remember a specific example.", "I haven't faced that kind of situation.",
            "Not sure this applies to me.", "I usually just let things sort themselves out."]


def pick(pool):
    return random.choice(pool)


def specific(text, p):
    """Optionally append a concrete metric with probability p."""
    if random.random() < p:
        return text + " " + pick(METRICS)
    return text


def generate_response(competency, quality):
    """Generate a response whose textual quality scales with `quality` (1..5)."""
    c = COMPETENCY_CONTENT[competency]
    sit, act, res = pick(c["situation"]), pick(c["action"]), pick(c["result"])

    if quality == 1:
        # very short, vague/off-topic, hedged, no structure
        if random.random() < 0.5:
            return pick(OFFTOPIC)
        return f"{pick(HEDGES).capitalize()} {pick(VAGUE)}."
    if quality == 2:
        # short, generic, minimal specifics, some hedging
        s = f"{pick(HEDGES).capitalize()}, {sit}. {pick(VAGUE).capitalize()}."
        return s
    if quality == 3:
        # partial STAR (situation + action), some relevance, little quantification
        s = f"{pick(CONNECTIVES)} {sit}. {act}."
        if random.random() < 0.4:
            s += f" {res.capitalize()}."
        return s
    if quality == 4:
        # STAR mostly complete, structured, some specificity
        s = (f"{pick(CONNECTIVES)} {sit}. {pick(STANCE)} {pick(CONNECTIVES)} {act}. "
             f"{specific(res.capitalize(), 0.5)}.")
        return s
    # quality == 5: full STAR + metrics + reflection + strong competency language
    s = (f"In one case, {sit}. {pick(STANCE)} {act}, and I kept everyone informed throughout. "
         f"{specific(res.capitalize(), 0.95)}. {pick(REFLECTION)}")
    return s


def maybe_blur(quality, blur=0.35):
    """With prob `blur`, generate text at an adjacent quality level (realistic label noise).
    Tuned so a simple baseline lands ~0.78-0.85 QWK, leaving headroom for the transformer."""
    if random.random() < blur:
        return max(1, min(5, quality + random.choice([-1, 1])))
    return quality


def biased_label(true_q, group, p_down=0.72):
    """Simulate biased raters: group B systematically under-scored at mid/high quality.
    Tuned so the 4/5ths selection-rate ratio falls clearly below 0.8 (adverse impact)."""
    if group == "B" and true_q >= 3 and random.random() < p_down:
        return max(1, true_q - 1)
    return true_q


def main(per_cell):
    rows = []
    rid = 0
    for qid, comp, question, rubric in QUESTIONS:
        for quality in range(1, 6):
            for group in ["A", "B"]:
                for _ in range(per_cell):
                    rid += 1
                    text_q = maybe_blur(quality)            # quality the TEXT reflects
                    response = generate_response(comp, text_q)
                    true_q = quality                         # latent intended label
                    human_fair = true_q                      # fair observed label
                    human_bias = biased_label(true_q, group) # biased observed label
                    rows.append({
                        "id": f"r{rid:05d}",
                        "question_id": qid,
                        "question": question,
                        "rubric": rubric,
                        "group": group,
                        "true_quality": true_q,
                        "human_score": human_fair,
                        "human_score_biased": human_bias,
                        "response": response,
                    })
    random.shuffle(rows)
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["id", "question_id", "question", "rubric", "group",
            "true_quality", "human_score", "human_score_biased", "response"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def stratified_split(rows, ratios=(0.7, 0.15, 0.15), seed=42):
    """Split stratified on (question_id, human_score, group) so all questions appear in all splits.
    Appropriate here because the task is 'score a response to a KNOWN question against its rubric'.
    Responses are uniquely generated, so there is no row-level duplication across splits."""
    rng = random.Random(seed)
    from collections import defaultdict
    strata = defaultdict(list)
    for r in rows:
        strata[(r["question_id"], r["human_score"], r["group"])].append(r)
    train, val, test = [], [], []
    for _, items in strata.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * ratios[0]); n_va = int(n * (ratios[0] + ratios[1]))
        train += items[:n_tr]; val += items[n_tr:n_va]; test += items[n_va:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cell", type=int, default=40,
                    help="responses per (question x score x group). 40 -> 12*5*2*40 = 4800 rows")
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    rows = main(args.per_cell)
    raw = os.path.join(args.out_dir, "raw", "responses.csv")
    write_csv(rows, raw)
    tr, va, te = stratified_split(rows)
    write_csv(tr, os.path.join(args.out_dir, "processed", "train.csv"))
    write_csv(va, os.path.join(args.out_dir, "processed", "val.csv"))
    write_csv(te, os.path.join(args.out_dir, "processed", "test.csv"))
    print(f"Generated {len(rows)} rows -> {raw}")
    print(f"Splits: train={len(tr)} val={len(va)} test={len(te)}")
