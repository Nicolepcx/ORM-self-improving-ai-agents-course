import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ui import page_header, takeaway

BLOG = "https://rikiyat.github.io/blog/weight-space-programming/"
LOGS = "https://github.com/RikiyaT/lm_understand-logs"
EXP = lambda name: f"{LOGS}/tree/main/{name}"


@st.cache_resource
def exp3() -> dict:
    """Precomputed once from the authors' released Exp 3 networks, ground truth and run logs (CC BY 4.0)."""
    return json.loads((Path(__file__).resolve().parent.parent / "labs" / "data" / "wsp_exp3.json").read_text())


page_header(
    session=5,
    kind="OUTLOOK",
    title="Weight-Space Programming: when LLMs edit weights directly",
    subtitle=(
        "Everything in this course improves a model the same way: generate behaviour, score it, and let **gradients** "
        "move the weights. A new study shows a frontier LLM doing something different: **reading neural network "
        "weights, reasoning about what they compute, and editing them directly**, with no training loop."
    ),
    minutes=10,
    notebook_path=None,
)
st.link_button("Read the original blog post (Takehi, Teoh, Zhang, Khattab · MIT CSAIL · Nov 2026)", BLOG,
               icon=":material/open_in_new:", type="primary")
st.caption("This page explains the study in our own words and shows its released data. All numbers come from the "
           f"authors' published logs ([lm_understand-logs]({LOGS}), CC BY 4.0). It's an early blog post, not a "
           "peer-reviewed paper.")

tabs = st.tabs(["① The idea", "② What they found", "③ Try it: read a network", "④ Astra's reasoning",
                "⑤ What might be possible"])

# ======================================================================== ① the idea
with tabs[0]:
    st.markdown("#### Two ways to improve a model")
    c1, c2 = st.columns(2, gap="large")
    with c1.container(border=True):
        st.markdown("**What this course does: learn from feedback**")
        st.markdown(
            "1. The model generates rollouts\n"
            "2. A reward model or judge scores them\n"
            "3. Advantages say what was better than expected\n"
            "4. A **gradient step** nudges *all* weights a tiny bit\n"
            "5. Repeat thousands of times\n\n"
            "The optimizer (Adam, Muon…) is a small, fixed formula. It doesn't know *what* any weight does."
        )
    with c2.container(border=True):
        st.markdown("**Weight-Space Programming: reason about the weights**")
        st.markdown(
            "1. An LLM **reads** the weights (as numbers, or with linear algebra)\n"
            "2. It works out **what each part computes**, or should compute\n"
            "3. It **edits** a targeted set of weights, or writes new ones\n"
            "4. No gradients, often no data and no training loop\n\n"
            "The “optimizer” is a model that understands, at least partly, what it's changing."
        )
    st.markdown(
        "The authors tested OpenAI's **GPT-6 Astra** on three abilities, always without letting it write a training "
        "loop: **reading** weights (what was this network trained to do?), **fixing** weights (repair a failing "
        "network by editing it) and **writing** weights (build a working network from zero)."
    )
    takeaway("<b>Why this matters for self-improving agents:</b> today an agent improves through data and gradients. "
             "If models can deliberately edit model weights, the loop could close: an agent that diagnoses and repairs "
             "a model, possibly its own. That is one possible path toward recursive self-improvement, and exactly "
             "why it needs the evaluation and oversight habits from this course.")

# ======================================================================== ② results
with tabs[1]:
    st.markdown("All numbers from the experiment READMEs in the released logs. Click through for every prompt, "
                "answer and evaluation.")
    st.markdown("#### Reading weights")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Which game?** · Exp 1")
        st.metric("Correct game, open answer", "4 / 4")
        st.caption("Small game-playing networks (17,253 weights, trained with PPO for 1B steps): Breakout, Pong, "
                   "Pac-Man, Freeway. Astra saw only the raw weights.")
        st.link_button("Logs", EXP("exp1_which_game"))
    with c2.container(border=True):
        st.markdown("**Which language?** · Exp 3")
        st.metric("Correct training text", "6 / 6 conversations")
        st.caption("Only the four middle blocks of 7.2M-weight language models, no input or output layer, no forward "
                   "pass. It also named every copying head with no false positives. *Try it in tab ③.*")
        st.link_button("Logs", EXP("exp3_language"))
    with c3.container(border=True):
        st.markdown("**Which model is sycophantic?** · Exp 7")
        st.metric("With an attention tool", "3 / 3 pairs")
        st.caption("Two Gemma 3 270M fine-tunes, one honest, one that agrees with whatever the user suggests. "
                   "From weights alone it **failed in 4 attempts**; seeing attention patterns on texts it wrote, it "
                   "found the sycophant every time.")
        st.link_button("Logs", EXP("exp7_sycophancy"))

    st.markdown("#### Fixing weights")
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1.container(border=True):
        st.markdown("**Repair a stuck control policy** · Exp 4")
        st.metric("Success after one edit", "1,024 / 1,024", "from 0 / 256")
        st.caption("A PPO car policy (1B steps) always stalled on a ramp. One reply, no tools, changing 3.5% of the "
                   "weights (6,073 of 171,664).")
        st.link_button("Logs", EXP("exp4_fix_car"))
    with c2.container(border=True):
        st.markdown("**Make Qwen3.5-0.8B better at math, without gradients** · Exp 8")
        st.dataframe(pd.DataFrame([
            {"forward-pass budget": "unedited", "MATH-500": 7.9, "GSM8K": 13.8, "IFEval": 21.0, "answers hitting the token limit": "93.0%"},
            {"forward-pass budget": "10", "MATH-500": 12.3, "GSM8K": 25.7, "IFEval": 49.0, "answers hitting the token limit": "87.2%"},
            {"forward-pass budget": "25", "MATH-500": 14.1, "GSM8K": 25.1, "IFEval": 39.9, "answers hitting the token limit": "86.4%"},
            {"forward-pass budget": "50", "MATH-500": 48.1, "GSM8K": 67.1, "IFEval": 62.3, "answers hitting the token limit": "33.1%"},
        ]), hide_index=True)
        st.caption("Astra found a direction in the MLPs of layers 15 and 19 that drives the model into reasoning "
                   "loops (a problem Qwen's own model card mentions) and removed it from 131,072 weights (0.016%). "
                   "**Counterpoint:** the same approach on Gemma 3 270M, which doesn't loop, gave no meaningful gain. "
                   "Editing removed an obstacle; it didn't add an ability.")
        st.link_button("Logs", EXP("exp8_math"))

    st.markdown("#### Writing weights from scratch")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1.container(border=True):
        st.markdown("**MinAtar game players, written one-shot from zero weights** · Exp 5")
        st.dataframe(pd.DataFrame([
            {"game": "Breakout", "PPO, 100M steps": 15.8, "PPO, 500M steps": 25.3, "Astra, one-shot": 517.4},
            {"game": "Seaquest", "PPO, 100M steps": 5.6, "PPO, 500M steps": 88.9, "Astra, one-shot": 53.1},
            {"game": "Asterix", "PPO, 100M steps": 25.8, "PPO, 500M steps": 55.2, "Astra, one-shot": 53.2},
        ]), hide_index=True)
        st.caption("No data, no gradients, no game rollouts: it reasoned out the weights of a 20,486-weight network "
                   "(using 637-815 nonzero weights). It beat 100M-step PPO on all three games and 500M-step PPO on "
                   "Breakout only.")
        st.link_button("Logs", EXP("exp5_write_from_scratch"))
    with c2.container(border=True):
        st.markdown("**A 1.5M-weight Go player** · Exp 6")
        st.metric("Win rate vs PPO (seed 0), 50 forward passes", "84.0%", "54.1% with 0 passes", delta_color="off")
        st.caption("From zero weights on a 9×9 board, with a Python sandbox and a small budget of forward passes to "
                   "test its network.")
        st.link_button("Logs", EXP("exp6_go_from_scratch"))

# ======================================================================== ③ try it
data = exp3()["networks"]
NAMES = {"Network A": "romaji", "Network B": "english", "Network C": "python"}
CORPORA = {"python": "Python source code", "romaji": "Japanese literature in romaji", "english": "English Wikipedia"}

with tabs[2]:
    st.markdown("#### Read a language model from its weights (Experiment 3)")
    st.markdown(
        "Three tiny character-level transformers (4 layers × 6 heads) were each trained on **one** of: Python code, "
        "Japanese literature in romaji, or English Wikipedia. Astra got only the **middle blocks**: no embeddings, "
        "no output layer, and **no forward pass allowed**. It could only do linear algebra on the weight matrices."
    )
    pick = st.radio("Pick a network", list(NAMES), horizontal=True, key="wsp_net")
    lang = NAMES[pick]
    net = data[lang]
    heads = [f"L{l}H{h}" for l in range(4) for h in range(6)]

    def grid(values, title, colorscale, fmt="{:.2f}"):
        z = [[values[f"L{l}H{h}"] for h in range(6)] for l in range(4)]
        fig = go.Figure(go.Heatmap(z=z, x=[f"head {h}" for h in range(6)], y=[f"layer {l}" for l in range(4)],
                                   text=[[fmt.format(v) for v in row] for row in z], texttemplate="%{text}",
                                   colorscale=colorscale, showscale=False, xgap=3, ygap=3))
        fig.update_layout(title=dict(text=title, font_size=14), height=300, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(autorange="reversed"))
        return fig

    st.markdown("**What the weights show** (computed from this network's real weights, the same quantities Astra "
                "started its analysis with):")
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(grid({h: net["heads"][h]["qk_norm"] for h in heads}, "Selectivity per head: ‖M‖",
                         "Blues"), config={"displayModeBar": False})
    c2.plotly_chart(grid({h: net["heads"][h]["ov_top_sv"][0] for h in heads},
                         "Write strength per head: top σ of B", "Purples",
                         "{:.1f}"), config={"displayModeBar": False})
    st.caption("M (query-key map: what a head looks for) = P·D·W_qᵀ·W_k·D·P / 8 and B (value-output map: what it "
               "writes) = P·W_o·W_v·D·P per head (D = LayerNorm gain, P = mean-centering), "
               "exactly as in Astra's first code run. Linear algebra on weights, no input involved.")

    st.markdown("**Your turn: predict, then reveal**")
    c1, c2 = st.columns(2)
    guess_lang = c1.selectbox("Which text was it trained on?", list(CORPORA.values()), key=f"wsp_guess_lang_{lang}")
    guess_heads = c2.multiselect("Which heads are **copying (induction) heads**?", heads, key=f"wsp_guess_heads_{lang}",
                                 help="Copying heads attend to the token right after an earlier occurrence of the "
                                      "current one, so the model can repeat earlier text. They need an earlier head "
                                      "to look back, so they can't be in layer 0.")
    if st.toggle("Reveal Astra's answer and the measured truth", key=f"wsp_reveal_{lang}"):
        truth_copy = [h for h in heads if net["truth"][h]["induct"] > 0.2]
        r0 = net["runs"]["r0"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Trained on", CORPORA[lang], "you got it" if guess_lang == CORPORA[lang] else "not your guess",
                  delta_color="normal" if guess_lang == CORPORA[lang] else "inverse")
        m2.metric("Astra's answer (conversation 1)", r0["corpus_guess"])
        m3.metric("Copying heads you found", f"{len(set(guess_heads) & set(truth_copy))} of {len(truth_copy)}",
                  f"{len(set(guess_heads) - set(truth_copy))} false" if guess_heads else None, delta_color="off")
        st.caption(f"Astra's probabilities: {r0['probabilities']}")
        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(grid({h: net["truth"][h]["induct"] for h in heads},
                             "Measured: induction score (copy > 0.2)", "Greens", "{:.3f}"),
                        config={"displayModeBar": False})
        labels = r0["head_labels"]
        rows = [{"head": h, "Astra's label": labels[h], "measured induction": round(net["truth"][h]["induct"], 3),
                 "copying head?": "yes" if h in truth_copy else ""} for h in heads if labels[h] != "unclear" or h in truth_copy]
        c2.markdown("**Astra's head labels** (from weights only)")
        c2.dataframe(pd.DataFrame(rows), hide_index=True, height=280)
        st.success(f"Copying heads measured by running the network: **{', '.join(truth_copy)}**. "
                   f"Astra named: **{', '.join(r0['copy_named'])}**, without ever running it.",
                   icon=":material/visibility:")

# ======================================================================== ④ replay
with tabs[3]:
    st.markdown("#### How Astra worked it out, step by step")
    st.markdown("In each turn Astra wrote one Python program over the weights; the harness ran it in a sandbox and "
                "sent back the printed output. This is the real log of its first conversation.")
    pick2 = st.radio("Network", list(NAMES), horizontal=True, key="wsp_replay_net")
    lang2 = NAMES[pick2]
    replay = data[lang2]["replay_r0"]
    k = st.slider("Code run", 1, len(replay), 1, key=f"wsp_step_{lang2}")
    step = replay[k - 1]
    st.caption(f"Run {k} of {len(replay)} · {step['seconds']} s in the sandbox · trained on: "
               f"{'(see tab ③)' if not st.session_state.get(f'wsp_reveal_{lang2}') else CORPORA[lang2]}")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Astra's code**")
        st.code(step["code"], language="python")
    with c2:
        st.markdown("**What it got back**")
        st.code(step["output"], language="text")
    with st.expander("Astra's final answer (as written, with its evidence)"):
        st.markdown(data[lang2]["final_answer_r0"])
    st.link_button("All six conversations in the logs", EXP("exp3_language/runs"), icon=":material/folder_open:")

# ======================================================================== ⑤ future
with tabs[4]:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### What might become possible")
        st.markdown(
            "- **Targeted repairs instead of retraining:** find the few weights behind a failure (a reasoning loop, a "
            "stuck policy) and fix them, at a fraction of the cost of another training run.\n"
            "- **Auditing and interpretability:** ask *what does this network do?* or *which of these models is "
            "sycophantic?* by inspecting it, not only by testing its outputs.\n"
            "- **Unlearning and merging:** remove or combine capabilities by reasoning about where they live.\n"
            "- **A new kind of optimizer:** instead of a fixed formula like Adam, a model that deliberately reasons about "
            "the weights it changes.\n"
            "- **Agents that improve models, and maybe themselves:** the step from *an agent edits a model* to *an "
            "agent edits its own successor* is what makes this a recursive self-improvement question."
        )
    with c2:
        st.markdown("#### What the study does not show (yet)")
        st.markdown(
            "- It's an **early blog post**, not peer-reviewed, with **small samples** (e.g. 3 model pairs, 4 attempts).\n"
            "- Several experiments **relax the rules** (forward-pass budgets, an attention tool).\n"
            "- **Editing didn't add abilities:** Gemma, which had no looping problem, didn't improve. The Qwen gain "
            "came from removing one specific, known failure.\n"
            "- The PPO baselines are **best-of-seeds**, and Astra didn't beat 500M-step PPO on every game.\n"
            "- Scaling to large models is limited by context length and harness cost. A benchmark is in progress."
        )
    st.markdown("#### Questions to discuss")
    for question, hint in [
        ("If an agent can edit its own weights, what should evaluate the edit before it's kept?",
         "The same tools as in this course: held-out evals, judges, regression benchmarks, and humans in the loop. "
         "An edit that raises one score can break another (see how IFEval moved with MATH-500 in Exp 8)."),
        ("Is a weight edit more or less risky than a gradient update?",
         "More targeted and more explainable, if the model explains what it changed and why. But also easier to "
         "steer toward a hidden goal: reward hacking moves from outputs to weights."),
        ("Where would you use this in an agent system today?",
         "Diagnosis first: inspecting a fine-tuned model for unwanted behaviour (like the sycophancy test) is lower "
         "risk than letting an agent rewrite production weights."),
    ]:
        st.markdown(f"**{question}**")
        with st.expander("Some thoughts"):
            st.markdown(hint)
    st.link_button("Try the edited Qwen3.5-0.8B (Hugging Face demo)",
                   "https://huggingface.co/spaces/akhaliq/weight-space-programming-demo", icon=":material/rocket_launch:")
    st.link_button("Edited model weights on Hugging Face", "https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming",
                   icon=":material/download:")

st.divider()
st.caption(
    "Source: R. Takehi, J. Teoh, A. Zhang, O. Khattab (MIT CSAIL), *Sparks of Weight-Space Programming: LLMs can "
    f"understand and write neural network weights*, blog post, November 2026, [{BLOG}]({BLOG}). Data, networks and "
    f"logs: [{LOGS}]({LOGS}), CC BY 4.0. Weight quantities on this page were computed from those networks."
)
