from __future__ import annotations

from IPython.display import display, Javascript


def start_exam_timer(
    enabled=True,
    minutes=15,
    warn_minutes=5,
    title="⏱️ Timer",
    end_message="Exercise time finished",
):
    """
    Browser-side countdown timer for Jupyter/Colab.

    enabled can be:
      - False: do nothing
      - True: use `minutes`
      - int/float: interpret as minutes and override `minutes`
    """
    if enabled is False:
        return

    if isinstance(enabled, (int, float)) and not isinstance(enabled, bool):
        minutes = float(enabled)

    if minutes <= 0:
        return

    total_seconds = int(round(minutes * 60))
    warn_seconds = int(round(warn_minutes * 60))

    js = f"""
    let totalSeconds = {total_seconds};
    let warned = false;

    const el = document.createElement("div");
    el.style.fontSize = "72px";
    el.style.fontWeight = "600";
    el.style.fontFamily = "monospace";
    el.style.textAlign = "center";
    el.style.marginTop = "40px";
    el.style.color = "#222";

    document.body.appendChild(el);

    function formatTime(seconds) {{
      const m = Math.floor(seconds / 60);
      const s = seconds % 60;
      return `${{m.toString().padStart(2,'0')}}:${{s.toString().padStart(2,'0')}}`;
    }}

    function speak(text) {{
      if (!("speechSynthesis" in window)) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(u);
    }}

    const interval = setInterval(() => {{
      if (!warned && {warn_seconds} > 0 && totalSeconds === {warn_seconds}) {{
        warned = true;
        speak("{warn_minutes} minutes remaining.");
      }}

      if (totalSeconds <= 60) {{
        el.style.color = "#d32f2f";
      }}

      el.innerText = `{title} ${{formatTime(totalSeconds)}}`;
      totalSeconds--;

      if (totalSeconds < 0) {{
        clearInterval(interval);
        el.innerText = "✅ Time’s up";
        speak("{end_message}");
      }}
    }}, 1000);
    """
    display(Javascript(js))
