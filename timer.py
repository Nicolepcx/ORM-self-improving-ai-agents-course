from IPython.display import display, Javascript

def start_exam_timer(
    enabled=True,
    minutes=15,
    warn_minutes=5,
    title="⏱️ Timer",
    end_message="Exercise time finished",
):
    if not enabled:
        return

    js = f"""
    let totalSeconds = {minutes} * 60;
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
      if (!warned && totalSeconds === {warn_minutes} * 60) {{
        warned = true;
        speak("Warning. {warn_minutes} minutes remaining.");
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
