import streamlit as st

from core.ui import REPO

st.title("Self-Improving AI Agents · Lab")
st.markdown(
    """
This app is the **live hands-on companion** to the course notebooks. Each page focuses on the one idea
you practice in that session: visuals and controls up front, the code that matters one click away.

**How to use it**
1. Pick the session from the left sidebar.
2. Do the task in the blue box at the top of the page.
3. Open the *code* panels when you want to see exactly what runs underneath.

The full notebooks, including the parts we skip live (like training a model on a GPU),
stay available on GitHub and in Colab, so you can go deeper after the session.
"""
)
st.link_button("Course repository on GitHub", f"https://github.com/{REPO}", icon=":material/open_in_new:")
