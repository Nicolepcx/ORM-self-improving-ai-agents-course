"""
Self-Improving AI Agents: course lab app.

Run locally:  ALLOW_NO_PASSWORD=1 streamlit run app.py
"""

import streamlit as st

from core.auth import is_authenticated, require_access
from core.ui import inject_css

st.set_page_config(page_title="Self-Improving AI Agents · Lab", page_icon=":material/school:", layout="wide")
inject_css()

pages = {
    "Course": [
        st.Page("views/home.py", title="Start here", icon=":material/home:", default=True),
    ],
    "Demos": [
        st.Page("views/d1_relative_policies.py", title="1 · Relative policies", icon=":material/leaderboard:", url_path="relative-policies"),
    ],
    "Hands-on": [
        st.Page("views/s1_frozen_lake.py", title="1 · Frozen Lake", icon=":material/ac_unit:", url_path="frozen-lake"),
        st.Page("views/s2_reward_function.py", title="2 · Reward function", icon=":material/rule:", url_path="reward-function"),
        st.Page("views/s3_rpls.py", title="3 · Rock Paper Lizard Spock", icon=":material/sports_esports:", url_path="rpls"),
        st.Page("views/s4_additional_histories.py", title="4 · Additional histories", icon=":material/account_tree:", url_path="additional-histories"),
        st.Page("views/s5_mcp_tools.py", title="5 · MCP tools", icon=":material/build:", url_path="mcp-tools"),
    ],
}

# Resolve the page before the password gate so deep links survive the login.
page = st.navigation(pages, position="sidebar" if is_authenticated() else "hidden")
require_access()
page.run()
