"""
Streamlit app for SEC 10-K Company Similarity Explorer.
Minimal demo version with login protection.
"""

import streamlit as st
from sec_rag.agent.similarity_agent import create_agent

# Page config
st.set_page_config(
    page_title="SEC 10-K Similarity Explorer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4788;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .example-btn {
        margin: 0.2rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def check_login(username: str, password: str) -> bool:
    """Check if credentials are valid."""
    try:
        correct_user = st.secrets["auth"]["username"]
        correct_pass = st.secrets["auth"]["password"]
        return username == correct_user and password == correct_pass
    except Exception:
        # Fallback for local development without secrets
        return username == "demo" and password == "demo"


def login_page():
    """Display login page."""
    st.markdown('<p class="main-header">📊 SEC 10-K Similarity Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Company Analysis</p>', unsafe_allow_html=True)
    
    st.info("🔒 This demo is password-protected. Please login to continue.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if check_login(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")


def initialize_agent():
    """Initialize the similarity agent (cached)."""
    if "agent" not in st.session_state:
        with st.spinner("🔄 Loading similarity engine..."):
            st.session_state.agent = create_agent(
                model="gemini-2.5-flash",
                verbose=False,
                max_iterations=15
            )
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": """Welcome! I can help you explore similarities between 37 S&P 500 companies using their SEC 10-K filings.

**What I can do:**
• Find similar companies by business model, risk profile, or financials
• Explain why companies are similar
• Extract specific information (directors, products, risks, etc.)
• Compare companies across different dimensions

**Available companies:** AAPL, MSFT, GOOG, META, NVDA, TSLA, AMZN, and 30 more!

Try one of the example queries below or ask your own question."""
                }
            ]


def main_app():
    """Main application interface."""
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">📊 SEC 10-K Similarity Explorer</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Analyzing 37 S&P 500 companies with AI</p>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()
    
    # Initialize agent
    initialize_agent()
    
    # Example queries
    st.markdown("### 💡 Try these examples:")
    cols = st.columns(3)
    
    examples = [
        "Find companies similar to TSLA",
        "Why are AAPL and MSFT similar?",
        "Who are the directors at GOOG?"
    ]
    
    for idx, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(example, key=f"example_{idx}", use_container_width=True):
                # Add to messages and process
                st.session_state.messages.append({"role": "user", "content": example})
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.agent.run(example)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    st.markdown("---")
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message["content"])
    
    # Input box (fixed at bottom)
    user_input = st.chat_input("Ask me about company similarities...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get agent response
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.agent.run(user_input)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to show new messages
        st.rerun()


def main():
    """Main entry point."""
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Show appropriate page
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()


if __name__ == "__main__":
    main()
