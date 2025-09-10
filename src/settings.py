import streamlit as st

def display_settings(snowflake_session):
    st.title("Settings")
    st.write("Configure your preferences here.")
    # Legacy Function Setting
    col1, col2 = st.columns([3, 1])
    if "legacy_function" not in st.session_state:
        st.session_state.legacy_function = False
    with col1:
        st.markdown("<div style='padding-top: 8px;'>Enable legacy function</div>", unsafe_allow_html=True)
    with col2:
        toggle = st.toggle("Enable Legacy Functionality", value=st.session_state.legacy_function, label_visibility="collapsed")
    if toggle:
        st.session_state.legacy_function = toggle
   
    # Private Preview Models Setting
    col1, col2 = st.columns([3, 1])
    if "show_private_preview_models" not in st.session_state:
        st.session_state.show_private_preview_models = False
    with col1:
        st.markdown("<div style='padding-top: 8px;'>Show Private Preview Models</div>", unsafe_allow_html=True)
    with col2:
        show_private_preview_models = st.toggle("Show Private Preview Models", value=st.session_state.show_private_preview_models, label_visibility="collapsed")
    if show_private_preview_models:
        st.session_state.show_private_preview_models = show_private_preview_models

    st.markdown("""
        <style>
            .header-section {
                background-color: #56CCF2;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                color: white;
                font-family: 'Arial', sans-serif;
            }
            .header-title {
                font-size: 3em;
                font-weight: bold;
            }
            .header-subtitle {
                font-size: 1.2em;
                margin-top: 10px;
            }
            .section {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
            .section-title {
                font-size: 1.8em;
                font-weight: bold;
                color: #56CCF2;
                margin-bottom: 10px;
            }
            .section-content {
                font-size: 1.1em;
                line-height: 1.5;
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)

    # Configuring Debug Mode Section
    st.markdown("""
        <div class="section">
            <div class="section-title">Configuring Debug Mode</div>
            <div class="section-content">
                Follow these steps to configure the application for Debug Mode:
                <ol>
                    <li>Open the <code>src/settings_config.json</code> file in a text editor.</li>
                    <li>Locate the <code>mode</code> parameter and set its value to <code>"debug"</code>:
                        <pre><code>
    {
        "mode": "debug",
        "snowflake": {
        "account": "your-account-url",
        "user": "your-username",
        "password": "your-password",
        "role": "your-role",
        "warehouse": "your-warehouse",
        "database": "your-database",
        "schema": "your-schema"
        }
    }
                        </code></pre>
                    </li>
                    <li>Save the file after making changes.</li>
                </ol>
            </div>
        </div>
    """, unsafe_allow_html=True)
