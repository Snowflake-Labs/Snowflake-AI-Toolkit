import streamlit as st
from src.utils import (
    render_image,
    check_setup_status,
    run_setup_silently,
)
from src import home, playground, build, notification
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session
from pathlib import Path
import json
from src.search import display_search
from src.cortex_agent import (
    display_cortex_agent,
)
from src.settings import display_settings

# Load the config file
config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Set the page title and layout
st.set_page_config(page_title="Snowflake AI Toolkit", layout="wide")

# Ensure session state variables are initialized
if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default page

if "snowflake_session" not in st.session_state:
    st.session_state.snowflake_session = None

if "legacy_function" not in st.session_state:
    st.session_state["legacy_function"] = False

if "show_private_preview_models" not in st.session_state:
    st.session_state["show_private_preview_models"] = False

if "setup_completed" not in st.session_state:
    st.session_state["setup_completed"] = False


# Establish the session if not already initialized
if st.session_state.snowflake_session is None:
    if config["mode"] == "native":
        try:
            st.session_state.snowflake_session = get_active_session()
        except Exception as e:
            st.error(f"Failed to get active session in native mode: {e}")
    elif config["mode"] == "debug":
        try:
            # Get Snowflake configuration from JSON file
            from src.utils import get_snowflake_config

            snowflake_config = get_snowflake_config()

            # Extract connection parameters
            account = snowflake_config.get("account")
            user = snowflake_config.get("user")
            password = snowflake_config.get("password")
            role = snowflake_config.get("role")
            warehouse = snowflake_config.get("warehouse")
            database = snowflake_config.get("database")
            schema = snowflake_config.get("schema")

            missing_vars = []
            if not all([account, user, password, role, warehouse, database, schema]):
                # Check which required fields are missing
                missing_vars = [
                    var
                    for var, val in [
                        ("account", account),
                        ("user", user),
                        ("password", password),
                        ("role", role),
                        ("warehouse", warehouse),
                        ("database", database),
                        ("schema", schema),
                    ]
                    if not val
                ]

            if missing_vars:
                st.error(
                    f"Missing Snowflake configuration fields in settings_config.json: {', '.join(missing_vars)}"
                )
                st.stop()

            private_key_path = snowflake_config.get("private_key_path")
            private_key_passphrase = snowflake_config.get("private_key_passphrase")
            private_key = snowflake_config.get("private_key")

            connection_parameters = {}
            if private_key_path and private_key_passphrase:
                connection_parameters = {
                    "account": account,
                    "user": user,
                    "private_key_file": private_key_path,
                    "private_key_file_pwd": private_key_passphrase,
                    "role": role,
                    "warehouse": warehouse,
                    "database": database,
                    "schema": schema,
                }
            elif private_key:
                # Use inline private key
                connection_parameters = {
                    "account": account,
                    "user": user,
                    "private_key": private_key,
                    "role": role,
                    "warehouse": warehouse,
                    "database": database,
                    "schema": schema,
                }
            else:
                connection_parameters = {
                    "account": account,
                    "user": user,
                    "password": password,
                    "role": role,
                    "warehouse": warehouse,
                    "database": database,
                    "schema": schema,
                }
            # connector = SnowflakeConnector(**connection_parameters)
            # truelens_session = TruSession(connector)
            # st.session_state.truelens_session = truelens_session
            # run_dashboard(truelens_session)
            st.session_state.snowflake_session = Session.builder.configs(
                connection_parameters
            ).create()
        except Exception as e:
            st.error(f"Failed to create session in debug mode: {e}")


# Check if session is successfully created
if st.session_state.snowflake_session is None:
    st.error("Failed to connect to Snowflake.")
else:
    # Get demo setup mode from config
    # 0 = Skip setup completely
    # 1 = Check setup status only (don't install)
    # 2 = Check and install if needed (default)
    demo_setup_mode = config.get("demo_setup_mode", 2)

    # Only run setup functions once on first load
    if not st.session_state.setup_completed and demo_setup_mode > 0:
        # Check if setup is already complete
        is_setup_complete, status_message = check_setup_status(
            st.session_state.snowflake_session
        )

        print(f"Setup status: {status_message}")

        if is_setup_complete:
            # Setup is already complete
            st.session_state.setup_completed = True
        elif demo_setup_mode == 2:
            # Setup is needed and installation is enabled
            try:
                with st.spinner(
                    "Setting up the environment... This may take a few moments."
                ):
                    # Run setup functions with minimal output
                    print("Running setup...")
                    success, message = run_setup_silently(
                        st.session_state.snowflake_session, config
                    )

                    if not success:
                        raise Exception(message)

                # Mark setup as completed and show success message
                st.session_state.setup_completed = True
                # with st.sidebar:
                #     st.success("✅ Demo environment setup completed!")

            except Exception as e:
                st.error(f"Error while setting up demo environment: {e}")
                # Don't mark as completed if there was an error
                st.session_state.setup_completed = False
        else:
            # demo_setup_mode == 1: Check only, don't install
            if not is_setup_complete:
                st.warning(
                    "⚠️ Demo environment is not set up. Please run setup manually or set demo_setup_mode to 2 in settings_config.json"
                )
            st.session_state.setup_completed = (
                True  # Mark as completed to avoid repeated checks
            )
    elif demo_setup_mode == 0:
        # Skip setup completely
        st.session_state.setup_completed = True

# Load custom CSS for sidebar styling
st.markdown(
    """
    <style>
            
    /* Sidebar Background */
    .css-18e3c8v {
        background-color: #333;
        padding: 20px;
        color: white;
    }

    /* Sidebar Sections */
    .sidebar-section {
        background-color: #444;
        border-radius: 8px;
        padding: 15px;
        color: white;
        border: 1px solid #555;
        width: 100%; /* Ensures uniform width */
        text-align: center;
    }
            
        [data-testid="stSidebar"] button { 
            width: 100%;    
            justify-content: center;
            text-align: center;
            item-align: center;
            cursor: pointer;
            border-radius: 8px;
            padding: 10px;
        } 
            
    </style>
""",
    unsafe_allow_html=True,
)


# Sidebar Navigation with logo
render_image(config["default_settings"]["logo_path"])

# Sidebar content with expanders and buttons under each category
with st.sidebar:
    st.title("📋 Menu")

    # Overview Section
    with st.expander("🔍 **Overview**", expanded=False):
        if st.button("📄 About"):
            st.session_state.page = "Home"
        # if st.button("⚙️ Setup"):
        #     st.session_state.page = "Setup"
        if st.button("⚙️ Settings"):
            st.session_state.page = "Settings"

    # Components Section
    with st.expander("✨ **Components**", expanded=False):
        if st.button("🎮 Playground"):
            st.session_state.page = "Playground"
        if st.button("🔧 Build"):
            st.session_state.page = "Build"
        if st.button("🔍 Cortex Search"):
            st.session_state.page = "Cortex Search"
        if st.button("🤖 Cortex Agent"):
            st.session_state.page = "Cortex Agent"
        if st.button("🔔 Notification"):
            st.session_state.page = "Notification"

# Pages dictionary with corresponding display functions
pages = {
    "Home": home.display_home,
    "Playground": playground.display_playground,
    "Build": build.display_build,
    "Notification": notification.display_notification,
    "Cortex Search": display_search,
    "Cortex Agent": display_cortex_agent,
    # "Setup": setup.display_setup,
    "Settings": display_settings,
}

# Render the selected page from the pages dictionary
if st.session_state.page in pages:
    try:
        pages[st.session_state.page](st.session_state.snowflake_session)
    except Exception as e:
        st.error(f"Error loading {st.session_state.page} page: {e}")
