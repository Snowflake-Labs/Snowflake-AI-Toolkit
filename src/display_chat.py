# ruff: noqa

import streamlit as st
import json
from pathlib import Path
from snowflake.core import Root
from snowflake.snowpark.exceptions import SnowparkSQLException

# Internal imports
from src.cortex_functions import *
from src.query_result_builder import *
from src.utils import *
from src.cortex_agent import *
from src.audio_components import *
from src.notification import *

# Load config file
config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Conditionally import streamlit_mic_recorder only in debug mode
if config.get("mode") == "debug":
    from streamlit_mic_recorder import speech_to_text


def initialize_chat_session_state():
    """Initialize all required session state variables for chat"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cortex_chat" not in st.session_state:
        st.session_state.cortex_chat = []
    if "show_private_preview_models" not in st.session_state:
        st.session_state.show_private_preview_models = False


def display_chat(session, options):
    """Main chat interface entry point"""
    # Initialize session state
    initialize_chat_session_state()

    # Options are now passed from playground.py where they're selected in initial_col2

    if options == "Search Service":
        display_search_service_chat(session)
    elif options == "RAG":
        display_rag_chat(session)
    elif options == "Cortex Agent":
        display_cortex_agent_chat(session)


def display_search_service_chat(session):
    """Search Service chat implementation"""
    slide_window = 20

    # Settings in expander
    with st.expander("Settings", expanded=True):
        st.subheader("Choose Your Search Service")
        col1, col2 = st.columns(2)
        with col1:
            db_list = list_databases(session)
            default_db = get_default_value("Search", "database")
            db_index = db_list.index(default_db) if default_db in db_list else 0
            selected_db = st.selectbox("Database", db_list, index=db_index)
        with col2:
            schema_list = list_schemas(session, selected_db)
            default_schema = get_default_value("Search", "schema")
            schema_index = (
                schema_list.index(default_schema)
                if default_schema in schema_list
                else 0
            )
            selected_schema = st.selectbox("Schema", schema_list, index=schema_index)

        col1, col2 = st.columns(2)
        with col1:
            cortex_services = list_cortex_services(
                session, selected_db, selected_schema
            )
            default_service = get_default_value("Search", "service")
            service_index = (
                cortex_services.index(default_service)
                if default_service in cortex_services
                else 0
            )
            selected_service = st.selectbox(
                "Service",
                cortex_services or [],
                index=service_index if cortex_services else 0,
            )
        attributes = []
        if selected_service:
            if "prev_selected_service" not in st.session_state:
                st.session_state.prev_selected_service = selected_service
            if st.session_state.prev_selected_service != selected_service:
                st.session_state.cortex_chat = []
                st.session_state.prev_selected_service = selected_service

            with col2:
                data = fetch_cortex_service(
                    session, selected_service, selected_db, selected_schema
                )
                row = data[0]
                cols = row.columns.split(",")
                attributes = row.attribute_columns.split(",")

                # Set default display column
                default_display_column = get_default_value("Search", "display_column")
                default_columns = (
                    [default_display_column] if default_display_column in cols else []
                )
                columns = st.multiselect(
                    "Display Columns", cols, default=default_columns
                )

        st.subheader("Create Filter & Limits")
        col3, col4 = st.columns(2)
        with col3:
            filter_column = st.selectbox("Filter Columns", attributes)
        with col4:
            filter_operator = st.selectbox(
                "Filter Operator", ["@eq", "@contains", "@gte", "@lte"]
            )
        filter_value = st.text_input(
            f"Enter value for {filter_operator} on {filter_column}"
        )

        if filter_column and filter_operator and filter_value:
            if filter_operator == "@eq":
                filter = {"@eq": {filter_column: filter_value}}
            elif filter_operator == "@contains":
                filter = {"@contains": {filter_column: filter_value}}
            elif filter_operator == "@gte":
                filter = {"@gte": {filter_column: filter_value}}
            elif filter_operator == "@lte":
                filter = {"@lte": {filter_column: filter_value}}
            st.write(f"Generated Filter: {filter}")
        else:
            filter = {}
        limit = st.slider("Limit Results", min_value=1, max_value=10, value=1)

        st.subheader("Choose Your Model")
        col5, col6 = st.columns(2)
        with col5:
            model_types = ["Base", "Fine Tuned"]
            if st.session_state.show_private_preview_models:
                model_types.append("Private Preview")
            model_type = st.selectbox("Model Type", model_types)
        with col6:
            if model_type == "Base":
                selected_model = st.selectbox(
                    "Model", config["default_settings"]["model"]
                )
            elif model_type == "Private Preview":
                selected_model = st.selectbox(
                    "Model",
                    config["default_settings"]["private_preview_models"],
                )
            else:
                fine_tuned_models = fetch_fine_tuned_models(session)
                selected_model = st.selectbox("Model", fine_tuned_models)

    # Chat container
    chat_placeholder = st.container(border=True, height=700)
    with chat_placeholder:
        st.subheader("Chat Messages")
        for message in st.session_state.get("cortex_chat", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    demo_response = None

    # Add audio input for Search Service chat (only in debug mode)
    if config.get("mode") == "debug":

        def create_audio_input_for_search():
            if "speech_to_text_search_output" not in st.session_state:
                st.session_state["speech_to_text_search_output"] = ""

            def update_search_question():
                audio_text = st.session_state.get(
                    "speech_to_text_search_output", ""
                ).strip()
                if audio_text:
                    # Process the audio input as a search question
                    if not columns:
                        show_toast_message(
                            "Please select columns to display.",
                            position="bottom-right",
                        )
                        return
                    st.session_state.cortex_chat.append(
                        {"role": "user", "content": audio_text}
                    )
                    with chat_placeholder:
                        with st.chat_message("user"):
                            st.markdown(audio_text)

                    try:
                        root = Root(session)
                        service = (
                            root.databases[selected_db]
                            .schemas[selected_schema]
                            .cortex_search_services[selected_service.lower()]
                        )
                        columns_lower = [col.lower() for col in columns]
                        resp = service.search(
                            query=audio_text,
                            columns=columns_lower,
                            filter=filter,
                            limit=int(limit),
                        )

                        retrieved_data = resp

                        def get_chat_history():
                            start_index = max(
                                0,
                                len(st.session_state.cortex_chat) - slide_window,
                            )
                            filtered_history = [
                                msg
                                for msg in st.session_state.messages[start_index:]
                                if not msg["content"].startswith("An error occurred")
                            ]
                            return filtered_history

                        chat_history = get_chat_history()

                        prompt = f"""
                                You are an AI assistant using Retrieval-Augmented Generation (RAG). Your task is to provide accurate and relevant answers based on the user's question, the retrieved context from a Cortex Search Service, and the prior chat history (if any). Follow these instructions:
                                1. Use the chat history to understand the conversation context, if context is empty, refer retrieved context.
                                2. Use the retrieved context to ground your answer in the provided data (this is in json form, so under the json keys and values fully).
                                3. Answer the question concisely and directly, without explicitly mentioning the sources (chat history or retrieved context) unless asked.   
                        
                                Note: Identify if the user is asking a question from the chat history or the retrieved context. If the user is asking a question from the chat history, answer the question based on the chat history. If the user is asking a question from the retrieved context, answer the question based on the retrieved context. If the user is asking a question from the chat history and the retrieved context, answer the question based on the chat history. If the user is asking a question that is not from the chat history or the retrieved context, answer the question based on the chat history.
                                
                                <chat_history>
                                {chat_history}
                                </chat_history>

                                <retrieved_context>
                                {retrieved_data}
                                </retrieved_context>

                                <question>
                                {audio_text}
                                </question>

                                Answer:
                                """

                        if prompt:
                            prompt = prompt.replace("'", "\\'")
                        res = execute_query_and_get_result(
                            session,
                            prompt,
                            selected_model,
                            "Generate RAG Response",
                        )

                        result_json = json.loads(res)
                        response_1 = result_json.get("choices", [{}])[0].get(
                            "messages", "No messages found"
                        )
                        st.session_state.cortex_chat.append(
                            {"role": "assistant", "content": response_1}
                        )
                        with chat_placeholder:
                            with st.chat_message("assistant"):
                                st.markdown(response_1)
                                # Add voice output for assistant response
                                create_voice_output_button(
                                    response_1, "search_assistant_response"
                                )
                    except Exception as e:
                        add_log_entry(session, "Generate Search Response", str(e))
                        with chat_placeholder:
                            with st.chat_message("assistant"):
                                st.markdown(
                                    "An error occurred. Please check logs for details."
                                )
                        st.session_state.cortex_chat.append(
                            {
                                "role": "assistant",
                                "content": "An error occurred. Please check logs for details.",
                            }
                        )
                    st.rerun()

            st.session_state["update_speech_to_text_search"] = update_search_question

            audio_input = speech_to_text(
                language="en",
                start_prompt="🎙️ Ask via voice",
                stop_prompt="🔴 Stop recording",
                just_once=False,
                callback=update_search_question,
                args=(),
                kwargs={},
                key="speech_to_text_search",
            )

        col_audio, col_spacer = st.columns([2, 8])
        with col_audio:
            create_audio_input_for_search()

    if question := st.chat_input("Enter your question"):
        if not columns:
            show_toast_message(
                "Please select columns to display.", position="bottom-right"
            )
            return
        st.session_state.cortex_chat.append({"role": "user", "content": question})
        with chat_placeholder:
            with st.chat_message("user"):
                st.markdown(question)

        # Check for demo response first
        # demo_response = get_demo_chat_response("Search Service", question)
        if demo_response:
            st.session_state.cortex_chat.append(
                {"role": "assistant", "content": demo_response}
            )
            with chat_placeholder:
                with st.chat_message("assistant"):
                    st.markdown(demo_response)
            return

        try:
            root = Root(session)
            service = (
                root.databases[selected_db]
                .schemas[selected_schema]
                .cortex_search_services[selected_service.lower()]
            )
            columns = [col.lower() for col in columns]
            resp = service.search(
                query=question, columns=columns, filter=filter, limit=int(limit)
            )

            retrieved_data = resp

            def get_chat_history():
                start_index = max(0, len(st.session_state.cortex_chat) - slide_window)
                filtered_history = [
                    msg
                    for msg in st.session_state.messages[start_index:]
                    if not msg["content"].startswith("An error occurred")
                ]
                return filtered_history

            chat_history = get_chat_history()

            prompt = f"""
                    You are an AI assistant using Retrieval-Augmented Generation (RAG). Your task is to provide accurate and relevant answers based on the user's question, the retrieved context from a Cortex Search Service, and the prior chat history (if any). Follow these instructions:
                    1. Use the chat history to understand the conversation context, if context is empty, refer retrieved context.
                    2. Use the retrieved context to ground your answer in the provided data (this is in json form, so under the json keys and values fully).
                    3. Answer the question concisely and directly, without explicitly mentioning the sources (chat history or retrieved context) unless asked.   
            
                    Note: Identify if the user is asking a question from the chat history or the retrieved context. If the user is asking a question from the chat history, answer the question based on the chat history. If the user is asking a question from the retrieved context, answer the question based on the retrieved context. If the user is asking a question from the chat history and the retrieved context, answer the question based on the chat history. If the user is asking a question that is not from the chat history or the retrieved context, answer the question based on the chat history.
                    
                    <chat_history>
                    {chat_history}
                    </chat_history>

                    <retrieved_context>
                    {retrieved_data}
                    </retrieved_context>

                    <question>
                    {question}
                    </question>

                    Answer:
                    """

            if prompt:
                prompt = prompt.replace("'", "\\'")
            res = execute_query_and_get_result(
                session, prompt, selected_model, "Generate RAG Response"
            )

            result_json = json.loads(res)
            response_1 = result_json.get("choices", [{}])[0].get(
                "messages", "No messages found"
            )
            st.session_state.cortex_chat.append(
                {"role": "assistant", "content": response_1}
            )
            with chat_placeholder:
                with st.chat_message("assistant"):
                    st.markdown(response_1)
                    # Add voice output for assistant response
                    create_voice_output_button(response_1, "search_response_voice")
        except Exception as e:
            add_log_entry(session, "Generate Search Response", str(e))
            with chat_placeholder:
                with st.chat_message("assistant"):
                    st.markdown("An error occurred. Please check logs for details.")
            st.session_state.cortex_chat.append(
                {
                    "role": "assistant",
                    "content": "An error occurred. Please check logs for details.",
                }
            )


def display_rag_chat(session):
    """RAG chat implementation"""
    slide_window = 20

    # Settings in expander
    with st.expander("Settings", expanded=True):
        st.subheader("Choose Your Embeddings")
        col1, col2 = st.columns(2)
        with col1:
            db_list = list_databases(session)
            default_db = get_default_value("RAG", "database")
            db_index = db_list.index(default_db) if default_db in db_list else 0
            selected_db = st.selectbox("Database", db_list, index=db_index)
        with col2:
            schema_list = list_schemas(session, selected_db)
            default_schema = get_default_value("RAG", "schema")
            schema_index = (
                schema_list.index(default_schema)
                if default_schema in schema_list
                else 0
            )
            selected_schema = st.selectbox("Schema", schema_list, index=schema_index)

        col1, col2 = st.columns(2)
        with col1:
            table_list = list_tables(session, selected_db, selected_schema) or []
            default_table = get_default_value("RAG", "table")
            table_index = (
                table_list.index(default_table) if default_table in table_list else 0
            )
            selected_table = st.selectbox("Table", table_list, index=table_index)
            if "prev_selected_table" not in st.session_state:
                st.session_state.prev_selected_table = selected_table
            if st.session_state.prev_selected_table != selected_table:
                st.session_state.messages = []
                st.session_state.prev_selected_table = selected_table

        with col2:
            if selected_table:
                required_columns = ["Vector_Embeddings"]
                missing_cols = validate_table_columns(
                    session,
                    selected_db,
                    selected_schema,
                    selected_table,
                    required_columns,
                )
                if missing_cols:
                    st.info(
                        "The table is missing vector_embeddings column. Please use the appropriate table."
                    )
                else:
                    default_column = get_default_value("RAG", "vector_column")
                    # Check if default column exists, otherwise use "Vector_Embeddings"
                    column_options = ["Vector_Embeddings"]
                    if default_column and default_column.upper() in [
                        col.upper() for col in column_options
                    ]:
                        column_index = 0  # Default to first option since we only have Vector_Embeddings
                    else:
                        column_index = 0
                    selected_column = st.selectbox(
                        "Column", column_options, index=column_index
                    )

        st.subheader("Choose Your Models")
        col1, col2 = st.columns(2)
        with col1:
            model_types = ["Base", "Fine Tuned"]
            if st.session_state.show_private_preview_models:
                model_types.append("Private Preview")
            model_type = st.selectbox("Model Type", model_types)
        with col2:
            if model_type == "Base":
                selected_model = st.selectbox(
                    "Model", config["default_settings"]["model"]
                )
            elif model_type == "Private Preview":
                selected_model = st.selectbox(
                    "Model",
                    config["default_settings"]["private_preview_models"],
                )
            else:
                fine_tuned_models = fetch_fine_tuned_models(session)
                selected_model = st.selectbox("Model", fine_tuned_models)
        st.info(
            "Use the same embedding type and model consistently when creating embeddings."
        )
        col4, col5 = st.columns(2)
        with col4:
            embeddings = list(config["default_settings"]["embeddings"].keys())
            default_embedding = get_default_value("RAG", "embeddings")
            # Filter out 'CORTEX_SUPPORTED' and get the remaining embedding types
            embedding_options = embeddings[1:]  # Skip 'CORTEX_SUPPORTED'
            embedding_index = (
                embedding_options.index(default_embedding)
                if default_embedding in embedding_options
                else 0
            )
            embedding_type = st.selectbox(
                "Embeddings", embedding_options, index=embedding_index
            )
        with col5:
            default_embedding_model = get_default_value("RAG", "embedding_model")
            model_options = config["default_settings"]["embeddings"][embedding_type]
            model_index = (
                model_options.index(default_embedding_model)
                if default_embedding_model in model_options
                else 0
            )
            embedding_model = st.selectbox(
                "Embedding Model", model_options, index=model_index
            )

    # Chat container
    rag_chat_container = st.container(border=True, height=700)
    with rag_chat_container:
        st.subheader("Chat Messages")
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    rag = st.checkbox("Use your own documents as context?", value=True)

    # Add audio input for chat (only in debug mode)
    if config.get("mode") == "debug":
        # Audio input helper function for chat
        def create_audio_input_for_chat(key_prefix, text_key):
            if f"{key_prefix}_output" not in st.session_state:
                st.session_state[f"{key_prefix}_output"] = ""

            def update_chat_text():
                audio_text = st.session_state.get(f"{key_prefix}_output", "").strip()
                if audio_text:
                    # Directly process the audio input as a chat message
                    st.session_state.messages.append(
                        {"role": "user", "content": audio_text}
                    )
                    with rag_chat_container:
                        with st.chat_message("user"):
                            st.markdown(audio_text)
                    # Process the response immediately
                    try:

                        def get_chat_history():
                            start_index = max(
                                0, len(st.session_state.messages) - slide_window
                            )
                            filtered_history = [
                                msg
                                for msg in st.session_state.messages[start_index:]
                                if not msg["content"].startswith("An error occurred")
                            ]
                            return filtered_history

                        chat_history = get_chat_history()
                        prompt = create_prompt_for_rag(
                            session,
                            audio_text,
                            rag,
                            selected_column,
                            selected_db,
                            selected_schema,
                            selected_table,
                            embedding_type,
                            embedding_model,
                            chat_history,
                        )
                        if prompt:
                            prompt = prompt.replace("'", "\\'")
                        result = execute_query_and_get_result(
                            session,
                            prompt,
                            selected_model,
                            "Generate RAG Response",
                        )
                        result_json = json.loads(result)
                        response = result_json.get("choices", [{}])[0].get(
                            "messages", "No messages found"
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        with rag_chat_container:
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                # Add voice output for assistant response
                                create_voice_output_button(
                                    response, "rag_assistant_response"
                                )
                    except Exception as e:
                        add_log_entry(session, "Generate RAG Response", str(e))
                        st.error(
                            "An error occurred :  Check if same embedding type and model are selected. Please check the logs for details."
                        )
                    st.rerun()

            st.session_state[f"update_{key_prefix}"] = update_chat_text

            audio_input = speech_to_text(
                language="en",
                start_prompt="🎙️ Ask via voice",
                stop_prompt="🔴 Stop recording",
                just_once=False,
                callback=update_chat_text,
                args=(),
                kwargs={},
                key=f"{key_prefix}",
            )

        col_audio, col_spacer = st.columns([2, 8])
        with col_audio:
            create_audio_input_for_chat("speech_to_text_rag_chat", "rag_chat_input")

    if question := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": question})
        with rag_chat_container:
            with st.chat_message("user"):
                st.markdown(question)

        # Check for demo response first
        # demo_response = get_demo_chat_response("RAG", question)
        demo_response = None
        if demo_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": demo_response}
            )
            with rag_chat_container:
                with st.chat_message("assistant"):
                    st.markdown(demo_response)
            return

        try:

            def get_chat_history():
                start_index = max(0, len(st.session_state.cortex_chat) - slide_window)
                filtered_history = [
                    msg
                    for msg in st.session_state.messages[start_index:]
                    if not msg["content"].startswith("An error occurred")
                ]
                return filtered_history

            chat_history = get_chat_history()

            prompt = create_prompt_for_rag(
                session,
                question,
                rag,
                selected_column,
                selected_db,
                selected_schema,
                selected_table,
                embedding_type,
                embedding_model,
                chat_history,
            )
            if prompt:
                prompt = prompt.replace("'", "\\'")
            result = execute_query_and_get_result(
                session, prompt, selected_model, "Generate RAG Response"
            )
            result_json = json.loads(result)
            response = result_json.get("choices", [{}])[0].get(
                "messages", "No messages found"
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            with rag_chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
                    # Add voice output for assistant response
                    create_voice_output_button(response, "rag_response_voice")
        except Exception as e:
            add_log_entry(session, "Generate RAG Response", str(e))
            st.error(
                "An error occurred :  Check if same embedding type and model are selected. Please check the logs for details."
            )


def display_cortex_agent_chat(session):
    """Cortex Agent chat implementation"""
    st.subheader("Chat with Agent")
    agent_manager = CortexAgentManager(session)
    agents = agent_manager.get_all_agents()
    chat_agent_name = st.selectbox(
        "Agent", [agent.name for agent in agents], key="chat_agent_name"
    )
    if chat_agent_name:
        agent = next(a for a in agents if a.name == chat_agent_name)

        # Get the question from session state if it exists, otherwise empty string
        question_value = st.session_state.get("question", "")

        # Add audio input for Cortex Agent (only in debug mode)
        if config.get("mode") == "debug":
            # Audio input helper function for Cortex Agent
            def create_audio_input_for_agent():
                if "speech_to_text_agent_output" not in st.session_state:
                    st.session_state["speech_to_text_agent_output"] = ""

                def update_agent_question():
                    audio_text = st.session_state.get(
                        "speech_to_text_agent_output", ""
                    ).strip()
                    if audio_text:
                        st.session_state["question"] = audio_text
                        st.rerun()

                st.session_state["update_speech_to_text_agent"] = update_agent_question

                audio_input = speech_to_text(
                    language="en",
                    start_prompt="🎙️ Ask via voice",
                    stop_prompt="🔴 Stop recording",
                    just_once=False,
                    callback=update_agent_question,
                    args=(),
                    kwargs={},
                    key="speech_to_text_agent",
                )

            col_audio, col_spacer = st.columns([2, 8])
            with col_audio:
                create_audio_input_for_agent()

        question = st.text_input(
            "Ask a question",
            placeholder="Type your question here...",
            key="question",
            value=question_value,
        )

        if st.button("Send", key="send"):
            if question.strip():
                # Check for demo response first
                # demo_response = get_demo_chat_response("Cortex Agent", question)
                demo_response = None
                if demo_response:
                    with st.chat_message("assistant"):
                        st.markdown(demo_response)
                    st.session_state.setdefault("messages", []).append(
                        {"role": "assistant", "content": demo_response}
                    )
                    return

                with st.spinner("Processing your request..."):
                    text, sql = agent.chat(session, question)
                    if text:
                        with st.chat_message("assistant"):
                            formatted_text = (
                                text.replace("•", "\n\n")
                                .replace("【†", "[")
                                .replace("†】", "]")
                            )  # Format bullet points
                            st.markdown(formatted_text)
                            # Add voice output for assistant response
                            create_voice_output_button(
                                formatted_text, "cortex_agent_response"
                            )
                        st.session_state.setdefault("messages", []).append(
                            {"role": "assistant", "content": text}
                        )
                    if sql:
                        st.markdown("### Generated SQL")
                        st.code(sql, language="sql")
                        sql_result = run_snowflake_query(session, sql)
                        st.write("### Query Results")
                        st.dataframe(sql_result)
            else:
                st.error("Question cannot be empty")
