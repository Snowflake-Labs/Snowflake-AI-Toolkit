import streamlit as st
import json
from src.utils import (
    get_default_value,
    list_databases,
    list_schemas,
    list_stages,
    list_files_in_stage,
    list_tables,
    list_table_columns,
)
from src.query_result_builder import fetch_fine_tuned_models


def get_functionality_settings(functionality, config, session=None):
    """
    Returns settings based on the selected functionality from config.
    """
    settings = {}
    defaults = config["default_settings"]

    if functionality == "Complete":
        col1, col2 = st.columns(2)
        with col1:
            model_types = ["Base", "Fine Tuned"]
            if st.session_state.show_private_preview_models:
                model_types.append("Private Preview")
            model_type = st.selectbox("Model Type", model_types)
        with col2:
            if model_type == "Base":
                settings["model"] = st.selectbox(
                    "Change chatbot model:", defaults["model"]
                )
            elif model_type == "Private Preview":
                settings["model"] = st.selectbox(
                    "Change chatbot model:", defaults["private_preview_models"]
                )
            else:
                fine_tuned_models = fetch_fine_tuned_models(session)
                settings["model"] = st.selectbox(
                    "Change chatbot model:", fine_tuned_models
                )
        settings["temperature"] = st.slider(
            "Temperature:",
            defaults["temperature_min"],
            defaults["temperature_max"],
            defaults["temperature"],
        )
        settings["max_tokens"] = st.slider(
            "Max Tokens:",
            defaults["max_tokens_min"],
            defaults["max_tokens_max"],
            defaults["max_tokens"],
        )
        settings["guardrails"] = st.checkbox(
            "Enable Guardrails", value=defaults["guardrails"]
        )
        if f"system_prompt_{functionality}_data" not in st.session_state:
            st.session_state[f"system_prompt_{functionality}_data"] = ""
        settings["system_prompt"] = st.text_area(
            "System Prompt (optional):",
            placeholder="Enter a system prompt...",
            key=f"system_prompt_{functionality}",
            value=st.session_state[f"system_prompt_{functionality}_data"],
        )

    elif functionality == "Complete Multimodal":
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox(
                "Models", config["default_settings"]["complete_multimodal"]
            )
            settings["model"] = selected_model
        with col2:
            db_list = list_databases(session)
            default_db = get_default_value(functionality, "database")
            db_index = db_list.index(default_db) if default_db in db_list else 0
            selected_db = st.selectbox("Databases", db_list, index=db_index)
            settings["db"] = selected_db
        with col1:
            schema_list = list_schemas(session, selected_db)
            default_schema = get_default_value(functionality, "schema")
            schema_index = (
                schema_list.index(default_schema)
                if default_schema in schema_list
                else 0
            )
            selected_schema = st.selectbox("Schemas", schema_list, index=schema_index)
            settings["schema"] = selected_schema
        with col2:
            stage_list = list_stages(session, selected_db, selected_schema)
            default_stage = get_default_value(functionality, "stage")
            stage_index = (
                stage_list.index(default_stage) if default_stage in stage_list else 0
            )
            selected_stage = st.selectbox("Stage", stage_list, index=stage_index)
            stage = f"{selected_db}.{selected_schema}.{selected_stage}"
            settings["stage"] = stage
        if selected_stage:
            file_list = list_files_in_stage(
                session, selected_db, selected_schema, selected_stage
            )
            file_list = [file.split("/")[-1] for file in file_list]
            # add index to the list, starts from 0
            file_list = [f"{i}: {file}" for i, file in enumerate(file_list)]
            if not file_list:
                st.warning("No files found in the selected stage.")
            else:
                files = st.multiselect("Images", file_list)
                # remove indices from the list
                files = [file.split(": ")[-1] for file in files]
                if not files:
                    st.warning("No files selected.")
                else:
                    settings["files"] = files

    elif functionality == "Translate":
        settings["source_lang"] = st.selectbox("Source Language", defaults["languages"])
        settings["target_lang"] = st.selectbox("Target Language", defaults["languages"])

    elif functionality == "Parse Document":
        col1, col2 = st.columns(2)
        with col1:
            db_list = list_databases(session)
            default_db = get_default_value(functionality, "database")
            db_index = db_list.index(default_db) if default_db in db_list else 0
            selected_db = st.selectbox("Databases", db_list, index=db_index)
            settings["db"] = selected_db
        with col2:
            schema_list = list_schemas(session, selected_db)
            default_schema = get_default_value(functionality, "schema")
            schema_index = (
                schema_list.index(default_schema)
                if default_schema in schema_list
                else 0
            )
            selected_schema = st.selectbox("Schemas", schema_list, index=schema_index)
            settings["schema"] = selected_schema
        with col1:
            stage_list = list_stages(session, selected_db, selected_schema)
            default_stage = get_default_value(functionality, "stage")
            stage_index = (
                stage_list.index(default_stage) if default_stage in stage_list else 0
            )
            selected_stage = st.selectbox("Stage", stage_list, index=stage_index)
            stage = f"@{selected_db}.{selected_schema}.{selected_stage}"
            settings["stage"] = stage
        if selected_stage:
            file_list = list_files_in_stage(
                session, selected_db, selected_schema, selected_stage
            )
            file_list = [file.split("/")[-1] for file in file_list]
            if not file_list:
                st.warning("No files found in the selected stage.")
            else:
                with col2:
                    file = st.selectbox("File", file_list)
                if not file:
                    st.warning("No files selected.")
                else:
                    settings["file"] = file

    elif functionality == "AI Translate":
        settings["source_lang"] = st.selectbox("Source Language", defaults["languages"])
        settings["target_lang"] = st.selectbox("Target Language", defaults["languages"])

    elif functionality == "AI Transcribe":
        col1, col2 = st.columns(2)

        with col1:
            db_list = list_databases(session)
            default_db = get_default_value(functionality, "database")
            db_index = db_list.index(default_db) if default_db in db_list else 0
            settings["db"] = st.selectbox("Database", db_list, index=db_index)
        with col2:
            schema_list = list_schemas(session, settings["db"])
            default_schema = get_default_value(functionality, "schema")
            schema_index = (
                schema_list.index(default_schema)
                if default_schema in schema_list
                else 0
            )
            settings["schema"] = st.selectbox("Schema", schema_list, index=schema_index)

        with col1:
            stage_list = list_stages(session, settings["db"], settings["schema"])
            if stage_list:
                default_stage = get_default_value(functionality, "stage")
                stage_index = (
                    stage_list.index(default_stage)
                    if default_stage in stage_list
                    else 0
                )
                settings["stage"] = st.selectbox("Stage", stage_list, index=stage_index)
                if settings["stage"]:
                    settings["stage"] = (
                        f"@{settings['db']}.{settings['schema']}.{settings['stage']}"
                    )
            else:
                st.warning("No stages found in the selected schema.")
                return None

        if settings.get("stage"):
            file_list = list_files_in_stage(
                session,
                settings["db"],
                settings["schema"],
                settings["stage"].split(".")[-1],
            )

            # Filter for audio files only
            audio_extensions = [".flac", ".mp3", ".ogg", ".wav", ".webm"]
            audio_files = [
                file.split("/")[-1]
                for file in file_list
                if any(file.lower().endswith(ext) for ext in audio_extensions)
            ]

            if not audio_files:
                st.warning(
                    "No audio files found in the selected stage. Supported formats: FLAC, MP3, Ogg, WAV, WebM"
                )
                return None

            with col2:
                default_file = get_default_value(functionality, "file")
                file_index = (
                    audio_files.index(default_file)
                    if default_file in audio_files
                    else 0
                )
                settings["audio_file"] = st.selectbox(
                    "Audio File", audio_files, index=file_index
                )

        # Timestamp granularity options
        granularity_options = ["None", "word", "speaker"]
        granularity_labels = {
            "None": "None (Full transcription only)",
            "word": "Word (Word-level timestamps)",
            "speaker": "Speaker (Speaker identification)",
        }

        selected_granularity = st.selectbox(
            "Timestamp Granularity",
            granularity_options,
            format_func=lambda x: granularity_labels[x],
        )

        settings["timestamp_granularity"] = (
            None if selected_granularity == "None" else selected_granularity
        )

    elif functionality == "AI Parse Document":
        col1, col2 = st.columns(2)

        with col1:
            db_list = list_databases(session)
            default_db = get_default_value(functionality, "database")
            db_index = db_list.index(default_db) if default_db in db_list else 0
            settings["db"] = st.selectbox("Database", db_list, index=db_index)
        with col2:
            schema_list = list_schemas(session, settings["db"])
            default_schema = get_default_value(functionality, "schema")
            schema_index = (
                schema_list.index(default_schema)
                if default_schema in schema_list
                else 0
            )
            settings["schema"] = st.selectbox("Schema", schema_list, index=schema_index)

        with col1:
            stage_list = list_stages(session, settings["db"], settings["schema"])
            default_stage = get_default_value(functionality, "stage")
            stage_index = (
                stage_list.index(default_stage) if default_stage in stage_list else 0
            )
            if stage_list:
                selected_stage = st.selectbox("Stage", stage_list, index=stage_index)
                settings["stage"] = (
                    f"@{settings['db']}.{settings['schema']}.{selected_stage}"
                )
            else:
                st.warning("No stages found in the selected schema.")
                return None

        if settings.get("stage"):
            file_list = list_files_in_stage(
                session,
                settings["db"],
                settings["schema"],
                settings["stage"].split(".")[-1],
            )

            # Filter for document files
            document_extensions = [
                ".pdf",
                ".docx",
                ".pptx",
                ".txt",
                ".rtf",
                ".odt",
                ".xlsx",
                ".ppt",
                ".doc",
            ]
            document_files = [
                file.split("/")[-1]
                for file in file_list
                if any(file.lower().endswith(ext) for ext in document_extensions)
            ]

            if not document_files:
                st.warning(
                    "No document files found in the selected stage. Supported formats: PDF, DOCX, PPTX, TXT, RTF, ODT, XLSX, PPT, DOC"
                )
                # Show all files if no documents found
                all_files = [file.split("/")[-1] for file in file_list]
                if all_files:
                    with col2:
                        default_file = get_default_value(functionality, "file")
                        file_index = (
                            all_files.index(default_file)
                            if default_file in all_files
                            else 0
                        )
                        settings["file"] = st.selectbox(
                            "All Files", all_files, index=file_index
                        )
            else:
                with col2:
                    default_file = get_default_value(functionality, "file")
                    file_index = (
                        document_files.index(default_file)
                        if default_file in document_files
                        else 0
                    )
                    settings["file"] = st.selectbox(
                        "Document File", document_files, index=file_index
                    )

        # Parsing mode options
        mode_options = ["OCR", "LAYOUT"]
        mode_labels = {
            "OCR": "OCR (Text extraction only)",
            "LAYOUT": "LAYOUT (Text + structure, including tables)",
        }

        settings["mode"] = st.selectbox(
            "Parsing Mode", mode_options, format_func=lambda x: mode_labels[x]
        )

        # Advanced options in an expandable section
        with st.expander("🔧 Advanced Options"):
            # Page splitting option
            settings["page_split"] = st.checkbox(
                "Enable Page Splitting",
                value=False,
                help="Process document page by page. Useful for large documents to avoid token limits. Only supports PDF, PPTX, DOCX.",
            )

            # Page filtering (only if page splitting is enabled)
            if settings["page_split"]:
                st.write("**Page Range Selection:**")
                col_start, col_end = st.columns(2)

                with col_start:
                    start_page = st.number_input(
                        "Start Page (0-based)",
                        min_value=0,
                        value=0,
                        step=1,
                        help="First page to process (0 = first page)",
                    )

                with col_end:
                    end_page = st.number_input(
                        "End Page (exclusive)",
                        min_value=start_page + 1,
                        value=start_page + 1,
                        step=1,
                        help="Last page + 1 (e.g., 3 means process until page 2)",
                    )

                # Store page filter settings
                if start_page >= 0 and end_page > start_page:
                    settings["page_filter"] = [{"start": start_page, "end": end_page}]
                else:
                    settings["page_filter"] = None
            else:
                settings["page_filter"] = None

    elif functionality == "AI Similarity":
        cols = st.columns(3)
        col1, col2 = st.columns(2)
        with col1:
            input_type = st.selectbox("Input Type", ["Text", "Image"])
            settings["input_type"] = input_type
        with col2:
            if input_type == "Text":
                default_model = "nv-embed-qa-4"
                model_options = [
                    "snowflake-arctic-embed-l-v2",
                    "nv-embed-qa-4",
                    "multilingual-e5-large",
                    "voyage-multilingual-2",
                    "snowflake-arctic-embed-m-v1.5",
                    "snowflake-arctic-embed-m",
                    "e5-base-v2",
                ]
            else:
                default_model = "voyage-multimodal-3"
                model_options = ["voyage-multimodal-3"]
            settings["model"] = st.selectbox(
                "Embedding Model",
                model_options,
                index=model_options.index(default_model),
            )

        if input_type == "Image":
            with cols[0]:
                db_list = list_databases(session)
                default_db = get_default_value(functionality, "database", "Image")
                db_index = db_list.index(default_db) if default_db in db_list else 0
                selected_db = st.selectbox("Databases", db_list, index=db_index)
                settings["db"] = selected_db
            with cols[1]:
                schema_list = list_schemas(session, selected_db)
                default_schema = get_default_value(functionality, "schema", "Image")
                schema_index = (
                    schema_list.index(default_schema)
                    if default_schema in schema_list
                    else 0
                )
                selected_schema = st.selectbox(
                    "Schemas", schema_list, index=schema_index
                )
                settings["schema"] = selected_schema
            with cols[2]:
                stage_list = list_stages(session, selected_db, selected_schema)
                default_stage = get_default_value(functionality, "stage", "Image")
                stage_index = (
                    stage_list.index(default_stage)
                    if default_stage in stage_list
                    else 0
                )
                selected_stage = st.selectbox("Stage", stage_list, index=stage_index)
                stage = f"@{selected_db}.{selected_schema}.{selected_stage}"
                settings["stage"] = stage
            if selected_stage:
                file_list = list_files_in_stage(
                    session, selected_db, selected_schema, selected_stage
                )
                file_list = [file.split("/")[-1] for file in file_list]
                if not file_list:
                    st.warning("No files found in the selected stage.")
                    return
                with col1:
                    settings["file1"] = st.selectbox("First Image File", file_list)
                with col2:
                    settings["file2"] = st.selectbox("Second Image File", file_list)
                if not settings["file1"] or not settings["file2"]:
                    st.warning("Please select two image files.")
                    return
                if settings["file1"] == settings["file2"]:
                    st.warning("Please select two different image files.")
                    return
            st.warning(
                "Ensure the stage is not encrypted with SNOWFLAKE_FULL, AWS_CSE, or AZURE_CSE, and is not a user or table stage."
            )

        elif input_type == "Text":
            if f"input1_{functionality}_data" not in st.session_state:
                st.session_state[f"input1_{functionality}_data"] = get_default_value(
                    functionality, "first text", "Text"
                )
            if f"input2_{functionality}_data" not in st.session_state:
                st.session_state[f"input2_{functionality}_data"] = get_default_value(
                    functionality, "second text", "Text"
                )

            settings["input1"] = st.text_area(
                "Enter first text:",
                placeholder="Type your first text here...",
                key=f"input1_{functionality}",
                value=st.session_state[f"input1_{functionality}_data"],
            )
            settings["input2"] = st.text_area(
                "Enter second text:",
                placeholder="Type your second text here...",
                key=f"input2_{functionality}",
                value=st.session_state[f"input2_{functionality}_data"],
            )

    elif functionality == "AI Classify":
        col1, col2 = st.columns(2)
        cols = st.columns(2)

        # Get input type from the main interface if available
        if "ai_classify_input_type" in st.session_state:
            input_type = st.session_state["ai_classify_input_type"]
        else:
            with col1:
                input_type = st.selectbox("Input Type", ["Text", "Image"])
        settings["input_type"] = input_type

        if input_type == "Image":
            if f"prompt_{functionality}_data" not in st.session_state:
                st.session_state[f"prompt_{functionality}_data"] = get_default_value(
                    functionality, "prompt", "Image"
                )
            settings["prompt"] = st.text_area(
                "Prompt (optional for image):",
                placeholder="e.g., Please classify the food in this image...",
                key=f"prompt_{functionality}",
                value=st.session_state[f"prompt_{functionality}_data"],
            )
            with cols[0]:
                db_list = list_databases(session)
                default_db = get_default_value(functionality, "database", "Image")
                db_index = db_list.index(default_db) if default_db in db_list else 0
                selected_db = st.selectbox("Databases", db_list, index=db_index)
                settings["db"] = selected_db
            with cols[1]:
                schema_list = list_schemas(session, selected_db)
                default_schema = get_default_value(functionality, "schema", "Image")
                schema_index = (
                    schema_list.index(default_schema)
                    if default_schema in schema_list
                    else 0
                )
                selected_schema = st.selectbox(
                    "Schemas", schema_list, index=schema_index
                )
                settings["schema"] = selected_schema
            with cols[0]:
                stage_list = list_stages(session, selected_db, selected_schema)
                default_stage = get_default_value(functionality, "stage", "Image")
                stage_index = (
                    stage_list.index(default_stage)
                    if default_stage in stage_list
                    else 0
                )
                selected_stage = st.selectbox("Stage", stage_list, index=stage_index)
                stage = f"@{selected_db}.{selected_schema}.{selected_stage}"
                settings["stage"] = stage
            if selected_stage:
                file_list = list_files_in_stage(
                    session, selected_db, selected_schema, selected_stage
                )
                file_list = [file.split("/")[-1] for file in file_list]
                if not file_list:
                    st.warning("No files found in the selected stage.")
                    return
                with cols[1]:
                    default_file = get_default_value(functionality, "image", "Image")
                    file_index = (
                        file_list.index(default_file)
                        if default_file in file_list
                        else 0
                    )
                    settings["file"] = st.selectbox(
                        "Image File", file_list, index=file_index
                    )
                if not settings["file"]:
                    st.warning("Please select an image file.")
                    return
                st.warning(
                    "Ensure the stage is not encrypted with SNOWFLAKE_FULL, AWS_CSE, or AZURE_CSE, and is not a user or table stage."
                )
        else:
            if f"text_{functionality}_data" not in st.session_state:
                st.session_state[f"text_{functionality}_data"] = get_default_value(
                    functionality, "text", "Text"
                )

            settings["text"] = st.text_area(
                "Enter text to classify:",
                placeholder="Type your text here...",
                key=f"text_{functionality}",
                value=st.session_state[f"text_{functionality}_data"],
            )

        with st.expander("Categories", expanded=True):
            if f"categories_{functionality}_data" not in st.session_state:
                st.session_state[f"categories_{functionality}_data"] = (
                    get_default_value(functionality, "categories", "Text")
                )
            category_input = st.text_area(
                "Enter categories (comma-separated or JSON format):",
                placeholder="e.g., positive, neutral, negative or [{'label': 'positive', 'description': 'good vibes'}, ...]",
                key=f"categories_{functionality}",
                value=st.session_state[f"categories_{functionality}_data"],
            )
            try:
                categories = json.loads(category_input)
                if not isinstance(categories, list):
                    raise ValueError("Categories must be a list.")
                if all(isinstance(c, dict) and "label" in c for c in categories):
                    settings["categories"] = categories
                else:
                    settings["categories"] = [
                        cat.strip() for cat in category_input.split(",") if cat.strip()
                    ]
            except json.JSONDecodeError:
                settings["categories"] = [
                    cat.strip() for cat in category_input.split(",") if cat.strip()
                ]
            if not settings["categories"]:
                st.warning("Please provide at least one category.")
                return
            if len(settings["categories"]) > 500:
                st.warning("Maximum 500 categories allowed.")
                return

    elif functionality == "AI Filter":
        col1, col2 = st.columns(2)
        cols = st.columns(2)

        # Get input type from the main interface if available
        if "ai_filter_input_type" in st.session_state:
            input_type = st.session_state["ai_filter_input_type"]
        else:
            with col1:
                input_type = st.selectbox("Input Type", ["Text", "Image"])
        settings["input_type"] = input_type

        if input_type == "Text":
            if f"text_{functionality}_data" not in st.session_state:
                st.session_state[f"text_{functionality}_data"] = get_default_value(
                    functionality, "first text", "Text"
                )
            settings["text"] = st.text_area(
                "Enter text to filter:",
                placeholder="e.g., Is this a positive review?",
                key=f"text_{functionality}",
                value=st.session_state[f"text_{functionality}_data"],
            )
        else:
            if f"predicate_{functionality}_data" not in st.session_state:
                st.session_state[f"predicate_{functionality}_data"] = get_default_value(
                    functionality, "prompt", "Image"
                )
            settings["predicate"] = st.text_area(
                "Enter prompt for image filtering:",
                placeholder="e.g., Is this a picture of a cat?",
                key=f"predicate_{functionality}",
                value=st.session_state[f"predicate_{functionality}_data"],
            )
            with cols[0]:
                db_list = list_databases(session)
                default_db = get_default_value(functionality, "database", "Image")
                db_index = db_list.index(default_db) if default_db in db_list else 0
                selected_db = st.selectbox("Databases", db_list, index=db_index)
                settings["db"] = selected_db
            with cols[1]:
                schema_list = list_schemas(session, selected_db)
                default_schema = get_default_value(functionality, "schema", "Image")
                schema_index = (
                    schema_list.index(default_schema)
                    if default_schema in schema_list
                    else 0
                )
                selected_schema = st.selectbox(
                    "Schemas", schema_list, index=schema_index
                )
                settings["schema"] = selected_schema
            with cols[0]:
                stage_list = list_stages(session, selected_db, selected_schema)
                default_stage = get_default_value(functionality, "stage", "Image")
                stage_index = (
                    stage_list.index(default_stage)
                    if default_stage in stage_list
                    else 0
                )
                selected_stage = st.selectbox("Stage", stage_list, index=stage_index)
                stage = f"@{selected_db}.{selected_schema}.{selected_stage}"
                settings["stage"] = stage
            if selected_stage:
                file_list = list_files_in_stage(
                    session, selected_db, selected_schema, selected_stage
                )
                file_list = [file.split("/")[-1] for file in file_list]
                if not file_list:
                    st.warning("No files found in the selected stage.")
                    return
                with cols[1]:
                    default_file = get_default_value(
                        functionality, "first_image", "Image"
                    )
                    file = st.selectbox(
                        "Image File",
                        file_list,
                        index=file_list.index(default_file)
                        if default_file in file_list
                        else 0,
                    )
                    settings["file"] = file
                if not settings["file"]:
                    st.warning("Please select an image file.")
                    return
                st.warning(
                    "Ensure the stage is not encrypted with SNOWFLAKE_FULL, AWS_CSE, or AZURE_CSE, and is not a user or table stage."
                )

    elif functionality == "AI Agg":
        col1, col2 = st.columns(2)
        cols = st.columns(2)

        # Get input type from the main interface if available
        if "ai_agg_input_type" in st.session_state:
            input_type = st.session_state["ai_agg_input_type"]
        else:
            with col1:
                input_type = st.selectbox("Input Type", ["Text", "Table"])
        settings["input_type"] = input_type

        if settings["input_type"] == "Text":
            if f"text_input_{functionality}_data" not in st.session_state:
                st.session_state[f"text_input_{functionality}_data"] = (
                    get_default_value(functionality, "expression", "Text")
                )
            settings["text_input"] = st.text_area(
                "Enter text input:",
                placeholder="e.g., [Excellent, Great, Mediocre]",
                key=f"text_input_{functionality}",
                value=st.session_state[f"text_input_{functionality}_data"],
            )
            if not settings["text_input"]:
                st.warning("Text input is required for Text mode.")
                return None
        else:
            with cols[0]:
                db_list = list_databases(session)
                default_db = get_default_value(functionality, "database", "Table")
                db_index = db_list.index(default_db) if default_db in db_list else 0
                settings["db"] = st.selectbox("Database", db_list, index=db_index)
            with cols[1]:
                schema_list = list_schemas(session, settings["db"])
                default_schema = get_default_value(functionality, "schema", "Table")
                schema_index = (
                    schema_list.index(default_schema)
                    if default_schema in schema_list
                    else 0
                )
                settings["schema"] = st.selectbox(
                    "Schema", schema_list, index=schema_index
                )
            with cols[0]:
                table_list = list_tables(session, settings["db"], settings["schema"])
                default_table = get_default_value(functionality, "table", "Table")
                table_index = (
                    table_list.index(default_table)
                    if default_table in table_list
                    else 0
                )
                settings["table"] = st.selectbox("Table", table_list, index=table_index)
            if settings["table"]:
                columns = list_table_columns(
                    session, settings["db"], settings["schema"], settings["table"]
                )
                with cols[1]:
                    default_text_column = get_default_value(
                        functionality, "text_column", "Table"
                    )
                    text_column_index = (
                        columns.index(default_text_column)
                        if default_text_column in columns
                        else 0
                    )
                    settings["text_column"] = st.selectbox(
                        "Text Column", columns, index=text_column_index
                    )
                with st.expander("Group By (Optional)", expanded=False):
                    settings["group_by_column"] = st.selectbox(
                        "Group By Column", [None] + columns
                    )
            else:
                st.warning("Please select a table.")
                return None

        # Handle prompt defaults based on input type changes
        current_input_type = settings.get("input_type", "Text")
        if f"last_input_type_{functionality}_agg" not in st.session_state:
            st.session_state[f"last_input_type_{functionality}_agg"] = (
                current_input_type
            )
        elif (
            st.session_state[f"last_input_type_{functionality}_agg"]
            != current_input_type
        ):
            # Input type has changed, reset the prompt session state
            if f"task_description_{functionality}_data" in st.session_state:
                del st.session_state[f"task_description_{functionality}_data"]
            st.session_state[f"last_input_type_{functionality}_agg"] = (
                current_input_type
            )

        if f"task_description_{functionality}_data" not in st.session_state:
            if current_input_type == "Table":
                st.session_state[f"task_description_{functionality}_data"] = (
                    get_default_value(functionality, "prompt_text", "Table")
                )
            else:
                st.session_state[f"task_description_{functionality}_data"] = (
                    get_default_value(functionality, "prompt", "Text")
                )
        settings["task_description"] = st.text_area(
            "Task Description",
            placeholder="e.g., Summarize the product reviews for a blog post targeting consumers",
            key=f"task_description_{functionality}",
            value=st.session_state[f"task_description_{functionality}_data"],
        )
        if not settings["task_description"]:
            st.warning("Task description is required.")
            return None

    elif functionality == "AI Summarize Agg":
        col1, col2 = st.columns(2)
        cols = st.columns(2)

        # Get input type from the main interface if available
        if "ai_summarize_agg_input_type" in st.session_state:
            input_type = st.session_state["ai_summarize_agg_input_type"]
        else:
            with col1:
                input_type = st.selectbox("Input Type", ["Text", "Table"])
        settings["input_type"] = input_type

        if settings["input_type"] == "Text":
            if f"text_input_{functionality}_data" not in st.session_state:
                st.session_state[f"text_input_{functionality}_data"] = (
                    get_default_value(functionality, "expression", "Text")
                )
            settings["text_input"] = st.text_area(
                "Enter text input:",
                placeholder="e.g., [Excellent, Great, Mediocre]",
                key=f"text_input_{functionality}",
                value=st.session_state[f"text_input_{functionality}_data"],
            )
            if not settings["text_input"]:
                st.warning("Text input is required for Text mode.")
                return None
        else:
            with cols[0]:
                db_list = list_databases(session)
                default_db = get_default_value(functionality, "database", "Table")
                db_index = db_list.index(default_db) if default_db in db_list else 0
                settings["db"] = st.selectbox("Database", db_list, index=db_index)
            with cols[1]:
                schema_list = list_schemas(session, settings["db"])
                default_schema = get_default_value(functionality, "schema", "Table")
                schema_index = (
                    schema_list.index(default_schema)
                    if default_schema in schema_list
                    else 0
                )
                settings["schema"] = st.selectbox(
                    "Schema", schema_list, index=schema_index
                )
            with cols[0]:
                table_list = list_tables(session, settings["db"], settings["schema"])
                default_table = get_default_value(functionality, "table", "Table")
                table_index = (
                    table_list.index(default_table)
                    if default_table in table_list
                    else 0
                )
                settings["table"] = st.selectbox("Table", table_list, index=table_index)
            if settings["table"]:
                columns = list_table_columns(
                    session, settings["db"], settings["schema"], settings["table"]
                )
                with cols[1]:
                    default_text_column = get_default_value(
                        functionality, "text_column", "Table"
                    )
                    text_column_index = (
                        columns.index(default_text_column)
                        if default_text_column in columns
                        else 0
                    )
                    settings["text_column"] = st.selectbox(
                        "Text Column", columns, index=text_column_index
                    )
                with st.expander("Group By (Optional)", expanded=False):
                    settings["group_by_column"] = st.selectbox(
                        "Group By Column", [None] + columns
                    )
            else:
                st.warning("Please select a table.")
                return None

    elif functionality == "AI Complete":
        cols = st.columns(3)
        col1, col2 = st.columns(2)
        with cols[0]:
            settings["input_type"] = st.selectbox(
                "Input Type", ["Text", "Image", "Prompt Object"]
            )

        # Reset prompt session state when input type changes
        if f"last_input_type_{functionality}" not in st.session_state:
            st.session_state[f"last_input_type_{functionality}"] = settings[
                "input_type"
            ]
        elif (
            st.session_state[f"last_input_type_{functionality}"]
            != settings["input_type"]
        ):
            # Input type has changed, reset the prompt session state
            if f"prompt_{functionality}_data" in st.session_state:
                del st.session_state[f"prompt_{functionality}_data"]
            st.session_state[f"last_input_type_{functionality}"] = settings[
                "input_type"
            ]

        with cols[1]:
            model_types = ["Base", "Fine Tuned"]
            if st.session_state.show_private_preview_models:
                model_types.append("Private Preview")
            model_type = st.selectbox("Model Type", model_types)
            if model_type == "Base":
                if settings["input_type"] == "Image":
                    models = [
                        "claude-4-opus",
                        "claude-4-sonnet",
                        "claude-3-7-sonnet",
                        "claude-3-5-sonnet",
                        "llama4-maverick",
                        "llama4-scout",
                        "openai-o4-mini",
                        "openai-gpt-4.1",
                        "pixtral-large",
                    ]
                    with cols[2]:
                        settings["model"] = st.selectbox(
                            "Change chatbot model:", models
                        )
                else:
                    with cols[2]:
                        settings["model"] = st.selectbox(
                            "Change chatbot model:", defaults["model"]
                        )
            elif model_type == "Private Preview":
                settings["model"] = st.selectbox(
                    "Change chatbot model:", defaults["private_preview_models"]
                )
            else:
                fine_tuned_models = fetch_fine_tuned_models(session)
                settings["model"] = st.selectbox(
                    "Change chatbot model:", fine_tuned_models
                )

        col1, col2 = st.columns(2)
        with col1:
            settings["show_details"] = st.checkbox("Show Detailed Output", value=False)
        with col2:
            settings["response_format_checkbox"] = st.checkbox(
                "Change Response Format", value=False
            )

        with st.expander("Model Parameters", expanded=False):
            settings["temperature"] = st.slider(
                "Temperature:", 0.0, 1.0, defaults["temperature"]
            )
            settings["top_p"] = st.slider("Top P:", 0.0, 1.0, 0.0)
            settings["max_tokens"] = st.slider(
                "Max Tokens:", 1, 8192, defaults["max_tokens"]
            )
            settings["guardrails"] = st.checkbox(
                "Enable Guardrails", value=defaults["guardrails"]
            )

        if settings["input_type"] == "Text":
            if f"prompt_{functionality}_data" not in st.session_state:
                st.session_state[f"prompt_{functionality}_data"] = get_default_value(
                    functionality, "prompt", "Text"
                )
            settings["prompt"] = st.text_area(
                "Enter a prompt:",
                placeholder="e.g., What are large language models?",
                key=f"prompt_{functionality}",
                value=st.session_state[f"prompt_{functionality}_data"],
            )
            st.session_state[f"prompt_{functionality}_data"] = settings["prompt"]

        elif settings["input_type"] == "Image":
            with col1:
                db_list = list_databases(session)
                default_db = get_default_value(functionality, "database", "Image")
                db_index = db_list.index(default_db) if default_db in db_list else 0
                settings["db"] = st.selectbox("Database", db_list, index=db_index)
            with col2:
                schema_list = list_schemas(session, settings["db"])
                default_schema = get_default_value(functionality, "schema", "Image")
                schema_index = (
                    schema_list.index(default_schema)
                    if default_schema in schema_list
                    else 0
                )
                settings["schema"] = st.selectbox(
                    "Schema", schema_list, index=schema_index
                )
            with col1:
                stage_list = list_stages(session, settings["db"], settings["schema"])
                default_stage = get_default_value(functionality, "stage", "Image")
                stage_index = (
                    stage_list.index(default_stage)
                    if default_stage in stage_list
                    else 0
                )
                settings["stage"] = st.selectbox("Stage", stage_list, index=stage_index)
                if settings["stage"]:
                    settings["stage"] = (
                        f"@{settings['db']}.{settings['schema']}.{settings['stage']}"
                    )
            if settings["stage"]:
                file_list = list_files_in_stage(
                    session,
                    settings["db"],
                    settings["schema"],
                    settings["stage"].split(".")[-1],
                )
                file_list = [file.split("/")[-1] for file in file_list]
                if not file_list:
                    st.warning("No files found in the selected stage.")
                    return None
                with col2:
                    default_file = get_default_value(functionality, "image", "Image")
                    file_index = (
                        file_list.index(default_file)
                        if default_file in file_list
                        else 0
                    )
                    settings["file"] = st.selectbox(
                        "Image File", file_list, index=file_index
                    )
                if not settings["file"]:
                    st.warning("Please select an image file.")
            if f"predicate_{functionality}_data" not in st.session_state:
                st.session_state[f"predicate_{functionality}_data"] = get_default_value(
                    functionality, "prompt", "Image"
                )
            settings["predicate"] = st.text_area(
                "Enter a prompt:",
                placeholder="e.g., Summarize the input image in no more than 2 words.",
                key=f"predicate_{functionality}",
                value=st.session_state[f"predicate_{functionality}_data"],
            )
            if not settings["predicate"]:
                st.warning("Prompt is required for Image input.")
            st.warning(
                "Ensure the stage has server-side encryption enabled and is not client-side encrypted."
            )

        else:  # Prompt Object
            with col1:
                db_list = list_databases(session)
                default_db = get_default_value(
                    functionality, "database", "Prompt Object"
                )
                db_index = db_list.index(default_db) if default_db in db_list else 0
                settings["db"] = st.selectbox("Database", db_list, index=db_index)
            with col2:
                schema_list = list_schemas(session, settings["db"])
                default_schema = get_default_value(
                    functionality, "schema", "Prompt Object"
                )
                schema_index = (
                    schema_list.index(default_schema)
                    if default_schema in schema_list
                    else 0
                )
                settings["schema"] = st.selectbox(
                    "Schema", schema_list, index=schema_index
                )
            with col1:
                stage_list = list_stages(session, settings["db"], settings["schema"])
                default_stage = get_default_value(
                    functionality, "stage", "Prompt Object"
                )
                stage_index = (
                    stage_list.index(default_stage)
                    if default_stage in stage_list
                    else 0
                )
                settings["stage"] = st.selectbox("Stage", stage_list, index=stage_index)
            if settings["stage"]:
                columns = list_files_in_stage(
                    session, settings["db"], settings["schema"], settings["stage"]
                )
                with col2:
                    default_multi_images = get_default_value(
                        functionality, "multi_images", "Prompt Object"
                    )
                    # Convert to list if it's not already
                    if isinstance(default_multi_images, list):
                        default_selection = [
                            img for img in default_multi_images if img in columns
                        ]
                    else:
                        default_selection = []
                    settings["multi_images"] = st.multiselect(
                        "Select Images", columns, default=default_selection
                    )
                    if not settings["multi_images"]:
                        st.warning("Please select at least one image.")
                # with col1:
                settings["file_column"] = "None"
            else:
                st.warning("Please select a stage.")

            # Add prompt input for Prompt Object mode
            if f"prompt_{functionality}_data" not in st.session_state:
                st.session_state[f"prompt_{functionality}_data"] = get_default_value(
                    functionality, "prompt_text", "Prompt Object"
                )
            settings["prompt"] = st.text_area(
                "Enter a prompt:",
                placeholder="e.g., Process the prompts in the selected column...",
                key=f"prompt_{functionality}",
                value=st.session_state[f"prompt_{functionality}_data"],
            )
            if not settings["prompt"]:
                st.warning("Prompt is required for Prompt Object input.")

        if settings["response_format_checkbox"]:
            with st.expander("Response Format (Optional)", expanded=False):
                if f"response_format_{functionality}_data" not in st.session_state:
                    st.session_state[f"response_format_{functionality}_data"] = ""
                response_format_input = st.text_area(
                    "JSON Schema (optional):",
                    placeholder="e.g., {'type': 'json', 'schema': {'type': 'object', 'properties': {...}}}",
                    key=f"response_format_{functionality}",
                    value=st.session_state[f"response_format_{functionality}_data"],
                )

                if response_format_input:
                    try:
                        settings["response_format"] = json.loads(response_format_input)
                    except json.JSONDecodeError:
                        st.error(
                            "Invalid JSON schema. Please provide a valid JSON object."
                        )
                        return None

    return settings
