# ruff: noqa

import streamlit as st
import json
from src.cortex_functions import *
from snowflake.snowpark.exceptions import SnowparkSQLException
from src.query_result_builder import *
from snowflake.core import Root
from src.utils import *
from pathlib import Path
from src.cortex_agent import *
from src.audio_components import *
from src.display_chat import display_chat
from src.display_aisql import get_playground_input
from src.playground_settings import get_functionality_settings


# Load the config file
config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Conditionally import streamlit_mic_recorder only in debug mode
if config.get("mode") == "debug":
    from streamlit_mic_recorder import speech_to_text

if "legacy_function" not in st.session_state:
    st.session_state.legacy_function = False

if "show_private_preview_models" not in st.session_state:
    st.session_state.show_private_preview_models = False


def execute_functionality(session, functionality, input_data, settings):
    """
    Executes the selected functionality in playground mode.
    """
    # Check for demo responses first
    input_text = ""
    print(f"execute Functionality: {functionality}, Input Data: {input_data}")
    if input_data:
        # Extract text from various input types for keyword matching
        if "prompt" in input_data:
            input_text = input_data["prompt"]
        elif "text" in input_data:
            input_text = input_data["text"]
        elif "query" in input_data:
            input_text = input_data["query"]
        elif "predicate" in input_data:
            input_text = input_data["predicate"]
    if settings:
        if "prompt" in settings:
            input_text = settings["prompt"]
        elif "text" in settings:
            input_text = settings["text"]
        elif "query" in settings:
            input_text = settings["query"]
        elif "predicate" in settings:
            input_text = settings["predicate"]

    # Check if audio input is available for this functionality (same logic as in playground)
    has_audio_input = False
    if config.get("mode") == "debug":
        if functionality == "Complete":
            has_audio_input = True
        elif functionality == "Complete Multimodal":
            has_audio_input = True
        elif functionality == "Translate":
            has_audio_input = True
        elif functionality == "Extract":
            has_audio_input = True
        elif functionality == "Sentiment":
            # Check if entity sentiment toggle is active
            if not input_data.get("toggle", False):
                has_audio_input = True
        # elif functionality == "AI Classify":
        #     # Check input type for AI Classify
        #     if settings and settings.get('input_type') == "Image":
        #         has_audio_input = True
        elif functionality == "AI Complete":
            # AI Complete has audio input and enhancement
            has_audio_input = True
        elif functionality == "AI Translate":
            has_audio_input = True
        elif functionality == "AI Transcribe":
            # AI Transcribe doesn't need audio input (it processes audio files)
            has_audio_input = False

    if functionality == "Complete":
        result_json = get_complete_result(
            session,
            settings["model"],
            input_data["prompt"],
            settings["temperature"],
            settings["max_tokens"],
            settings["guardrails"],
            settings["system_prompt"],
        )
        result_formatted = format_result(result_json)
        st.write("Completion Result")
        # Display "Messages:" with inline voice button if audio input is available

        create_inline_label_with_voice(
            "Play Audio", result_formatted["messages"], "complete_result"
        )
        st.success(result_formatted["messages"])

    elif functionality == "Complete Multimodal":
        result = get_complete_multimodal_result(
            session,
            settings["model"],
            input_data["prompt"],
            settings["stage"],
            settings["files"],
        )
        # st.write("Completion Multimodal Result")
        # Add voice output button only if audio input is available - placed ABOVE the result

        create_inline_label_with_voice(
            "Play Audio", result, "complete_multimodal_result"
        )
        if len(settings["files"]) == 1:
            path = f"@{settings['stage']}/{settings['files'][0]}"
            image = session.file.get_stream(path, decompress=False).read()
            st.image(image)
        st.write(result)

    elif functionality == "Translate":
        result = get_translation(
            session,
            input_data["text"],
            settings["source_lang"],
            settings["target_lang"],
        )
        # Display inline label with voice button if audio input is available

        create_inline_label_with_voice("Play Audio", str(result), "translate_result")
        st.write(f"{result}")

    elif functionality == "Summarize":
        result = get_summary(session, input_data["text"])
        create_inline_label_with_voice(
            "Play Audio", str(result), "summarize_result_inline"
        )
        st.write(f"**Summary:** {result}")

    elif functionality == "Extract":
        result = get_extraction(session, input_data["text"], input_data["query"])
        # Display inline label with voice button if audio input is available
        create_inline_label_with_voice("Play Audio", str(result), "extract_result")
        st.write(f"{result}")

    elif functionality == "Sentiment":
        if input_data["toggle"]:
            result = get_entity_sentiment(
                session, input_data["text"], input_data["entities"]
            )
            create_inline_label_with_voice(
                "Play Audio", str(result), "entity_sentiment_result"
            )
            st.write(f"**Entity Sentiment Analysis Result:** {result}")
        else:
            result = get_sentiment(session, input_data["text"])
            # Display inline label with voice button if audio input is available
            create_inline_label_with_voice(
                "Play Audio", str(result), "sentiment_result"
            )
            st.write(f"{result}")

    elif functionality == "Classify Text":
        result = get_classification(
            session, input_data["text"], input_data["categories"]
        )
        create_inline_label_with_voice("Play Audio", str(result), "classify_result")
        st.write(f"**Classification Result:** {result}")

    elif functionality == "Parse Document":
        result = get_parse_document(
            session, settings["stage"], settings["file"], input_data["mode"]
        )
        st.write(f"**Parsed Document Result:**")
        # print(result)
        res = json.loads(result)
        create_inline_label_with_voice(
            "Play Audio", str(res["content"]), "parse_document_result"
        )
        st.write(res["content"])

    elif functionality == "AI Similarity":
        try:
            # Build config object
            config_object = {}
            if settings.get("model"):
                config_object["model"] = settings["model"]

            # Execute AI_SIMILARITY
            result = get_ai_similarity_result(
                session,
                settings["input1"] if settings["input_type"] == "Text" else None,
                settings["input2"] if settings["input_type"] == "Text" else None,
                config_object,
                settings["input_type"],
                settings.get("stage"),
                settings.get("file1"),
                settings.get("file2"),
            )

            # Display results
            st.write("**Similarity Score:**")
            similarity_text = f"Similarity score: {result:.4f} (Range: -1 to 1)"
            create_inline_label_with_voice(
                "Play Audio", similarity_text, "ai_similarity_result"
            )
            st.write(f"{result:.4f} (Range: -1 to 1)")
        except ValueError as e:
            st.error(f"Input Error: {e}")
        except SnowparkSQLException as e:
            st.error(f"SQL Error executing AI_SIMILARITY: {e}")
            st.write("Check the console for the generated SQL query.")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

    elif functionality == "AI Classify":
        try:
            config_object = {}
            if settings.get("task_description"):
                config_object["task_description"] = settings["task_description"]
            if settings.get("output_mode"):
                config_object["output_mode"] = settings["output_mode"]
            if settings.get("examples"):
                config_object["examples"] = settings["examples"]

            if settings["input_type"] == "Image":
                input_data_for_query = {"prompt": settings.get("prompt", "")}
                result = get_ai_classify_result(
                    session,
                    input_data_for_query,
                    settings["categories"],
                    config_object,
                    input_type=settings["input_type"],
                    stage=settings.get("stage"),
                    file_name=settings.get("file"),
                )
                if settings.get("file"):
                    path = f"{settings['stage']}/{settings['file']}"
                    image = session.file.get_stream(path, decompress=False).read()
                    st.image(image)
            else:
                input_data_for_query = settings.get("text", "")
                result = get_ai_classify_result(
                    session,
                    input_data_for_query,
                    settings["categories"],
                    config_object,
                    input_type=settings["input_type"],
                )

            # Display results
            st.write("**Classification Result:**")
            try:
                result_json = json.loads(result)
                create_inline_label_with_voice(
                    "Play Audio:", str(result_json), "classify_result"
                )
                st.json(result_json)
                # Add voice output button only if audio input is available
                if has_audio_input and settings["input_type"] == "Image":
                    # Convert JSON result to readable text for voice output
                    result_text = (
                        json.dumps(result_json, indent=2)
                        if isinstance(result_json, dict)
                        else str(result_json)
                    )
                    create_inline_label_with_voice(
                        "Play Audio:", result_text, "classify_result"
                    )

            except json.JSONDecodeError:
                st.write(result)
                # Add voice output button only if audio input is available
                create_inline_label_with_voice(
                    "Play Audio:", str(result), "ai_classify_result"
                )
        except ValueError as e:
            st.error(f"Input Error: {e}")
        except SnowparkSQLException as e:
            st.error(f"SQL Error executing AI_CLASSIFY: {e}")
            st.write("Check the console for the generated SQL query.")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

    elif functionality == "AI Filter":
        try:
            # Build config object
            config_object = {}
            if settings.get("task_description"):
                config_object["task_description"] = settings["task_description"]
            if settings.get("output_mode"):
                config_object["output_mode"] = settings["output_mode"]
            if settings.get("examples"):
                config_object["examples"] = settings["examples"]

            if settings["input_type"] == "Image":
                if not settings.get("predicate"):
                    raise ValueError("Prompt is required for image input.")
                input_data_for_query = settings.get("predicate", "")
                result = get_ai_filter_result(
                    session, input_data_for_query, settings["stage"], settings["file"]
                )
                # Display the image
                path = f"{settings['stage']}/{settings['file']}"
                image = session.file.get_stream(path, decompress=False).read()
                st.image(image)
            else:
                input_data_for_query = settings.get("text", "")
                result = get_ai_filter_result(session, input_data_for_query)

            # Display results
            st.write("**Filter Result:**")
            try:
                result_json = json.loads(result)
                create_inline_label_with_voice(
                    "Play Audio:", str(result_json), "ai_filter_result"
                )
                st.json(result_json)
            except json.JSONDecodeError:
                st.write(result)
        except ValueError as e:
            st.error(f"Input Error: {e}")
        except SnowparkSQLException as e:
            st.error(f"SQL Error executing AI_FILTER: {e}")
            st.write("Check the console for the generated SQL query.")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

    elif functionality == "AI Agg":
        try:
            if settings["input_type"] == "Text":
                # Execute AI_AGG on text input
                query = f"""
                    SELECT AI_AGG('{settings["text_input"].replace("'", "''")}', '{settings["task_description"].replace("'", "''")}')
                """
                result = session.sql(query).collect()
                create_inline_label_with_voice(
                    "Play Audio:", str(result), "ai_agg_result"
                )
                st.write("**Aggregation Result:**")
                st.write(result[0][0])

            else:  # Table input
                table_name = (
                    f"{settings['db']}.{settings['schema']}.{settings['table']}"
                )
                text_column = settings["text_column"]
                task_description = settings["task_description"].replace("'", "''")

                if settings.get("group_by_column"):
                    # Grouped aggregation
                    group_by_column = settings["group_by_column"]
                    query = f"""
                        SELECT {group_by_column},
                               AI_AGG({text_column}, '{task_description}') AS summarized_result
                        FROM {table_name}
                        GROUP BY {group_by_column}
                    """
                    result = session.sql(query).collect()
                    create_inline_label_with_voice(
                        "Play Audio:", str(result), "ai_agg_result"
                    )
                    st.write("**Grouped Aggregation Results:**")
                    st.dataframe(result)

                else:
                    # Simple aggregation
                    query = f"""
                        SELECT AI_AGG({text_column}, '{task_description}') AS summarized_result
                        FROM {table_name}
                    """
                    result = session.sql(query).collect()
                    create_inline_label_with_voice(
                        "Play Audio:", str(result), "ai_agg_result"
                    )
                    st.write("**Aggregation Result:**")
                    st.write(result[0][0])
        except SnowparkSQLException as e:
            st.error(f"SQL Error executing AI_AGG: {e}")
            st.write("Check the console for the generated SQL query.")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

    elif functionality == "AI Summarize Agg":
        try:
            if settings["input_type"] == "Text":
                # Execute AI_SUMMARIZE_AGG on text input
                query = f"""
                    SELECT AI_SUMMARIZE_AGG('{settings["text_input"].replace("'", "''")}')
                """
                result = session.sql(query).collect()
                create_inline_label_with_voice(
                    "Play Audio:", str(result[0][0]), "ai_summarize_agg_result"
                )
                st.write("**Summary Result:**")
                st.write(result[0][0])

            else:  # Table input
                table_name = (
                    f"{settings['db']}.{settings['schema']}.{settings['table']}"
                )
                text_column = settings["text_column"]

                if settings.get("group_by_column"):
                    # Grouped summary
                    group_by_column = settings["group_by_column"]
                    query = f"""
                        SELECT {group_by_column},
                               AI_SUMMARIZE_AGG({text_column}) AS summarized_result
                        FROM {table_name}
                        GROUP BY {group_by_column}
                    """
                    result = session.sql(query).collect()
                    create_inline_label_with_voice(
                        "Play Audio:", str(result), "ai_summarize_result"
                    )
                    st.write("**Grouped Summary Results:**")
                    st.dataframe(result)
                else:
                    # Simple summary
                    query = f"""
                        SELECT AI_SUMMARIZE_AGG({text_column}) AS summarized_result
                        FROM {table_name}
                    """
                    result = session.sql(query).collect()
                    create_inline_label_with_voice(
                        "Play Audio:", str(result), "ai_summarize_result"
                    )
                    st.write("**Summary Result:**")
                    st.write(result[0][0])
        except SnowparkSQLException as e:
            st.error(f"SQL Error executing AI_SUMMARIZE_AGG: {e}")
            st.write("Check the console for the generated SQL query.")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

    elif functionality == "AI Complete":
        try:
            # Build model parameters
            model_parameters = {}
            if settings.get("temperature"):
                model_parameters["temperature"] = settings["temperature"]
            if settings.get("top_p"):
                model_parameters["top_p"] = settings["top_p"]
            if settings.get("max_tokens"):
                model_parameters["max_tokens"] = settings["max_tokens"]
            if settings.get("guardrails"):
                model_parameters["guardrails"] = settings["guardrails"]

            if settings["input_type"] == "Text":
                escaped_prompt = settings["prompt"].replace("'", "''")
                query_parts = [
                    "SELECT AI_COMPLETE(",
                    f"'{settings['model']}',",
                    f"'{escaped_prompt}'",
                ]

                if model_parameters:
                    query_parts.append(
                        f", PARSE_JSON('{json.dumps(model_parameters)}')"
                    )
                else:
                    query_parts.append(", NULL")

                if settings.get("response_format"):
                    query_parts.append(
                        f", PARSE_JSON('{json.dumps(settings['response_format'])}')"
                    )
                else:
                    query_parts.append(", NULL")

                if settings.get("show_details") is not None:
                    query_parts.append(
                        f", {str(settings['show_details']).upper()}"
                    )  # TRUE / FALSE
                else:
                    query_parts.append(", FALSE")

                query = "".join(query_parts) + ")"
                result = session.sql(query).collect()[0][0]
                create_inline_label_with_voice(
                    "Play Audio:", str(result), "ai_complete_result"
                )
                # Display results
                st.write("**Completion Result:**")
                if settings.get("show_details") or settings.get("response_format"):
                    try:
                        result_json = json.loads(result)
                        st.json(result_json)
                        # Add voice output button only if audio input is available
                        if has_audio_input:
                            # Convert JSON result to readable text for voice output
                            result_text = (
                                json.dumps(result_json, indent=2)
                                if isinstance(result_json, dict)
                                else str(result_json)
                            )
                    except json.JSONDecodeError:
                        st.write(result)
                else:
                    result = result.replace('"', "")
                    st.write(result)

            elif settings["input_type"] == "Image":
                # AI_COMPLETE with single image
                query = f"""
                    SELECT AI_COMPLETE(
                        '{settings["model"]}',
                        '{settings["predicate"].replace("'", "''")}',
                        TO_FILE('{settings["stage"]}', '{settings["file"]}'),
                        '{json.dumps(model_parameters)}'::VARIANT
                    )
                """
                result = session.sql(query).collect()[0][0]

                # Display image and result
                create_inline_label_with_voice(
                    "Play Audio:", str(result), "ai_complete_result"
                )
                st.write("**Input Image:**")
                path = f"{settings['stage']}/{settings['file']}"
                image = session.file.get_stream(path, decompress=False).read()
                st.image(image)
                st.write("**Completion Result:**")
                try:
                    result_json = json.loads(result)
                    st.json(result_json)
                    # Add voice output button only if audio input is available
                    if has_audio_input:
                        result_text = (
                            json.dumps(result_json, indent=2)
                            if isinstance(result_json, dict)
                            else str(result_json)
                        )
                except json.JSONDecodeError:
                    st.write(result)

            else:  # Prompt Object
                # AI_COMPLETE with multiple images from stage
                multi_images = settings.get("multi_images", [])
                prompt_text = settings.get("prompt", "")

                if multi_images and prompt_text:
                    st.write("**Selected Images:**")
                    # Display all selected images
                    cols = st.columns(
                        min(len(multi_images), 3)
                    )  # Max 3 columns for display
                    for idx, image_file in enumerate(multi_images):
                        with cols[idx % 3]:
                            try:
                                path = f"{settings['stage']}/{image_file}"
                                image = session.file.get_stream(
                                    path, decompress=False
                                ).read()
                                st.image(
                                    image, caption=image_file, use_column_width=True
                                )
                            except Exception as e:
                                st.error(f"Failed to load image {image_file}: {e}")

                    st.write("**Completion Results:**")
                    # Process each image with the prompt
                    results = []
                    for image_file in multi_images:
                        try:
                            query = f"""
                                SELECT AI_COMPLETE(
                                    '{settings["model"]}',
                                    '{prompt_text.replace("'", "''")}',
                                    TO_FILE('{settings["stage"]}', '{image_file}'),
                                    '{json.dumps(model_parameters)}'::VARIANT
                                ) AS result
                            """
                            result = session.sql(query).collect()[0][0]
                            results.append({"image": image_file, "result": result})
                        except Exception as e:
                            results.append(
                                {"image": image_file, "result": f"Error: {e}"}
                            )

                    # Display results in a dataframe
                    st.dataframe(results)
                else:
                    st.warning("Please select images and enter a prompt.")

        except SnowparkSQLException as e:
            st.error(f"SQL Error executing AI_COMPLETE: {e}")
            st.write("Check the console for the generated SQL query.")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

    elif functionality == "AI Translate":
        result = get_ai_translation(
            session,
            input_data["text"],
            settings["source_lang"],
            settings["target_lang"],
        )
        # Display inline label with voice button if audio input is available

        create_inline_label_with_voice("Play Audio", str(result), "translate_result")
        st.write(f"{result}")

    elif functionality == "AI Transcribe":
        result = get_ai_transcribe_result(
            session,
            settings["stage"],
            settings["audio_file"],
            settings.get("timestamp_granularity"),
        )

        if "error" in result:
            st.error(f"Transcription Error: {result['error']}")
        else:
            st.write("**Transcription Result:**")

            # Display audio duration
            if "audio_duration" in result:
                st.info(f"Audio Duration: {result['audio_duration']:.2f} seconds")

            # Display full transcription text with voice playback
            if "text" in result:
                # Add download button for transcription text
                audio_filename = settings.get("audio_file", "transcription")
                # Remove file extension if present
                if "." in audio_filename:
                    audio_filename = audio_filename.rsplit(".", 1)[0]

                create_inline_label_with_voice(
                    "Play Audio", result["text"], "ai_transcribe_result"
                )
                download_transcript(result["text"], audio_filename)
                st.success(f"**Full Transcription:** {result['text']}")

            # Display segments if available
            if "segments" in result and result["segments"]:
                st.write("**Detailed Segments:**")
                with st.expander("Show/Hide Segments", expanded=False):
                    for i, segment in enumerate(result["segments"]):
                        st.markdown(
                            f"**Segment {i + 1}** ({segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s)"
                        )
                        if "speaker_label" in segment:
                            st.write(f"**Speaker:** {segment['speaker_label']}")
                        st.write(f"**Text:** {segment.get('text', 'N/A')}")
                        if (
                            i < len(result["segments"]) - 1
                        ):  # Don't add separator after last segment
                            st.markdown("---")

    elif functionality == "AI Parse Document":
        result = get_ai_parse_document(
            session,
            settings["stage"],
            settings["file"],
            settings.get("mode", "OCR"),
            settings.get("page_split", False),
            settings.get("page_filter"),
        )

        if "error" in result:
            st.error(f"Document Parsing Error: {result['error']}")
        else:
            st.write("**AI Parse Document Result:**")

            # Display metadata if available
            if "metadata" in result:
                metadata = result["metadata"]
                st.info(
                    f"📄 **Document Info:** {metadata.get('pageCount', 'Unknown')} pages"
                )

            # Check if document was split into pages (pages array in result)
            if (
                "pages" in result
                and result["pages"]
                and isinstance(result["pages"], list)
            ):
                st.write("**Document Pages:**")

                # Create tabs for each page
                page_tabs = []
                for i, page in enumerate(result["pages"]):
                    page_tabs.append(f"Page {page.get('index', i) + 1}")

                if len(page_tabs) > 1:
                    selected_tab = st.selectbox("Select Page:", page_tabs)
                    selected_page_index = page_tabs.index(selected_tab)
                    page = result["pages"][selected_page_index]
                else:
                    page = result["pages"][0]

                # Display page content with voice playback
                page_content = page.get("content", "No content available")
                create_inline_label_with_voice(
                    "Play Audio",
                    page_content,
                    f"ai_parse_document_page_{page.get('index', 0)}",
                )

                # Display content based on mode
                if settings.get("mode") == "LAYOUT":
                    st.markdown("**Page Content (Formatted):**")
                    st.markdown(page_content)
                else:
                    st.text_area(
                        "**Page Content (Text):**",
                        page_content,
                        height=300,
                        disabled=True,
                    )

                # Show page index info
                if "index" in page:
                    st.caption(f"Page Index: {page['index']}")

            elif "content" in result and result["content"]:
                # Single document content (the most common case)
                content = result["content"]
                if content and content.strip():  # Check if content is not empty
                    create_inline_label_with_voice(
                        "Play Audio", content, "ai_parse_document_content"
                    )

                    # Display content based on mode
                    if settings.get("mode") == "LAYOUT":
                        st.markdown("**Document Content (Formatted):**")
                        st.markdown(content)
                    else:
                        st.markdown("**Document Content (Text):**")
                        st.markdown(content)

                    # Show content statistics
                    st.caption(f"Content length: {len(content)} characters")
                else:
                    st.warning("Document was parsed but contains no text content.")
            else:
                st.warning("No content found in the parsed document.")
                # Debug info for troubleshooting
                st.write("**Debug Info:**")
                st.write(
                    f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dictionary'}"
                )
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key != "content":  # Don't show content again
                            st.write(
                                f"- {key}: {type(value)} ({len(str(value)) if value else 0} chars)"
                            )

            # Display error information if present
            if "errorInformation" in result:
                st.error(f"**Parsing Issues:** {result['errorInformation']}")


def display_playground(session):
    """
    Displays the playground mode interface in Streamlit.
    """
    st.title("AI Playground")

    slide_window = 20

    functionality_list = []

    if st.session_state.legacy_function:
        # Use dynamic column layout based on whether Input Type is needed
        functionality_list = sorted(
            [
                "Complete",
                "Complete Multimodal",
                "Translate",
                "Summarize",
                "Extract",
                "Sentiment",
                "Classify Text",
                "Parse Document",
                "AI Complete",
                "AI Similarity",
                "AI Classify",
                "AI Filter",
                "AI Agg",
                "AI Summarize Agg",
                "AI Translate",
                "AI Transcribe",
                "AI Parse Document",
            ]
        )
    else:
        functionality_list = sorted(
            [
                "Extract",
                "Sentiment",
                "AI Complete",
                "AI Similarity",
                "AI Classify",
                "AI Filter",
                "AI Agg",
                "AI Summarize Agg",
                "AI Translate",
                "AI Transcribe",
                "AI Parse Document",
            ]
        )

    # First get initial selections
    initial_col1, initial_col2 = st.columns(2)
    with initial_col1:
        choices = st.selectbox("Choose Functionality", ["AISQL Functions", "Chat"])

    if choices == "AISQL Functions":
        with initial_col2:
            functionality = st.selectbox("Choose functionality:", functionality_list)

        # Check if Input Type is needed for this functionality
        needs_input_type = functionality in [
            "AI Classify",
            "AI Filter",
            "AI Agg",
            "AI Summarize Agg",
        ]

        if needs_input_type:
            # Add Input Type in a new row below
            input_type_col = st.columns(1)[0]
            with input_type_col:
                if functionality == "AI Classify":
                    input_type_ai_classify = st.selectbox(
                        "Input Type", ["Text", "Image"], key="ai_classify_input_type"
                    )
                elif functionality == "AI Filter":
                    input_type_ai_filter = st.selectbox(
                        "Input Type", ["Text", "Image"], key="ai_filter_input_type"
                    )
                elif functionality == "AI Agg":
                    input_type_ai_agg = st.selectbox(
                        "Input Type", ["Text", "Table"], key="ai_agg_input_type"
                    )
                elif functionality == "AI Summarize Agg":
                    input_type_ai_summarize_agg = st.selectbox(
                        "Input Type",
                        ["Text", "Table"],
                        key="ai_summarize_agg_input_type",
                    )

        if functionality != "Select Functionality":
            settings = get_functionality_settings(functionality, config, session)
            input_data = get_playground_input(functionality)

            # Check if audio input is needed for this functionality (only in debug mode)
            has_audio_input = False
            if config.get("mode") == "debug":
                if functionality == "Complete":
                    has_audio_input = True
                elif functionality == "Complete Multimodal":
                    has_audio_input = True
                elif functionality == "Translate":
                    has_audio_input = True
                elif functionality == "Extract":
                    has_audio_input = True
                elif functionality == "Sentiment":
                    # Check if entity sentiment toggle is active
                    if not input_data.get("toggle", False):
                        has_audio_input = True
                elif functionality == "Summarize":
                    has_audio_input = True
                elif functionality == "Classify Text":
                    has_audio_input = True
                elif functionality == "Parse Document":
                    has_audio_input = True
                elif functionality == "AI Similarity":
                    has_audio_input = True
                elif functionality == "AI Filter":
                    has_audio_input = True
                elif functionality == "AI Agg":
                    has_audio_input = True
                elif functionality == "AI Summarize Agg":
                    has_audio_input = True
                elif functionality == "AI Classify":
                    # Check input type for AI Classify - enable for all input types
                    has_audio_input = True
                elif functionality == "AI Complete":
                    # AI Complete has audio input and enhancement, but not for Prompt Object
                    if settings and settings.get("input_type") in ["Text", "Image"]:
                        has_audio_input = True
                elif functionality == "AI Translate":
                    has_audio_input = True
                elif functionality == "AI Transcribe":
                    # AI Transcribe processes audio files, doesn't need text input
                    has_audio_input = False
                elif functionality == "AI Parse Document":
                    # AI Parse Document processes document files, doesn't need text input
                    has_audio_input = False

            has_enhancement = False
            if functionality == "Complete":
                has_enhancement = True
            elif functionality == "Complete Multimodal":
                has_enhancement = True
            elif functionality == "Translate":
                has_enhancement = True
            elif functionality == "Extract":
                has_enhancement = True
            elif functionality == "Sentiment":
                # Check if entity sentiment toggle is active
                if not input_data.get("toggle", False):
                    has_enhancement = True
            elif functionality == "Summarize":
                has_enhancement = True
            elif functionality == "Classify Text":
                has_enhancement = True
            elif functionality == "Parse Document":
                has_enhancement = True
            elif functionality == "AI Similarity":
                has_enhancement = True
            elif functionality == "AI Filter":
                has_enhancement = True
            elif functionality == "AI Agg":
                # AI Agg has enhancement capabilities for task description
                has_enhancement = True
            elif functionality == "AI Summarize Agg":
                has_enhancement = True
            elif functionality == "AI Classify":
                has_enhancement = True
            elif functionality == "AI Complete":
                # AI Complete has enhancement capabilities for Text and Image input types
                if settings and settings.get("input_type") in ["Text", "Image"]:
                    has_enhancement = True
            elif functionality == "AI Translate":
                has_enhancement = True

            if has_audio_input and has_enhancement:
                audio_col, enhance_col, run_col = st.columns([1.5, 4, 30])

                with audio_col:
                    if functionality == "Complete":
                        create_audio_input_for_run(
                            f"speech_to_text_{functionality}",
                            f"prompt_text_{functionality}_data",
                        )
                    elif functionality == "Complete Multimodal":
                        create_audio_input_for_run(
                            f"speech_to_text_{functionality}",
                            f"prompt_text_{functionality}_data",
                        )
                    elif functionality == "Translate":
                        create_audio_input_for_run(
                            f"speech_to_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "Extract":
                        create_audio_input_for_run(
                            f"speech_to_text_query_{functionality}",
                            f"query_{functionality}_data",
                        )
                    elif functionality == "Sentiment":
                        create_audio_input_for_run(
                            f"speech_to_text_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "Summarize":
                        create_audio_input_for_run(
                            f"speech_to_text_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "Classify Text":
                        create_audio_input_for_run(
                            f"speech_to_text_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "Parse Document":
                        create_audio_input_for_run(
                            f"speech_to_text_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "AI Similarity":
                        create_audio_input_for_run(
                            f"speech_to_text_text1_{functionality}",
                            f"text1_{functionality}_data",
                        )
                    elif functionality == "AI Filter":
                        create_audio_input_for_run(
                            f"speech_to_text_predicate_{functionality}",
                            f"predicate_{functionality}_data",
                        )
                    elif functionality == "AI Agg":
                        create_audio_input_for_run(
                            f"speech_to_text_task_description_{functionality}",
                            f"task_description_{functionality}_data",
                        )
                    elif functionality == "AI Summarize Agg":
                        create_audio_input_for_run(
                            f"speech_to_text_task_description_{functionality}",
                            f"task_description_{functionality}_data",
                        )
                    elif functionality == "AI Classify":
                        # Enable for all input types
                        if settings and settings.get("input_type") == "Text":
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                        elif settings and settings.get("input_type") == "Image":
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                        else:
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                    elif functionality == "AI Complete":
                        # Determine the appropriate key based on input type
                        if settings and settings.get("input_type") == "Text":
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                        elif settings and settings.get("input_type") == "Image":
                            create_audio_input_for_run(
                                f"speech_to_text_predicate_{functionality}",
                                f"predicate_{functionality}_data",
                            )

                    elif functionality == "AI Translate":
                        create_audio_input_for_run(
                            f"speech_to_text_{functionality}",
                            f"text_{functionality}_data",
                        )

                with enhance_col:
                    # Helper function to get the appropriate prompt field key
                    def get_prompt_field_key(functionality, settings):
                        if functionality == "Complete":
                            return f"prompt_text_{functionality}_data"
                        elif functionality == "Complete Multimodal":
                            return f"prompt_text_{functionality}_data"
                        elif functionality == "Translate":
                            return f"text_{functionality}_data"
                        elif functionality == "Extract":
                            return f"query_{functionality}_data"
                        elif functionality == "Sentiment":
                            return f"text_{functionality}_data"
                        elif functionality == "Summarize":
                            return f"text_{functionality}_data"
                        elif functionality == "Classify Text":
                            return f"text_{functionality}_data"
                        elif functionality == "Parse Document":
                            return f"text_{functionality}_data"
                        elif functionality == "AI Similarity":
                            return f"text1_{functionality}_data"
                        elif functionality == "AI Filter":
                            return f"predicate_{functionality}_data"
                        elif functionality == "AI Agg":
                            return f"task_description_{functionality}_data"
                        elif functionality == "AI Summarize Agg":
                            return f"task_description_{functionality}_data"
                        elif functionality == "AI Classify":
                            return f"prompt_{functionality}_data"
                        elif functionality == "AI Complete":
                            if settings and settings.get("input_type") == "Text":
                                key = f"prompt_{functionality}_data"
                                return key
                            elif settings and settings.get("input_type") == "Image":
                                key = f"predicate_{functionality}_data"
                                return key
                            else:
                                return None
                        elif functionality == "AI Translate":
                            return f"text_{functionality}_data"
                        return None

                    # Function to handle enhancement when selectbox changes
                    def handle_enhancement_change():
                        print("Entered into enhancement!")
                        enhancement_key = f"enhancement_type_{functionality}"
                        if enhancement_key in st.session_state:
                            print("Enhancement key found in session state.")
                            selected_enhancement = st.session_state[enhancement_key]
                            if selected_enhancement.lower() != "refine":
                                prompt_key = get_prompt_field_key(
                                    functionality, settings
                                )
                                print(
                                    f"DEBUG1: handle_enhancement_change called with selected_enhancement='{selected_enhancement}', prompt_key='{prompt_key}'"
                                )
                                if prompt_key and prompt_key in st.session_state:
                                    current_prompt = st.session_state[prompt_key]
                                    print(f"DEBUG2: current_prompt='{current_prompt}'")
                                    if current_prompt and current_prompt.strip():
                                        print("DEBUG3: Current prompt is valid.")
                                        try:
                                            # Get model from settings if available
                                            model = None
                                            if settings and "model" in settings:
                                                model = settings["model"]

                                            print(
                                                f"DEBUG4: Calling enhance_prompt with model='{model}'"
                                            )
                                            enhanced_prompt = enhance_prompt(
                                                session,
                                                current_prompt,
                                                selected_enhancement.lower(),
                                                model,
                                            )
                                            print("enhaned_prompt", enhanced_prompt)
                                            if (
                                                enhanced_prompt
                                                and enhanced_prompt != current_prompt
                                            ):
                                                st.session_state[prompt_key] = (
                                                    enhanced_prompt
                                                )
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"Error enhancing prompt: {e}")

                    # Enhancement type selector with auto-trigger
                    enhancement_options = [
                        "Refine",
                        "Elaborate",
                        "Rephrase",
                        "Shorten",
                        "Formal",
                        "Informal",
                    ]
                    selected_enhancement = st.selectbox(
                        "some",
                        enhancement_options,
                        index=0,  # Default to "Refine"
                        key=f"enhancement_type_{functionality}",
                        on_change=handle_enhancement_change,
                        label_visibility="collapsed",
                    )

                with run_col:
                    if st.button("Run"):
                        st.session_state["execute_functionality"] = True
                        st.session_state["execution_params"] = {
                            "session": session,
                            "functionality": functionality,
                            "input_data": input_data,
                            "settings": settings,
                        }

            elif has_audio_input and not has_enhancement:
                audio_col, run_col = st.columns([1.5, 30])

                with audio_col:
                    if functionality == "Summarize":
                        create_audio_input_for_run(
                            f"speech_to_text_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "Classify Text":
                        create_audio_input_for_run(
                            f"speech_to_text_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "Parse Document":
                        create_audio_input_for_run(
                            f"speech_to_text_text_{functionality}",
                            f"text_{functionality}_data",
                        )
                    elif functionality == "AI Similarity":
                        create_audio_input_for_run(
                            f"speech_to_text_text1_{functionality}",
                            f"text1_{functionality}_data",
                        )
                    elif functionality == "AI Filter":
                        create_audio_input_for_run(
                            f"speech_to_text_predicate_{functionality}",
                            f"predicate_{functionality}_data",
                        )
                    elif functionality == "AI Summarize Agg":
                        create_audio_input_for_run(
                            f"speech_to_text_task_description_{functionality}",
                            f"task_description_{functionality}_data",
                        )
                    elif functionality == "AI Classify":
                        # Enable for all input types
                        if settings and settings.get("input_type") == "Text":
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                        elif settings and settings.get("input_type") == "Image":
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                        else:
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                    elif functionality == "AI Complete":
                        # Determine the appropriate key based on input type
                        if settings and settings.get("input_type") == "Text":
                            create_audio_input_for_run(
                                f"speech_to_text_prompt_{functionality}",
                                f"prompt_{functionality}_data",
                            )
                        elif settings and settings.get("input_type") == "Image":
                            create_audio_input_for_run(
                                f"speech_to_text_predicate_{functionality}",
                                f"predicate_{functionality}_data",
                            )

                with run_col:
                    if st.button("Run"):
                        st.session_state["execute_functionality"] = True
                        st.session_state["execution_params"] = {
                            "session": session,
                            "functionality": functionality,
                            "input_data": input_data,
                            "settings": settings,
                        }
            elif not has_audio_input and has_enhancement:
                enhance_col, run_col = st.columns([4, 30])

                with enhance_col:
                    # Helper function to get the appropriate prompt field key
                    def get_prompt_field_key(functionality, settings):
                        print(
                            f"DEBUG2: get_prompt_field_key called with functionality='{functionality}', settings={settings}"
                        )
                        if functionality == "Complete":
                            return f"prompt_text_{functionality}_data"
                        elif functionality == "Complete Multimodal":
                            return f"prompt_text_{functionality}_data"
                        elif functionality == "Translate":
                            return f"text_{functionality}_data"
                        elif functionality == "Extract":
                            return f"query_{functionality}_data"
                        elif functionality == "Sentiment":
                            return f"text_{functionality}_data"
                        elif functionality == "AI Complete":
                            if settings and settings.get("input_type") == "Text":
                                key = f"prompt_{functionality}_data"
                                return key
                            elif settings and settings.get("input_type") == "Image":
                                key = f"predicate_{functionality}_data"
                                return key
                            else:
                                return None
                        elif functionality == "AI Agg":
                            return f"task_description_{functionality}_data"
                        elif functionality == "AI Translate":
                            return f"text_{functionality}_data"
                        return None

                    # Function to handle enhancement when selectbox changes
                    def handle_enhancement_change():
                        enhancement_key = f"enhancement_type_{functionality}"
                        if enhancement_key in st.session_state:
                            selected_enhancement = st.session_state[enhancement_key]
                            if selected_enhancement.lower() != "refine":
                                prompt_key = get_prompt_field_key(
                                    functionality, settings
                                )
                                if prompt_key and prompt_key in st.session_state:
                                    current_prompt = st.session_state[prompt_key]
                                    if current_prompt and current_prompt.strip():
                                        try:
                                            # Get model from settings if available
                                            model = None
                                            if settings and "model" in settings:
                                                model = settings["model"]

                                            enhanced_prompt = enhance_prompt(
                                                session,
                                                current_prompt,
                                                selected_enhancement.lower(),
                                                model,
                                            )
                                            print("enhaned_prompt", enhanced_prompt)
                                            if (
                                                enhanced_prompt
                                                and enhanced_prompt != current_prompt
                                            ):
                                                st.session_state[prompt_key] = (
                                                    enhanced_prompt
                                                )
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"Error enhancing prompt: {e}")

                    # Enhancement type selector with auto-trigger
                    enhancement_options = [
                        "Refine",
                        "Elaborate",
                        "Rephrase",
                        "Shorten",
                        "Formal",
                        "Informal",
                    ]
                    selected_enhancement = st.selectbox(
                        "Enhancement",
                        enhancement_options,
                        index=0,  # Default to "Refine"
                        key=f"enhancement_type_{functionality}",
                        on_change=handle_enhancement_change,
                        label_visibility="collapsed",
                    )

                with run_col:
                    if st.button("Run"):
                        st.session_state["execute_functionality"] = True
                        st.session_state["execution_params"] = {
                            "session": session,
                            "functionality": functionality,
                            "input_data": input_data,
                            "settings": settings,
                        }
            else:
                # No audio input or enhancement needed, just show the Run button without extra spacing
                if st.button("Run"):
                    st.session_state["execute_functionality"] = True
                    st.session_state["execution_params"] = {
                        "session": session,
                        "functionality": functionality,
                        "input_data": input_data,
                        "settings": settings,
                    }

            # Execute functionality outside column structure for full width display
            if st.session_state.get("execute_functionality", False):
                st.session_state["execute_functionality"] = False
                params = st.session_state.get("execution_params", {})
                try:
                    execute_functionality(
                        params["session"],
                        params["functionality"],
                        params["input_data"],
                        params["settings"],
                    )
                except SnowparkSQLException as e:
                    st.error(f"Error: {e}")

    elif choices == "Chat":
        with initial_col2:
            options = st.selectbox(
                "Choose one of the options", ["Search Service", "RAG", "Cortex Agent"]
            )
        display_chat(session, options)
