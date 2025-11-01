import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
from pathlib import Path
import json
import time
import base64
import snowflake.connector
import os
from src.cortex_functions import *
from typing import List, Dict, Union, Optional
# from streamlit_mic_recorder import speech_to_text

# Load the config file
config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)


def get_snowflake_config():
    """
    Get Snowflake connection configuration from settings_config.json

    Returns:
        dict: Snowflake connection parameters
    """
    return config.get("snowflake_connection", {})


def determine_environment_and_file_source(session):
    """
    Determines whether we're running locally or in deployed Snowflake environment
    and returns appropriate file source configuration.

    Returns:
        tuple: (environment_type, source_path)
        - environment_type: "deployed", "local", or "unknown"
        - source_path: path to use for file operations
    """
    print("🔍 Detecting environment and file source...")

    # Method 1: Check if deployment stage exists and has files
    try:
        deployment_stage = (
            "@SNOWFLAKE_AI_TOOLKIT.PUBLIC.sf_ai_stage/snowflake_ai_toolkit"
        )
        files = session.sql(f"LIST {deployment_stage}/data/").collect()
        if len(files) > 0:
            print(f"✓ Detected deployed environment - using stage: {deployment_stage}")
            return "deployed", deployment_stage
    except Exception as e:
        print(f"🔍 Deployment stage check failed: {e}")

    # Method 2: Check if local data folder exists in current directory
    import os

    if os.path.exists("data") and os.path.isdir("data"):
        print("✓ Detected local environment - using current directory data folder")
        return "local", "data"

    # Method 3: Try to find data folder relative to script location
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_data_path = os.path.join(script_dir, "data")
        if os.path.exists(local_data_path):
            print(
                f"✓ Detected local environment - using script-relative data folder: {local_data_path}"
            )
            return "local", local_data_path
    except Exception as e:
        print(f"🔍 Script-relative path check failed: {e}")

    # Method 4: Debug current environment
    try:
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(
            f"🔍 Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}"
        )
    except Exception as e:
        print(f"🔍 Directory listing failed: {e}")

    print("⚠️ Could not determine environment - no suitable file source found")
    return "unknown", None


def upload_files_from_local(session, source_base_path, file_mappings):
    """
    Uploads files from local filesystem using PUT commands.

    Args:
        session: Snowflake session object
        source_base_path: Base path for local files
        file_mappings: List of file mapping configurations

    Returns:
        tuple: (files_uploaded, files_skipped)
    """
    import os

    files_uploaded = 0
    files_skipped = 0

    for mapping in file_mappings:
        if "source_folder" in mapping:
            # Handle folder-based mappings (for images, contracts, etc.)
            if source_base_path == "data":
                source_folder = mapping["source_folder"]
            else:
                # Extract relative path from mapping and combine with base path
                relative_path = (
                    mapping["source_folder"].replace("data/", "").replace("data\\", "")
                )
                source_folder = os.path.join(source_base_path, relative_path)

            stage_path = mapping["stage_path"]
            file_extensions = mapping["file_extensions"]
            description = mapping["description"]
            stage_name = mapping["stage_name"]

            print(f"🔍 Processing {description} from {source_folder}")

            if os.path.exists(source_folder):
                # Get existing files in stage
                existing_filenames = get_existing_stage_files(session, stage_path)

                # Process all files in the source folder
                for filename in os.listdir(source_folder):
                    file_path = os.path.join(source_folder, filename)

                    if os.path.isfile(file_path):
                        file_ext = os.path.splitext(filename)[1].lower()

                        if file_ext in file_extensions:
                            if filename not in existing_filenames:
                                try:
                                    # Upload file using PUT command
                                    normalized_path = file_path.replace("\\", "/")
                                    put_query = f"PUT 'file://{normalized_path}' {stage_path} AUTO_COMPRESS=FALSE"
                                    session.sql(put_query).collect()

                                    files_uploaded += 1
                                    print(
                                        f"✓ Uploaded {filename} to {stage_name} stage ({description})"
                                    )

                                except Exception as e:
                                    print(
                                        f"✗ Failed to upload {filename} to {stage_name}: {e}"
                                    )
                                    continue
                            else:
                                files_skipped += 1
                                print(
                                    f"⏭ File {filename} already exists in {stage_name}, skipping"
                                )
            else:
                print(f"❌ Source folder {source_folder} does not exist")

        elif "source_file" in mapping:
            # Handle individual file mappings (for CSV files, etc.)
            if source_base_path == "data":
                source_file = mapping["source_file"]
            else:
                # Extract relative path and combine with base path
                relative_path = (
                    mapping["source_file"].replace("data/", "").replace("data\\", "")
                )
                source_file = os.path.join(source_base_path, relative_path)

            description = mapping["description"]
            filename = os.path.basename(source_file)

            if os.path.exists(source_file):
                print(f"✓ Found local file: {source_file}")
                files_uploaded += 1
            else:
                print(f"❌ Local file not found: {source_file}")

    return files_uploaded, files_skipped


def copy_files_from_stage(session, deployment_stage, file_mappings):
    """
    Copies files from deployment stage using COPY FILES commands.

    Args:
        session: Snowflake session object
        deployment_stage: Path to deployment stage
        file_mappings: List of file mapping configurations

    Returns:
        tuple: (files_copied, files_skipped)
    """
    import os

    files_copied = 0
    files_skipped = 0

    for mapping in file_mappings:
        if "source_folder" in mapping:
            # Handle folder-based mappings (for images, contracts, etc.)
            relative_path = (
                mapping["source_folder"].replace("data/", "").replace("data\\", "")
            )
            source_path = f"{deployment_stage}/data/{relative_path}/"

            stage_path = mapping["stage_path"]
            file_extensions = mapping["file_extensions"]
            description = mapping["description"]
            stage_name = mapping["stage_name"]

            print(f"🔍 Processing {description} from {source_path}")

            # Get existing files in target stage
            existing_filenames = get_existing_stage_files(session, stage_path)

            # Get files from source deployment stage
            try:
                source_files = session.sql(f"LIST {source_path}").collect()
                print(
                    f"Found {len(source_files)} files in deployment stage at {source_path}"
                )

                for file_info in source_files:
                    file_name = file_info["name"]
                    filename_only = file_name.split("/")[-1].split("\\")[-1]
                    file_ext = os.path.splitext(filename_only)[1].lower()

                    if file_ext in file_extensions:
                        if filename_only not in existing_filenames:
                            try:
                                # Copy file from deployment stage to target stage
                                copy_query = f"""
                                COPY FILES INTO {stage_path}
                                FROM {source_path}{filename_only}
                                """
                                session.sql(copy_query).collect()

                                files_copied += 1
                                print(
                                    f"✓ Copied {filename_only} to {stage_name} stage ({description})"
                                )

                            except Exception as e:
                                print(
                                    f"✗ Failed to copy {filename_only} to {stage_name}: {e}"
                                )
                                continue
                        else:
                            files_skipped += 1
                            print(
                                f"⏭ File {filename_only} already exists in {stage_name}, skipping"
                            )

            except Exception as e:
                print(
                    f"❌ Could not list files from deployment stage {source_path}: {e}"
                )

        elif "source_file" in mapping:
            # Handle individual file mappings (for CSV files, etc.)
            relative_path = mapping["source_file"].replace(f"{deployment_stage}/", "")
            source_file = f"{deployment_stage}/{relative_path}"
            description = mapping["description"]
            filename = source_file.split("/")[-1]

            print(f"🔍 Processing {description}: {filename}")
            files_copied += 1  # Assume success for individual files

    return files_copied, files_skipped


def get_existing_stage_files(session, stage_path):
    """
    Gets list of existing files in a stage.

    Args:
        session: Snowflake session object
        stage_path: Stage path to list

    Returns:
        list: List of filenames in the stage
    """
    try:
        existing_files = session.sql(f"LIST {stage_path}").collect()
        existing_filenames = []
        for file in existing_files:
            file_name = file["name"]
            # Handle both forward and backslash separators, get just the filename
            if "/" in file_name:
                filename_only = file_name.split("/")[-1]
            elif "\\" in file_name:
                filename_only = file_name.split("\\")[-1]
            else:
                filename_only = file_name
            existing_filenames.append(filename_only)
        return existing_filenames
    except Exception as e:
        print(f"Could not list files in stage {stage_path} (might be empty): {e}")
        return []


def render_image(filepath: str):
    """
    Renders an image in Streamlit from a filepath.

    Args:
        filepath (str): Path to the image file. Must have a valid file extension.
    """
    mime_type = filepath.split(".")[-1:][0].lower()
    with open(filepath, "rb") as f:
        content_bytes = f.read()
        content_b64encoded = base64.b64encode(content_bytes).decode()
        image_string = f"data:image/{mime_type};base64,{content_b64encoded}"
        image_html = f"""
            <div style="text-align: center;">
                <img src="{image_string}" alt="App Logo" style="width: 200px;">
            </div>
        """
        st.sidebar.markdown(image_html, unsafe_allow_html=True)


def list_cortex_services(session, database, schema):
    q = f"SHOW CORTEX SEARCH SERVICES IN {database}.{schema}"
    return [row["name"] for row in session.sql(q).collect()]


def fetch_cortex_service(session, service_name, database, schema):
    q = f"SHOW CORTEX SEARCH SERVICEs LIKE '{service_name}' IN {database}.{schema}"
    return session.sql(q).collect()


def cortex_search_data_scan(session, db, schema, service_name):
    service = f"{db}.{schema}.{service_name}"
    q = f"SELECT * FROM TABLE (CORTEX_SEARCH_DATA_SCAN (SERVICE_NAME => '{service}'));"
    return session.sql(q).collect()


def cortex_search_service_display(session, db, schema, service_name):
    service = f"{db}.{schema}.{service_name}"
    q = f"DESCRIBE CORTEX SEARCH SERVICE {service};"
    return session.sql(q).collect()


def list_databases(session):
    """
    Lists all databases in Snowflake.

    Args:
        session: Snowflake session object

    Returns:
        list: List of database names
    """
    return [row["name"] for row in session.sql("SHOW DATABASES").collect()]


def list_schemas(session, database: str):
    """
    Lists schemas in the specified database.

    Args:
        session: Snowflake session object
        database (str): Name of the database

    Returns:
        list: List of schema names
    """
    return [row["name"] for row in session.sql(f"SHOW SCHEMAS IN {database}").collect()]


def list_stages(session, database: str, schema: str):
    """
    Lists stages in the specified database and schema.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema

    Returns:
        list: List of stage names
    """
    stages = [
        stage["name"]
        for stage in session.sql(f"SHOW STAGES IN {database}.{schema}").collect()
    ]
    return stages


def list_files_in_stage(session, database: str, schema: str, stage: str):
    """
    Lists files in the specified stage.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema
        stage (str): Name of the stage

    Returns:
        list: List of file names in the stage
    """
    stage_path = f"@{database}.{schema}.{stage}"
    files = [file["name"] for file in session.sql(f"LIST {stage_path}").collect()]
    return files


def list_file_details_in_stage(session, database, schema, stage_name):
    """
    Lists detailed information about files in the specified stage.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema
        stage_name (str): Name of the stage

    Returns:
        list: List of dictionaries containing file details (name, size, last modified)
    """
    stage_path = f"@{database}.{schema}.{stage_name}"
    query = f"LIST {stage_path}"
    try:
        files = session.sql(query).collect()
        return [
            {
                "Filename": file["name"],
                "Size (Bytes)": file["size"],
                "Last Modified": file["last_modified"],
            }
            for file in files
        ]
    except Exception as e:
        st.error(f"Failed to list files in stage '{stage_name}': {e}")
        return []


def list_tables(session, database: str, schema: str):
    """
    Lists tables in the specified database and schema.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema

    Returns:
        list: List of table names
    """
    tables = [
        table["name"]
        for table in session.sql(f"SHOW TABLES IN {database}.{schema}").collect()
    ]
    return tables


def list_columns(session, database: str, schema: str, table: str):
    """
    Lists columns in the specified table.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema
        table (str): Name of the table

    Returns:
        list: List of column names
    """
    return [
        row["column_name"]
        for row in session.sql(f"SHOW COLUMNS IN {database}.{schema}.{table}").collect()
    ]


def show_spinner(message: str):
    """
    Displays a spinner with a custom message in Streamlit.

    Args:
        message (str): Message to display with the spinner

    Yields:
        None
    """
    with st.spinner(message):
        yield


def validate_table_columns(session, database, schema, table, required_columns):
    """
    Validates that a table has all required columns.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema
        table (str): Name of the table
        required_columns (list): List of required column names

    Returns:
        list: List of missing column names

    Raises:
        RuntimeError: If column validation fails
    """
    try:
        # Query to get the column names in the specified table
        query = f"SHOW COLUMNS IN {database}.{schema}.{table}"
        columns = session.sql(query).collect()

        # Extract existing column names from the query result
        existing_columns = [column["column_name"].upper() for column in columns]

        # Check for missing columns
        missing_columns = [
            col for col in required_columns if col.upper() not in existing_columns
        ]

        return missing_columns
    except Exception as e:
        raise RuntimeError(f"Failed to validate columns for table '{table}': {e}")


def create_prompt_for_rag(
    session,
    question: str,
    rag: bool,
    column: str,
    database: str,
    schema: str,
    table: str,
    embedding_type: str,
    embedding_model: str,
    chat_history: list,
):
    """
    Creates a prompt for Retrieval-Augmented Generation (RAG).

    Args:
        session: Snowflake session object
        question (str): User's question
        rag (bool): Whether to use RAG
        column (str): Column name containing embeddings
        database (str): Name of the database
        schema (str): Name of the schema
        table (str): Name of the table
        embedding_type (str): Type of embedding
        embedding_model (str): Name of the embedding model
        chat_history (list): List of chat messages
    Returns:
        str: Generated prompt
    """
    if rag and column:
        cmd = f"""
        WITH results AS (
            SELECT RELATIVE_PATH,
                VECTOR_COSINE_SIMILARITY({column},
                SNOWFLAKE.CORTEX.{embedding_type}('{embedding_model}', ?)) AS similarity,
                chunk
            FROM {database}.{schema}.{table}
            ORDER BY similarity DESC
            LIMIT 3
        )
        SELECT chunk, relative_path FROM results;
        """

        question_rewrite = session.sql(cmd, [question]).collect()

        # Include chat history in the prompt
        chat_history_str = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in chat_history
        )

        prompt = f"""
        You are an AI assistant using RAG. Use the past messages and retrieved context to provide relevant answers. Note: Need not mention what the answer is based on.

        <chat_history>
        {chat_history_str}
        </chat_history>

        <retrieved_context>
        {question_rewrite}
        </retrieved_context>

        <question>
        {question}
        </question>

        Answer:
        """
    else:
        if len(chat_history):
            chat_history_str = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in chat_history
            )
        else:
            chat_history_str = ""

        prompt = f"""
        You are an AI assistant. Use the past messages to understand context and provide relevant answers. Note: Need not mention what the answer is based on.

        <chat_history>
        {chat_history_str}
        </chat_history>

        <question>
        {question}
        </question>

        Answer:
        """
    return prompt


def get_cortex_complete_result(session, query: str):
    """
    Executes a Cortex complete query and returns the result.

    Args:
        session: Snowflake session object
        query (str): SQL query to execute

    Returns:
        str: Query result
    """
    return session.sql(query).collect()[0][0]


def enhance_prompt(
    session, original_prompt: str, enhancement_type: str = "refine", model: str = None
):
    """
    Enhances a user prompt using Snowflake's CORTEX.COMPLETE function with different enhancement types.

    Args:
        session: Snowflake session object
        original_prompt (str): The original prompt to enhance
        enhancement_type (str): Type of enhancement - "refine", "elaborate", "rephrase", "shorten", "formal", "informal"
        model (str): Model to use for enhancement (optional, defaults to config)

    Returns:
        str: Enhanced prompt
    """
    if not original_prompt or not original_prompt.strip():
        return original_prompt

    # Use default model from config if not provided
    if not model:
        model = config.get("default_settings", {}).get("model", ["mistral-large"])
        if isinstance(model, list):
            model = model[0]

    # Create enhancement instructions based on type
    enhancement_instructions = {
        "refine": f"""You are an expert prompt engineer. Refine the following prompt to make it more detailed, specific, and effective for getting better AI responses. 

Guidelines:
1. Keep the original intent and meaning intact
2. Add specific details and context where appropriate
3. Make the prompt clearer and more actionable
4. Add relevant constraints or formatting instructions if beneficial
5. Ensure the enhanced prompt is concise but comprehensive

Original prompt: "{original_prompt}"

Please provide only the refined prompt without any explanations or additional text:""",
        "formal": f"""You are an expert prompt engineer. Convert the following prompt to a formal tone, making it more professional, structured, and appropriate for business or academic contexts.

Guidelines:
1. Use formal language and professional vocabulary
2. Structure sentences with proper grammar and syntax
3. Remove casual expressions and colloquialisms
4. Maintain the original intent and requirements
5. Make it sound authoritative and well-structured

Original prompt: "{original_prompt}"

Please provide only the formalized prompt without any explanations or additional text:""",
        "informal": f"""You are an expert prompt engineer. Convert the following prompt to an informal, conversational tone while maintaining its effectiveness and clarity.

Guidelines:
1. Use casual, friendly language that feels natural
2. Make it sound conversational and approachable
3. Remove overly formal or stiff language
4. Keep the original intent and all requirements intact
5. Ensure it sounds engaging and easy to understand

Original prompt: "{original_prompt}"

Please provide only the informal prompt without any explanations or additional text:""",
        "elaborate": f"""You are an expert prompt engineer. Elaborate on the following prompt by adding more detail, context, and specific instructions to make it more comprehensive and effective.

Guidelines:
1. Expand on the original request with relevant details
2. Add context that would help produce better responses
3. Include specific formatting or output requirements
4. Provide examples or clarifications where helpful
5. Keep the core intent clear and focused

Original prompt: "{original_prompt}"

Please provide only the elaborated prompt without any explanations or additional text:""",
        "rephrase": f"""You are an expert prompt engineer. Rephrase the following prompt using different wording while maintaining the exact same meaning and intent.

Guidelines:
1. Use alternative vocabulary and sentence structures
2. Keep the same meaning and intent
3. Make it clearer and more effective
4. Ensure the rephrased version is engaging and well-structured

Original prompt: "{original_prompt}"

Please provide only the rephrased prompt without any explanations or additional text:""",
        "shorten": f"""You are an expert prompt engineer. Shorten the following prompt while preserving all essential information and maintaining effectiveness.

Guidelines:
1. Remove unnecessary words and redundancy
2. Keep all critical information and requirements
3. Maintain clarity and effectiveness
4. Ensure the shortened version is still complete and actionable

Original prompt: "{original_prompt}"

Please provide only the shortened prompt without any explanations or additional text:""",
    }

    # Get the appropriate enhancement instruction
    enhancement_instruction = enhancement_instructions.get(
        enhancement_type.lower(), enhancement_instructions["refine"]
    )

    try:
        # Execute CORTEX.COMPLETE query
        from src.notification import execute_cortex_with_logging, escape_sql_string

        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{escape_sql_string(enhancement_instruction)}'
        )
        """
        result, notification_id = execute_cortex_with_logging(
            session=session,
            cortex_query=query,
            function_name="COMPLETE",
            model_name=model,
            input_text=enhancement_instruction,
            details=f"Prompt enhancement - {enhancement_type}",
        )
        if result:
            # Clean the result to make it JSON-parse friendly
            cleaned_result = result.strip()

            # Remove common prefixes/suffixes that models might add
            if cleaned_result.startswith('"') and cleaned_result.endswith('"'):
                cleaned_result = cleaned_result[1:-1]

            # Handle escaped quotes and newlines properly
            cleaned_result = (
                cleaned_result.replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\\\", "\\")
            )

            return cleaned_result
        else:
            return original_prompt

    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        return original_prompt


def list_existing_models(session):
    """
    Lists existing models in Snowflake.

    Args:
        session: Snowflake session object

    Returns:
        list: List of model names
    """
    query = "SHOW MODELS"  # Hypothetical query to show models
    return [model["name"] for model in session.sql(query).collect()]


def list_fine_tuned_models(session):
    """
    Lists fine-tuned models in Snowflake.

    Args:
        session: Snowflake session object

    Returns:
        list: List of fine-tuned model names
    """
    query = "SHOW FINE_TUNED_MODELS"  # Hypothetical query to show fine-tuned models
    return [model["name"] for model in session.sql(query).collect()]


def get_table_preview(session, database, schema, table):
    """
    Fetches a preview of the top 5 rows from a table.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema
        table (str): Name of the table

    Returns:
        pandas.DataFrame: DataFrame containing preview data
    """
    query = f"SELECT * FROM {database}.{schema}.{table} LIMIT 5"
    df = session.sql(query).to_pandas()
    return df


def load_css(filepath):
    """
    Loads and applies custom CSS from a file.

    Args:
        filepath (str): Path to the CSS file
    """
    with open(filepath) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def format_result(result_json):
    """
    Formats the result from a Cortex query.

    Args:
        result_json (dict): JSON response from Cortex

    Returns:
        dict: Formatted result containing messages, model used, and usage statistics
    """
    messages = result_json.get("choices", [{}])[0].get("messages", "No messages found")
    model_used = result_json.get("model", "No model specified")
    usage = result_json.get("usage", {})
    return {"messages": messages, "model_used": model_used, "usage": usage}


def write_result_to_output_table(session, output_table, output_column, result):
    """
    Writes a result to the specified output table and column.

    Args:
        session: Snowflake session object
        output_table (str): Name of the output table
        output_column (str): Name of the output column
        result: Result to write
    """
    insert_query = f"INSERT INTO {output_table} ({output_column}) VALUES (?)"
    session.sql(insert_query, [result]).collect()


def check_setup_status(session: Session):
    """
    Checks if the demo environment is already fully set up.
    Returns True if setup is complete, False if setup is needed.

    Args:
        session (Session): Snowflake session object

    Returns:
        tuple: (is_setup_complete: bool, status_message: str)
    """
    try:
        # Get configuration
        database_name = config["demo_database"]
        demo_schema = config["demo_schema"]

        # Check key components
        checks = []

        # 1. Check if demo database exists
        database_check = session.sql(f"SHOW DATABASES LIKE '{database_name}'").collect()
        if not database_check:
            return False, "Demo database not found"

        # 2. Check if demo schema exists
        schema_check = session.sql(
            f"SHOW SCHEMAS LIKE '{demo_schema}' IN {database_name}"
        ).collect()
        if not schema_check:
            return False, "Demo schema not found"

        # 3. Check if key tables exist
        key_tables = ["SAMPLE_STORES", "MEDICAL_NOTES", "SALES_CONVERSATIONS"]
        for table in key_tables:
            table_check = session.sql(
                f"SHOW TABLES LIKE '{table}' IN {database_name}.{demo_schema}"
            ).collect()
            if not table_check:
                return False, f"Table {table} not found"

        # 4. Check if key stages exist
        key_stages = [
            "MY_IMAGES",
            "CONTRACTS",
            "REPAIR_MANUAL",
            "SAMPLE_FILES",
            "MY_AUDIOS",
        ]
        for stage in key_stages:
            stage_check = session.sql(
                f"SHOW STAGES LIKE '{stage}' IN {database_name}.{demo_schema}"
            ).collect()
            if not stage_check:
                return False, f"Stage {stage} not found"

        # 5. Quick data check - ensure tables have data
        try:
            sales_count = session.sql(
                f"SELECT COUNT(*) FROM {database_name}.{demo_schema}.SALES_CONVERSATIONS"
            ).collect()[0][0]
            medical_count = session.sql(
                f"SELECT COUNT(*) FROM {database_name}.{demo_schema}.MEDICAL_NOTES"
            ).collect()[0][0]

            if sales_count == 0 or medical_count == 0:
                return False, "Tables exist but missing data"

        except:
            return False, "Error checking table data"

        return True, f"Demo environment ready in {database_name}.{demo_schema}"

    except Exception as e:
        return False, f"Error checking setup status: {str(e)}"


@st.fragment
def download_transcript(text, audio_filename):
    st.download_button(
        label="📥 Download Transcription",
        data=text,
        file_name=f"{audio_filename}_transcription.txt",
        mime="text/plain",
        help="Download the full transcription as a text file",
    )


def run_setup_silently(session: Session, config: dict):
    """
    Runs all setup functions with minimal output.

    Args:
        session (Session): Snowflake session object
        config (dict): Configuration dictionary
    """
    import sys
    from io import StringIO

    # # Capture stdout to reduce verbose output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        create_database_and_stage_if_not_exists(session)
        create_demo_database_and_stage_if_not_exists(session)
        create_stages_tables_for_demo(session)
        snowflake_config = get_snowflake_config()
        setup_pdf_text_chunker_demo(
            session, snowflake_config.get("database"), snowflake_config.get("schema")
        )
        create_search_and_rag_for_demo(session)
        create_starter_sql(session)

        # Restore stdout and return captured output for debugging if needed
        sys.stdout = old_stdout
        return True, "Setup completed successfully"

    except Exception as e:
        # Restore stdout
        sys.stdout = old_stdout
        return False, f"Setup failed: {str(e)}"
    finally:
        # Ensure stdout is always restored
        sys.stdout = old_stdout


def create_database_and_stage_if_not_exists(session: Session):
    """
    Creates the CORTEX_TOOLKIT database and MY_STAGE stage if they do not already exist.

    Args:
        session (Session): Snowflake session object
    """
    # Fetch database and stage details from the config file
    snowflake_config = get_snowflake_config()
    database_name = snowflake_config.get("database")
    stage_name = config["stage"]

    # Check if the database exists, and create if it doesn't
    database_query = f"SHOW DATABASES LIKE '{database_name}'"
    existing_databases = session.sql(database_query).collect()

    if not existing_databases:
        session.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}").collect()
    else:
        pass
        # print(f"Database '{database_name}' already exists. Skipping creation.")

    # Check if the stage exists, and create if it doesn't
    stage_query = f"SHOW STAGES LIKE '{stage_name}'"
    existing_stages = session.sql(stage_query).collect()

    if not existing_stages:
        session.sql(
            f"CREATE STAGE IF NOT EXISTS {database_name}.PUBLIC.{stage_name}"
        ).collect()
    else:
        pass
        # print(f"Stage '{stage_name}' already exists in '{database_name}'. Skipping creation.")


def create_demo_database_and_stage_if_not_exists(session: Session):
    """
    Creates the demo database, schema, stage and sample tables if they do not already exist.
    Sample tables are populated from Snowflake's sample dataset.

    Args:
        session (Session): Snowflake session object
    """
    try:
        # Use demo configurations from config
        database_name = config["demo_database"]  # "demo"
        demo_schema = config["demo_schema"]  # "DEMO_SCHEMA"

        # Stage names from config
        # demo_stage = config["demo_stage"]  # "demo_stage"
        my_images_stage = config["my_images"]  # "MY_IMAGES"
        my_audios_stage = config["my_audios"]  # "MY_AUDIOS"
        contracts_stage = config["contracts"]  # "CONTRACTS"
        repair_manual_stage = config["repair_manual"]  # "REPAIR_MANUAL"

        # Sample table names from config
        store_view = config["store_view"]  # "SAMPLE_STORE"
        store_returns_view = config["store_returns_view"]  # "SAMPLE_STORE_RETURNS"
        store_sales_view = config["store_sales_view"]  # "SAMPLE_STORE_SALES"
        reason_view = config["reason_view"]  # "SAMPLE_REASON"

        # Source dataset information
        source_db = "SNOWFLAKE_SAMPLE_DATA"
        source_schema = "TPCDS_SF10TCL"

        # Quick check: if demo database, schema, and key stages already exist, skip most setup
        try:
            database_check = session.sql(
                f"SHOW DATABASES LIKE '{database_name}'"
            ).collect()
            if database_check:
                schema_check = session.sql(
                    f"SHOW SCHEMAS LIKE '{demo_schema}' IN {database_name}"
                ).collect()
                if schema_check:
                    # Check if at least one key table exists
                    table_check = session.sql(
                        f"SHOW TABLES LIKE '{store_view}' IN {database_name}.{demo_schema}"
                    ).collect()
                    # Check if ALL required stages exist (including the new MY_AUDIOS)
                    my_images_stage_check = session.sql(
                        f"SHOW STAGES LIKE '{my_images_stage}' IN {database_name}.{demo_schema}"
                    ).collect()
                    my_audios_stage_check = session.sql(
                        f"SHOW STAGES LIKE '{my_audios_stage}' IN {database_name}.{demo_schema}"
                    ).collect()
                    contracts_stage_check = session.sql(
                        f"SHOW STAGES LIKE '{contracts_stage}' IN {database_name}.{demo_schema}"
                    ).collect()
                    repair_manual_stage_check = session.sql(
                        f"SHOW STAGES LIKE '{repair_manual_stage}' IN {database_name}.{demo_schema}"
                    ).collect()

                    # Only skip setup if ALL stages exist
                    all_stages_exist = (
                        my_images_stage_check
                        and my_audios_stage_check
                        and contracts_stage_check
                        and repair_manual_stage_check
                    )

                    if table_check and all_stages_exist:
                        print(
                            f"✓ Demo environment already exists in {database_name}.{demo_schema}"
                        )
                        print(
                            "⏭ Skipping database and table creation, but checking for new files to upload..."
                        )

                        # Still check for new files to upload
                        file_upload_only = True
                    else:
                        file_upload_only = False
                else:
                    file_upload_only = False
            else:
                file_upload_only = False
        except:
            file_upload_only = False

        if not file_upload_only:
            print("🚀 Setting up demo environment...")

            # 1. Create demo database if it doesn't exist
            database_query = f"SHOW DATABASES LIKE '{database_name}'"
            existing_databases = session.sql(database_query).collect()

            if not existing_databases:
                session.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}").collect()
                print(f"Created database '{database_name}'")

            # 2. Create demo schema in demo database if it doesn't exist
            schema_query = f"SHOW SCHEMAS LIKE '{demo_schema}' IN {database_name}"
            existing_schemas = session.sql(schema_query).collect()

            if not existing_schemas:
                session.sql(
                    f"CREATE SCHEMA IF NOT EXISTS {database_name}.{demo_schema}"
                ).collect()
                print(f"Created schema '{demo_schema}' in {database_name}")

            # 3. Create demo stages in demo database if they don't exist
            stages_to_create = [
                {"name": my_images_stage, "description": "Demo images stage"},
                {"name": my_audios_stage, "description": "Demo audio files stage"},
                {"name": contracts_stage, "description": "Contracts PDF stage"},
                {
                    "name": repair_manual_stage,
                    "description": "Repair manuals PDF stage",
                },
            ]
            for stage_info in stages_to_create:
                current_stage_name = stage_info["name"]
                description = stage_info["description"]

                create_stage_query = f"""
                    CREATE OR REPLACE STAGE {database_name}.{demo_schema}.{current_stage_name}
                    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
                    DIRECTORY = (ENABLE = TRUE)
                    COMMENT = 'Internal stage with server-side encryption and directory table enabled.';
                    """
                session.sql(create_stage_query).collect()
                print(
                    f"Created stage '{current_stage_name}' with server-side encryption ({description}) in {database_name}.{demo_schema}"
                )

            # 4. Create sample tables if they don't exist
            sample_tables = [
                {
                    "target_table": store_view,
                    "source_table": "STORE",
                    "limit": 1000,
                    "description": "Store information",
                },
                {
                    "target_table": store_returns_view,
                    "source_table": "STORE_RETURNS",
                    "limit": 1000,
                    "description": "Store returns data",
                },
                {
                    "target_table": store_sales_view,
                    "source_table": "STORE_SALES",
                    "limit": 1000,
                    "description": "Store sales data",
                },
                {
                    "target_table": reason_view,
                    "source_table": "REASON",
                    "limit": None,  # Copy all records
                    "description": "Reason codes",
                },
            ]

            for table_info in sample_tables:
                target_table = table_info["target_table"]
                source_table = table_info["source_table"]
                limit = table_info["limit"]
                description = table_info["description"]

                # Check if table exists in demo database
                table_query = f"SHOW TABLES LIKE '{target_table}' IN {database_name}.{demo_schema}"
                existing_tables = session.sql(table_query).collect()

                if not existing_tables:
                    # Create table with data from sample dataset in demo database
                    if limit:
                        create_query = f"""
                        CREATE TABLE {database_name}.{demo_schema}.{target_table} AS
                        SELECT * FROM {source_db}.{source_schema}.{source_table}
                        LIMIT {limit}
                        """
                    else:
                        create_query = f"""
                        CREATE TABLE {database_name}.{demo_schema}.{target_table} AS
                        SELECT * FROM {source_db}.{source_schema}.{source_table}
                        """

                    session.sql(create_query).collect()

                    # Get record count for confirmation
                    count_query = f"SELECT COUNT(*) as count FROM {database_name}.{demo_schema}.{target_table}"
                    record_count = session.sql(count_query).collect()[0]["COUNT"]

                    print(
                        f"Created table '{target_table}' with {record_count:,} records ({description})"
                    )
                else:
                    print(f"Table '{target_table}' already exists, skipping creation")

            print("Demo database and sample tables setup completed successfully!")

        # Ensure all stages exist (even in file_upload_only mode)
        # This handles cases where new stages were added after initial setup
        print("🔧 Ensuring all required stages exist...")
        stages_to_check = [
            {"name": my_images_stage, "description": "Demo images stage"},
            {"name": my_audios_stage, "description": "Demo audio files stage"},
            {"name": contracts_stage, "description": "Contracts PDF stage"},
            {"name": repair_manual_stage, "description": "Repair manuals PDF stage"},
        ]

        for stage_info in stages_to_check:
            stage_name = stage_info["name"]
            description = stage_info["description"]

            # Check if stage exists
            stage_exists_check = session.sql(
                f"SHOW STAGES LIKE '{stage_name}' IN {database_name}.{demo_schema}"
            ).collect()

            if not stage_exists_check:
                # Create the missing stage
                create_stage_query = f"""
                    CREATE OR REPLACE STAGE {database_name}.{demo_schema}.{stage_name}
                    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
                    DIRECTORY = (ENABLE = TRUE)
                    COMMENT = 'Internal stage with server-side encryption and directory table enabled.';
                    """
                session.sql(create_stage_query).collect()
                print(f"✅ Created missing stage '{stage_name}' ({description})")
            else:
                print(f"✓ Stage '{stage_name}' already exists")

        # 5. Handle file operations based on environment
        try:
            # Detect environment and determine file source
            environment, source_path = determine_environment_and_file_source(session)

            if environment == "unknown":
                print("⚠️ Could not determine environment - skipping file operations")
                if file_upload_only:
                    print("✅ File operation check completed (no files processed)!")
                else:
                    print(
                        "✅ Demo database, sample tables setup completed (no files processed)!"
                    )
                return

            # Define file mappings for both environments
            file_mappings = [
                {
                    "source_folder": "data/images",
                    "stage_name": my_images_stage,
                    "stage_path": f"@{database_name}.{demo_schema}.{my_images_stage}",
                    "file_extensions": [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".bmp",
                        ".webp",
                    ],
                    "description": "Images from data folder",
                },
                {
                    "source_folder": "data/samples/audio",
                    "stage_name": my_audios_stage,
                    "stage_path": f"@{database_name}.{demo_schema}.{my_audios_stage}",
                    "file_extensions": [".flac", ".mp3", ".ogg", ".wav", ".webm"],
                    "description": "Audio files from data/samples folder",
                },
                {
                    "source_folder": "data/rag/contracts",
                    "stage_name": contracts_stage,
                    "stage_path": f"@{database_name}.{demo_schema}.{contracts_stage}",
                    "file_extensions": [".pdf", ".doc", ".docx", ".txt"],
                    "description": "Contract documents",
                },
                {
                    "source_folder": "data/rag/repair_manuals",
                    "stage_name": repair_manual_stage,
                    "stage_path": f"@{database_name}.{demo_schema}.{repair_manual_stage}",
                    "file_extensions": [".pdf", ".doc", ".docx", ".txt"],
                    "description": "Repair manual documents",
                },
            ]

            # Process files based on environment
            if environment == "deployed":
                files_processed, files_skipped = copy_files_from_stage(
                    session, source_path, file_mappings
                )
                operation_type = "copied"
            elif environment == "local":
                files_processed, files_skipped = upload_files_from_local(
                    session, source_path, file_mappings
                )
                operation_type = "uploaded"

            # Summary
            if files_processed > 0:
                print(
                    f"📤 Successfully {operation_type} {files_processed} new files to demo stages"
                )
            if files_skipped > 0:
                print(f"⏭ Skipped {files_skipped} existing files in demo stages")
            if files_processed == 0 and files_skipped == 0:
                print(
                    f"📂 No files found to {operation_type.replace('ed', '')} to demo stages"
                )

        except Exception as e:
            print(f"Error processing files: {e}")
            # Don't fail the entire function if file operations fail

        if file_upload_only:
            print("✅ File upload check completed!")
        else:
            print(
                "✅ Demo database, sample tables, and data files setup completed successfully!"
            )

    except Exception as e:
        print(f"Error setting up demo database and tables: {e}")
        # Don't raise the exception to avoid breaking the application
        # but log it for debugging purposes


def get_default_value(functionality, field_name, sub_type="Text"):
    """
    Get default value from default_params with fallback to empty string
    """
    try:
        from src.defaults import default_params
    except ImportError:
        default_params = {}
    try:
        if functionality in default_params:
            func_defaults = default_params[functionality]

            # Handle nested structures (AI functions with Text/Image subtypes)
            if isinstance(func_defaults, dict) and sub_type in func_defaults:
                sub_defaults = func_defaults[sub_type]
                if field_name in sub_defaults:
                    return sub_defaults[field_name]

            # Handle direct field access
            if isinstance(func_defaults, dict) and field_name in func_defaults:
                return func_defaults[field_name]

        return ""  # Fallback to empty string
    except Exception:
        return ""  # Safe fallback


def create_stage(session, database, schema, stage_name):
    """
    Creates a stage in the specified database and schema.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema
        stage_name (str): Name of the stage to create

    Raises:
        SnowparkSQLException: If stage creation fails
    """
    query = f"CREATE STAGE IF NOT EXISTS {database}.{schema}.{stage_name}"
    try:
        session.sql(query).collect()
    except SnowparkSQLException as e:
        st.error(f"Failed to create stage: {e}")
        raise e


def upload_file_to_stage(session, database, schema, stage_name, file):
    """
    Uploads a file to the specified stage in Snowflake using the PUT command.

    Args:
        session: Snowflake session object
        database (str): Name of the database
        schema (str): Name of the schema
        stage_name (str): Name of the stage where the file will be uploaded
        file: File object from Streamlit file uploader

    Raises:
        Exception: If file upload fails
    """
    import tempfile
    import os

    # Construct stage path
    stage_path = f"@{database}.{schema}.{stage_name}"

    # Save the uploaded file temporarily
    temp_dir = tempfile.gettempdir()  # Use system temporary directory
    temp_file_path = os.path.join(temp_dir, file.name)
    temp_file_path = temp_file_path.replace(
        "\\", "/"
    )  # Ensure the path uses forward slashes for compatibility
    print("temp_file_path:", temp_file_path)

    try:
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())

        # Upload the file to the Snowflake stage
        put_query = f"PUT 'file://{temp_file_path}' {stage_path} AUTO_COMPRESS=FALSE"
        print("PUT Query:", put_query)  # For debugging
        session.sql(put_query).collect()

        st.success(f"File '{file.name}' uploaded successfully to stage '{stage_name}'.")
    except Exception as e:
        # Log the full traceback
        import traceback

        trace = traceback.format_exc()
        st.error(f"Failed to upload file: {e}")
        st.error(f"Traceback:\n{trace}")
        raise e
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


import streamlit as st
import time


def show_toast_message(message, duration=3, toast_type="info", position="top-right"):
    """
    Displays a toast message in Streamlit using a temporary container.

    Args:
        message (str): Message to display in the toast
        duration (int, optional): Duration in seconds to show the toast. Defaults to 3.
        toast_type (str, optional): Type of toast ("info", "success", "warning", "error"). Defaults to "info".
        position (str, optional): Position of the toast ("top-right", "top-left", "bottom-right", "bottom-left"). Defaults to "top-right".
    """
    # Define color styles based on the toast type
    toast_colors = {
        "info": "#007bff",
        "success": "#28a745",
        "warning": "#ffc107",
        "error": "#dc3545",
    }

    # Define position styles
    position_styles = {
        "top-right": "top: 20px; right: 20px;",
        "top-left": "top: 20px; left: 20px;",
        "bottom-right": "bottom: 20px; right: 20px;",
        "bottom-left": "bottom: 20px; left: 20px;",
    }

    color = toast_colors.get(toast_type, "#007bff")  # Default to "info" color
    pos_style = position_styles.get(
        position, "top: 20px; right: 20px;"
    )  # Default to "top-right"

    # Create a temporary container to display the toast
    toast_container = st.empty()

    # Use custom HTML and CSS to display a toast-like message
    toast_html = f"""
    <div style="
        position: fixed;
        {pos_style}
        background-color: {color};
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        font-family: Arial, sans-serif;
    ">
        {message}
    </div>
    """

    # Display the toast message
    toast_container.markdown(toast_html, unsafe_allow_html=True)

    # Wait for the specified duration, then clear the container
    time.sleep(duration)
    toast_container.empty()


def setup_pdf_text_chunker(session):
    """
    Sets up the pdf_text_chunker UDF in the current database and schema.

    Args:
        session: Snowflake session object

    Note:
        Creates a Python UDF that can process PDF files and split them into text chunks
    """
    # Check if UDF already exists
    try:
        udf_check_query = "SHOW USER FUNCTIONS LIKE 'pdf_text_chunker'"
        existing_udfs = session.sql(udf_check_query).collect()
        if existing_udfs:
            # st.info("UDF pdf_text_chunker already exists. Skipping creation.")
            return
    except Exception as e:
        st.error(f"Error checking UDF existence: {e}")
        return

    # Create UDF if it doesn't exist
    create_udf_query = """
    CREATE OR REPLACE FUNCTION pdf_text_chunker(file_url STRING)
    RETURNS TABLE (chunk VARCHAR)
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.9'
    HANDLER = 'pdf_text_chunker'
    PACKAGES = ('snowflake-snowpark-python', 'PyPDF2', 'langchain')
    AS
    $$
import PyPDF2
import io
import pandas as pd
from snowflake.snowpark.files import SnowflakeFile
from langchain.text_splitter import RecursiveCharacterTextSplitter

class pdf_text_chunker:
    def read_pdf(self, file_url: str) -> str:
        with SnowflakeFile.open(file_url, 'rb') as f:
            buffer = io.BytesIO(f.readall())
        reader = PyPDF2.PdfReader(buffer)
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text().replace('\\n', ' ').replace('\\0', ' ')
            except:
                text = "Unable to Extract"
        return text

    def process(self, file_url: str):
        text = self.read_pdf(file_url)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        df = pd.DataFrame(chunks, columns=['chunk'])
        yield from df.itertuples(index=False, name=None)
    $$
    """
    try:
        session.sql(create_udf_query).collect()
        # st.success("UDF pdf_text_chunker created successfully.")
    except Exception as e:
        st.error(f"Error creating UDF: {e}")


def setup_pdf_text_chunker_demo(session, db, schema):
    """
    Sets up the pdf_text_chunker UDF in the current database and schema.

    Args:
        session: Snowflake session object

    Note:
        Creates a Python UDF that can process PDF files and split them into text chunks
    """

    # Create UDF if it doesn't exist
    # TODO: Have to make schema dynamic
    create_udf_query = """
    CREATE OR REPLACE FUNCTION snowflake_ai_toolkit.public.pdf_text_chunker(file_url STRING)
    RETURNS TABLE (chunk VARCHAR)
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.9'
    HANDLER = 'pdf_text_chunker'
    PACKAGES = ('snowflake-snowpark-python', 'PyPDF2', 'langchain')
    AS
    $$
import PyPDF2
import io
import pandas as pd
from snowflake.snowpark.files import SnowflakeFile
from langchain.text_splitter import RecursiveCharacterTextSplitter

class pdf_text_chunker:
    def read_pdf(self, file_url: str) -> str:
        with SnowflakeFile.open(file_url, 'rb') as f:
            buffer = io.BytesIO(f.readall())
        reader = PyPDF2.PdfReader(buffer)
        text = ""
        for page in reader.pages:
            try:
                text += page.extract_text().replace('\\n', ' ').replace('\\0', ' ')
            except:
                text = "Unable to Extract"
        return text

    def process(self, file_url: str):
        text = self.read_pdf(file_url)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        df = pd.DataFrame(chunks, columns=['chunk'])
        yield from df.itertuples(index=False, name=None)
    $$
    """
    try:
        session.sql(create_udf_query).collect()
        # st.success("UDF pdf_text_chunker created successfully.")
    except Exception as e:
        print("UDF Already exists!")


def make_llm_call(session, system_prompt, prompt, model):
    prompt = prompt.replace("'", "''").replace("\n", "\\n").replace("\\", "\\\\")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    messages_json = escape_sql_string(json.dumps(messages))

    query = f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        '{model}',
        PARSE_JSON('{messages_json}')
    );
    """
    try:
        from src.notification import execute_cortex_with_logging

        result, notification_id = execute_cortex_with_logging(
            session=session,
            cortex_query=query,
            function_name="COMPLETE",
            model_name=model,
            input_text=prompt,
            details="LLM call with system and user prompts",
        )
        return result
    except SnowparkSQLException as e:
        raise e


def get_ai_complete_result(
    session,
    model,
    prompt,
    model_parameters=None,
    response_format=None,
    show_details=False,
):
    """
    Executes the AI_COMPLETE function for a text prompt.

    Args:
        session: Snowflake session object
        model: String specifying the language model
        prompt: Text prompt for completion
        model_parameters: Optional dict with temperature, top_p, max_tokens, guardrails
        response_format: Optional JSON schema for structured output
        show_details: Boolean to include detailed output (choices, created, model, usage)

    Returns:
        Completion result as a string or JSON object
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if not model:
        raise ValueError("Model must be specified.")

    prompt = prompt.replace("'", "''")

    model_parameters_json = None
    if model_parameters:
        if not isinstance(model_parameters, dict):
            raise ValueError("Model parameters must be a dictionary.")
        valid_params = {}
        allowed_keys = {"temperature", "top_p", "max_tokens", "guardrails"}
        for key in model_parameters:
            if key not in allowed_keys:
                print(f"Warning: Ignoring invalid model parameter '{key}'")
                continue
            valid_params[key] = model_parameters[key]
        if valid_params:
            model_parameters_json = json.dumps(
                valid_params, ensure_ascii=False
            ).replace("'", "''")

    response_format_json = None
    if response_format:
        if not isinstance(response_format, dict):
            raise ValueError("Response format must be a dictionary.")
        response_format_json = json.dumps(response_format, ensure_ascii=False).replace(
            "'", "''"
        )

    try:
        query = f"SELECT AI_COMPLETE('{model}', '{prompt}')"
        print(f"Generated SQL Query: {query}")

        result = session.sql(query).collect()[0][0]
        return result
    except SnowparkSQLException as e:
        print(f"SQL Error: {e}")
        raise


def get_ai_similarity_result(
    session,
    input1,
    input2,
    config_object=None,
    input_type="Text",
    stage=None,
    file1=None,
    file2=None,
):
    """
    Executes the AI_SIMILARITY function for text or image inputs.

    Args:
        session: Snowflake session object
        input1: First text input or file path for image
        input2: Second text input or file path for image
        config_object: Optional configuration object with 'model' key
        input_type: "Text" or "Image"
        stage: Stage path for image inputs
        file1: First image file name
        file2: Second image file name

    Returns:
        Float similarity score between -1 and 1
    """
    if input_type == "Text" and (not input1 or not input2):
        raise ValueError("Both text inputs must be non-empty.")
    if input_type == "Image" and (not stage or not file1 or not file2):
        raise ValueError("Stage and both image files must be provided.")

    config_json = None
    if config_object:
        if not isinstance(config_object, dict):
            raise ValueError("Config object must be a dictionary.")
        valid_config = {}
        allowed_keys = {"model"}
        for key in config_object:
            if key not in allowed_keys:
                print(f"Warning: Ignoring invalid config key '{key}'")
                continue
            valid_config[key] = config_object[key]
        if valid_config:
            config_json = json.dumps(valid_config, ensure_ascii=False).replace(
                "'", "''"
            )

    try:
        if input_type == "Text":
            input1 = input1.replace("'", "''")
            input2 = input2.replace("'", "''")
            query = f"SELECT AI_SIMILARITY('{input1}', '{input2}'"
        else:
            file_path1 = f"{stage}/{file1}"
            file_path2 = f"{stage}/{file2}"
            query = (
                f"SELECT AI_SIMILARITY(TO_FILE('{file_path1}'), TO_FILE('{file_path2}')"
            )

        if config_json:
            query += f", PARSE_JSON('{config_json}')"
        query += ")"

        print(f"Generated SQL Query: {query}")

        result = session.sql(query).collect()[0][0]
        return result
    except SnowparkSQLException as e:
        print(f"SQL Error: {e}")
        raise


def get_ai_classify_result(
    session,
    input_data: Union[str, Dict],
    categories: List[Union[str, Dict]],
    config_object: Optional[Dict] = None,
    input_type: str = "Text",
    stage: Optional[str] = None,
    file_name: Optional[str] = None,
) -> str:
    """
    Execute AI_CLASSIFY query for text or image classification.

    Args:
        session: Snowflake session object.
        input_data: Text input or dict with prompt for classification.
        categories: List of category dictionaries or strings.
        config_object: Optional configuration (task_description, output_mode, examples).
        input_type: 'Text' or 'Image'.
        stage: Stage path for image files (e.g., @db.schema.stage).
        file_name: File name for image input.

    Returns:
        Classification result as a JSON string or error message.
    """
    try:
        if input_type not in ["Text", "Image"]:
            return json.dumps(
                {"error": "Invalid input_type. Must be 'Text' or 'Image'."}
            )

        if not categories:
            return json.dumps({"error": "Categories list cannot be empty."})

        if isinstance(categories, list):
            if all(isinstance(cat, str) for cat in categories):
                categories_str = ", ".join(f"'{cat}'" for cat in categories)
            elif all(isinstance(cat, dict) for cat in categories):
                categories_str = ", ".join(
                    f"'{cat.get('name', '')}'" for cat in categories if cat.get("name")
                )
            else:
                return json.dumps(
                    {
                        "error": "Categories must be a list of strings or dictionaries with 'name' key."
                    }
                )
        else:
            return json.dumps({"error": "Categories must be a list."})

        if input_type == "Text":
            if isinstance(input_data, dict):
                input_text = input_data.get("prompt", "")
            elif isinstance(input_data, str):
                input_text = input_data
            else:
                return json.dumps(
                    {
                        "error": "input_data must be a string or dict with 'prompt' key for Text input_type."
                    }
                )

            if not input_text:
                return json.dumps(
                    {"error": "Input text cannot be empty for Text input_type."}
                )

            query = f"SELECT AI_CLASSIFY('{input_text}', ARRAY_CONSTRUCT({categories_str})) AS result"

        elif input_type == "Image":
            if not stage or not file_name:
                return json.dumps(
                    {"error": "Stage and file_name are required for Image input_type."}
                )
            if not isinstance(input_data, dict):
                return json.dumps(
                    {"error": "input_data must be a dict for Image input_type."}
                )

            query = f"SELECT AI_CLASSIFY('{stage}/{file_name}', ARRAY_CONSTRUCT({categories_str})) AS result"

        if config_object:
            task_description = config_object.get("task_description", "")
            output_mode = config_object.get("output_mode", "label")
            examples = config_object.get("examples", [])

            if task_description or examples or output_mode != "label":
                pass

        result = session.sql(query).collect()

        if not result:
            return json.dumps({"error": "No result returned from AI_CLASSIFY query."})

        classification_result = result[0]["RESULT"]

        if isinstance(classification_result, str):
            try:
                parsed_result = json.loads(classification_result)
                return json.dumps(parsed_result)
            except json.JSONDecodeError:
                return json.dumps({"classification": classification_result})
        else:
            return json.dumps(classification_result)

    except snowflake.connector.errors.ProgrammingError as e:
        return json.dumps({"error": f"Query execution failed: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})


def get_ai_filter_result(
    session,
    input_data: str,
    stage: Optional[str] = None,
    file_name: Optional[str] = None,
) -> str:
    """
    Execute AI_FILTER query for text or image filtering.

    Args:
        session: Snowflake session object.
        input_data: Text input or predicate for image filtering.
        stage: Stage path for image files (e.g., @db.schema.stage).
        file_name: File name for image input.

    Returns:
        Filter result as a JSON string or error message.
    """
    try:
        input_type = "Image" if stage and file_name else "Text"

        if input_type == "Text":
            if not input_data:
                return json.dumps(
                    {"error": "Input text cannot be empty for Text input_type."}
                )
            if not isinstance(input_data, str):
                return json.dumps(
                    {"error": "input_data must be a string for Text input_type."}
                )
            query = f"SELECT AI_FILTER('{input_data}') AS result"

        elif input_type == "Image":
            if not stage or not file_name:
                return json.dumps(
                    {"error": "Stage and file_name are required for Image input_type."}
                )
            if not isinstance(input_data, str):
                return json.dumps(
                    {"error": "Predicate must be a string for Image input_type."}
                )

            query = f"SELECT AI_FILTER('{input_data}', TO_FILE('{stage}/{file_name}')) AS result"

        result = session.sql(query).collect()

        if not result:
            return json.dumps({"error": "No result returned from AI_FILTER query."})

        filter_result = result[0]["RESULT"]

        if isinstance(filter_result, bool):
            return json.dumps({"result": filter_result})
        elif isinstance(filter_result, str):
            try:
                parsed_result = json.loads(filter_result)
                return json.dumps(parsed_result)
            except json.JSONDecodeError:
                return json.dumps({"result": filter_result})
        else:
            return json.dumps({"result": filter_result})

    except snowflake.connector.errors.ProgrammingError as e:
        return json.dumps({"error": f"Query execution failed: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})


def list_table_columns(session, database, schema, table):
    """
    Lists columns in a specified table.
    """
    try:
        query = f"DESCRIBE TABLE {database}.{schema}.{table}"
        result = session.sql(query).collect()
        return [row["name"] for row in result]
    except Exception as e:
        st.error(f"Error listing columns: {e}")
        return []


def create_stages_tables_for_demo(session):
    """
    Creates the SAMPLE_FILES stage and demo tables (MEDICAL_NOTES, TRAIN_SAMPLES, VALIDATE_SAMPLES)
    in the demo database if they do not already exist. Also uploads the corresponding CSV files.

    Args:
        session (Session): Snowflake session object
    """
    try:
        # Use demo configurations from config
        database_name = config["demo_database"]  # "SNOWFLAKE_AI_TOOLKIT"
        demo_schema = config["demo_schema"]  # "DEMO"
        sample_files_stage = "SAMPLE_FILES"

        print("🚀 Setting up demo stage and tables for sample files...")

        # 1. Create SAMPLE_FILES stage if it doesn't exist
        stage_query = (
            f"SHOW STAGES LIKE '{sample_files_stage}' IN {database_name}.{demo_schema}"
        )
        existing_stages = session.sql(stage_query).collect()

        if not existing_stages:
            create_stage_query = f"""
                CREATE OR REPLACE STAGE {database_name}.{demo_schema}.{sample_files_stage}
                ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
                DIRECTORY = (ENABLE = TRUE)
                COMMENT = 'Internal stage for sample CSV files with server-side encryption and directory table enabled.';
                """
            session.sql(create_stage_query).collect()
            print(
                f"Created stage '{sample_files_stage}' with server-side encryption in {database_name}.{demo_schema}"
            )
        else:
            print(f"Stage '{sample_files_stage}' already exists, skipping creation")

        # 2. Handle CSV file operations based on environment
        try:
            # Detect environment and determine file source
            environment, source_path = determine_environment_and_file_source(session)

            if environment == "unknown":
                print(
                    "⚠️ Could not determine environment - skipping CSV file operations"
                )
            else:
                # Define file mappings for CSV files
                file_mappings = [
                    {
                        "source_file": "data/search/health_prescription_data.csv",
                        "description": "Health prescription data for medical notes",
                    },
                    {
                        "source_file": "data/fine-tune/train.csv",
                        "description": "Training data for fine-tuning",
                    },
                    {
                        "source_file": "data/fine-tune/validate.csv",
                        "description": "Validation data for fine-tuning",
                    },
                    {
                        "source_file": "scripts/sales_metrics_model.yaml",
                        "description": "Sales metrics model configuration",
                    },
                ]

                stage_path = f"@{database_name}.{demo_schema}.{sample_files_stage}"

                # Get list of existing files in stage
                existing_filenames = get_existing_stage_files(session, stage_path)
                print(
                    f"Found {len(existing_filenames)} existing files in {sample_files_stage} stage: {existing_filenames}"
                )

                files_uploaded = 0
                files_skipped = 0

                # Process files based on environment
                if environment == "deployed":
                    # Copy files from deployment stage
                    for mapping in file_mappings:
                        filename = mapping["source_file"].split("/")[-1]
                        description = mapping["description"]

                        if filename not in existing_filenames:
                            try:
                                # Construct source path in deployment stage
                                if mapping["source_file"].startswith("scripts/"):
                                    source_file = (
                                        f"{source_path}/{mapping['source_file']}"
                                    )
                                else:
                                    source_file = (
                                        f"{source_path}/{mapping['source_file']}"
                                    )

                                # Copy file from deployment stage to SAMPLE_FILES stage
                                copy_query = f"""
                                COPY FILES INTO {stage_path}
                                FROM {source_file}
                                """
                                session.sql(copy_query).collect()

                                files_uploaded += 1
                                print(
                                    f"✓ Copied {filename} to {sample_files_stage} stage ({description})"
                                )

                            except Exception as e:
                                print(
                                    f"✗ Failed to copy {filename} to {sample_files_stage}: {e}"
                                )
                                continue
                        else:
                            files_skipped += 1
                            print(
                                f"⏭ File {filename} already exists in {sample_files_stage}, skipping"
                            )

                elif environment == "local":
                    # Upload files from local filesystem
                    import os

                    for mapping in file_mappings:
                        if source_path == "data":
                            source_file = mapping["source_file"]
                        else:
                            # Handle absolute path
                            if mapping["source_file"].startswith("scripts/"):
                                # For scripts, go up one level from data folder
                                script_base = os.path.dirname(source_path)
                                source_file = os.path.join(
                                    script_base, mapping["source_file"]
                                )
                            else:
                                relative_path = mapping["source_file"].replace(
                                    "data/", ""
                                )
                                source_file = os.path.join(source_path, relative_path)

                        filename = os.path.basename(source_file)
                        description = mapping["description"]

                        if os.path.exists(source_file):
                            if filename not in existing_filenames:
                                try:
                                    # Upload file using PUT command
                                    normalized_path = source_file.replace("\\", "/")
                                    put_query = f"PUT 'file://{normalized_path}' {stage_path} AUTO_COMPRESS=FALSE"
                                    session.sql(put_query).collect()

                                    files_uploaded += 1
                                    print(
                                        f"✓ Uploaded {filename} to {sample_files_stage} stage ({description})"
                                    )

                                except Exception as e:
                                    print(
                                        f"✗ Failed to upload {filename} to {sample_files_stage}: {e}"
                                    )
                                    continue
                            else:
                                files_skipped += 1
                                print(
                                    f"⏭ File {filename} already exists in {sample_files_stage}, skipping"
                                )
                        else:
                            print(
                                f"⚠ Source file {source_file} does not exist, skipping"
                            )

                # Summary
                operation_type = "copied" if environment == "deployed" else "uploaded"
                if files_uploaded > 0:
                    print(
                        f"📤 Successfully {operation_type} {files_uploaded} new files to {sample_files_stage} stage"
                    )
                if files_skipped > 0:
                    print(
                        f"⏭ Skipped {files_skipped} existing files in {sample_files_stage} stage"
                    )

        except Exception as e:
            print(f"Error processing CSV files: {e}")

        # 3. Create tables if they don't exist
        table_definitions = [
            {
                "table_name": "MEDICAL_NOTES",
                "source_file": "health_prescription_data.csv",
                "description": "Medical notes and prescription data",
                "create_query": f"""
                    CREATE TABLE IF NOT EXISTS {database_name}.{demo_schema}.MEDICAL_NOTES (
                        SUBJECT_ID INTEGER,
                        ROW_ID INTEGER,
                        HADM_ID INTEGER,
                        CATEGORY VARCHAR(500),
                        ADMISSION_TYPE VARCHAR(500),
                        DIAGNOSIS VARCHAR(1000),
                        TEXT VARCHAR(16777216)
                    )
                """,
                "copy_query": f"""
                    COPY INTO {database_name}.{demo_schema}.MEDICAL_NOTES
                    FROM {stage_path}/health_prescription_data.csv
                    FILE_FORMAT = (TYPE = 'CSV' FIELD_DELIMITER = ',' SKIP_HEADER = 1 
                                  FIELD_OPTIONALLY_ENCLOSED_BY = '"' ESCAPE_UNENCLOSED_FIELD = NONE)
                    ON_ERROR = 'CONTINUE'
                """,
            },
            {
                "table_name": "TRAIN_SAMPLES",
                "source_file": "train.csv",
                "description": "Training samples for fine-tuning",
                "create_query": f"""
                    CREATE TABLE IF NOT EXISTS {database_name}.{demo_schema}.TRAIN_SAMPLES (
                        COMPLETION VARCHAR(16777216),
                        PROMPT VARCHAR(16777216)
                    )
                """,
                "copy_query": f"""
                    COPY INTO {database_name}.{demo_schema}.TRAIN_SAMPLES
                    FROM {stage_path}/train.csv
                    FILE_FORMAT = (TYPE = 'CSV' FIELD_DELIMITER = ',' SKIP_HEADER = 1 
                                  FIELD_OPTIONALLY_ENCLOSED_BY = '"' ESCAPE_UNENCLOSED_FIELD = NONE)
                    ON_ERROR = 'CONTINUE'
                """,
            },
            {
                "table_name": "VALIDATE_SAMPLES",
                "source_file": "validate.csv",
                "description": "Validation samples for fine-tuning",
                "create_query": f"""
                    CREATE TABLE IF NOT EXISTS {database_name}.{demo_schema}.VALIDATE_SAMPLES (
                        COMPLETION VARCHAR(16777216),
                        PROMPT VARCHAR(16777216)
                    )
                """,
                "copy_query": f"""
                    COPY INTO {database_name}.{demo_schema}.VALIDATE_SAMPLES
                    FROM {stage_path}/validate.csv
                    FILE_FORMAT = (TYPE = 'CSV' FIELD_DELIMITER = ',' SKIP_HEADER = 1 
                                  FIELD_OPTIONALLY_ENCLOSED_BY = '"' ESCAPE_UNENCLOSED_FIELD = NONE)
                    ON_ERROR = 'CONTINUE'
                """,
            },
        ]

        for table_def in table_definitions:
            table_name = table_def["table_name"]
            source_file = table_def["source_file"]
            description = table_def["description"]
            create_query = table_def["create_query"]
            copy_query = table_def["copy_query"]

            # Check if table exists
            table_query = (
                f"SHOW TABLES LIKE '{table_name}' IN {database_name}.{demo_schema}"
            )
            existing_tables = session.sql(table_query).collect()

            if not existing_tables:
                try:
                    # Create table
                    session.sql(create_query).collect()
                    print(f"Created table '{table_name}' ({description})")

                    # Load data from CSV file if it was uploaded
                    if source_file.split("/")[-1] in [
                        f.split("/")[-1]
                        for f in [m["source_file"] for m in file_mappings]
                    ]:
                        session.sql(copy_query).collect()

                        # Get record count for confirmation
                        count_query = f"SELECT COUNT(*) as count FROM {database_name}.{demo_schema}.{table_name}"
                        record_count = session.sql(count_query).collect()[0]["COUNT"]

                        print(
                            f"Loaded {record_count:,} records into table '{table_name}' from {source_file}"
                        )
                    else:
                        print(
                            f"Source file {source_file} not found for table '{table_name}'"
                        )

                except Exception as e:
                    print(f"✗ Failed to create or load table '{table_name}': {e}")
                    continue
            else:
                # Table exists, check if it has data
                count_query = f"SELECT COUNT(*) as count FROM {database_name}.{demo_schema}.{table_name}"
                try:
                    record_count = session.sql(count_query).collect()[0]["COUNT"]
                    print(
                        f"Table '{table_name}' already exists with {record_count:,} records, skipping creation"
                    )
                except Exception as e:
                    print(
                        f"Table '{table_name}' exists but could not get record count: {e}"
                    )

        print("✅ Demo stage and tables setup completed successfully!")

    except Exception as e:
        print(f"Error setting up demo stage and tables: {e}")
        # Don't raise the exception to avoid breaking the application
        # but log it for debugging purposes


def create_search_and_rag_for_demo(session):
    """
    Creates Cortex Search Service on MEDICAL_NOTES table and triggers RAG processes
    for repair manuals and contracts stages in the demo environment.

    Args:
        session (Session): Snowflake session object
    """
    try:
        from src.cortex_functions import create_vector_embedding_from_stage
        from src.notification import add_notification_entry

        # Use demo configurations from config
        database_name = config["demo_database"]
        demo_schema = config["demo_schema"]
        snowflake_config = get_snowflake_config()
        warehouse = snowflake_config.get("warehouse", "COMPUTE_WH")
        contracts_stage = config["contracts"]
        repair_manual_stage = config["repair_manual"]

        # Default embedding settings
        embedding_type = "EMBED_TEXT_1024"
        embedding_model = "snowflake-arctic-embed-l-v2.0"

        print("🚀 Setting up search service and RAG processes for demo...")

        # 0. Setup PDF text chunker UDF first (required for vector embeddings)
        print("📝 Setting up PDF text chunker UDF...")
        setup_pdf_text_chunker_demo(session, database_name, demo_schema)

        # 1. Create Cortex Search Service on MEDICAL_NOTES table
        search_service_name = "MEDNOTES_SEARCH_SERVICE_Demo"

        # Check if search service already exists
        try:
            search_check_query = f"SHOW CORTEX SEARCH SERVICES LIKE '{search_service_name}' IN {database_name}.{demo_schema}"
            existing_services = session.sql(search_check_query).collect()

            if not existing_services:
                # Create the search service
                create_search_query = f"""
                CREATE OR REPLACE CORTEX SEARCH SERVICE {database_name}.{demo_schema}.{search_service_name}
                  ON TEXT
                  ATTRIBUTES ADMISSION_TYPE, DIAGNOSIS
                  WAREHOUSE = {warehouse}
                  TARGET_LAG = '30 day'
                  AS (
                    SELECT
                        SUBJECT_ID,
                        ROW_ID,
                        HADM_ID,
                        CATEGORY,
                        ADMISSION_TYPE,
                        DIAGNOSIS,
                        TEXT
                    FROM {database_name}.{demo_schema}.MEDICAL_NOTES
                );
                """

                session.sql(create_search_query).collect()
                print(
                    f"✓ Created Cortex Search Service '{search_service_name}' on MEDICAL_NOTES table"
                )
            else:
                print(
                    f"⏭ Cortex Search Service '{search_service_name}' already exists, skipping creation"
                )

        except Exception as e:
            print(f"✗ Failed to create search service '{search_service_name}': {e}")

        # 2. Setup RAG for Repair Manuals
        repair_manual_table = "REPAIR_MANUAL_EMBEDDINGS_Demo"
        try:
            # Check if table already exists
            table_check_query = f"SHOW TABLES LIKE '{repair_manual_table}' IN {database_name}.{demo_schema}"
            existing_tables = session.sql(table_check_query).collect()

            if not existing_tables:
                # Create notification entry for repair manual RAG process
                details = f"Creating vector embeddings from {repair_manual_stage} stage using {embedding_model}"
                notification_id = add_notification_entry(
                    session, "Create Embedding", "In-Progress", details
                )

                print(
                    f"🔄 Starting RAG process for repair manuals (Stage: {repair_manual_stage})"
                )
                print(f"   Output table: {repair_manual_table}")
                print(f"   Embedding model: {embedding_model}")

                # Refresh stage metadata and check if stage has files before processing
                print(f"🔄 Refreshing stage metadata for {repair_manual_stage}...")
                try:
                    # First, refresh the stage to update metadata
                    refresh_query = f"ALTER STAGE {database_name}.{demo_schema}.{repair_manual_stage} REFRESH"
                    session.sql(refresh_query).collect()
                    print(f"✓ Stage {repair_manual_stage} refreshed")
                except Exception as refresh_error:
                    print(
                        f"⚠️ Could not refresh stage {repair_manual_stage}: {refresh_error}"
                    )

                # Use LIST command instead of DIRECTORY() for more reliable file detection
                stage_files_query = (
                    f"LIST '@{database_name}.{demo_schema}.{repair_manual_stage}'"
                )
                try:
                    file_list = session.sql(stage_files_query).collect()
                    file_count = len(file_list)
                    print(f"   Files in stage: {file_count}")

                    if file_count > 0:
                        print("   Files found:")
                        for i, file_info in enumerate(file_list[:5]):
                            print(f"     - {file_info['name']}")
                        if file_count > 5:
                            print(f"     ... and {file_count - 5} more files")
                    else:
                        print(
                            f"⚠️ Warning: No files found in stage {repair_manual_stage}"
                        )
                        print(
                            "⚠️ This might be because files were just uploaded and stage cache needs time to update"
                        )
                        print("🔄 Proceeding with embedding creation anyway...")

                except Exception as e:
                    print(f"⚠️ Could not list files in stage {repair_manual_stage}: {e}")
                    print("🔄 Proceeding with embedding creation anyway...")

                # Create vector embeddings from stage
                create_vector_embedding_from_stage(
                    session=session,
                    db=database_name,
                    schema=demo_schema,
                    stage=repair_manual_stage,
                    embedding_type=embedding_type,
                    embedding_model=embedding_model,
                    output_table=repair_manual_table,
                    # notification_id=notification_id
                )

                # Verify embeddings were created
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {database_name}.{demo_schema}.{repair_manual_table}"
                    embedding_count = session.sql(count_query).collect()[0]["COUNT"]
                    print(
                        f"✓ RAG process completed for repair manuals: {embedding_count} embeddings created (Notification ID: {notification_id})"
                    )
                except Exception as count_error:
                    print(
                        f"⚠️ Could not verify embedding count for repair manuals: {count_error}"
                    )
                    print(
                        f"✓ RAG process initiated for repair manuals (Notification ID: {notification_id})"
                    )
            else:
                print(
                    f"⏭ Table '{repair_manual_table}' already exists, skipping RAG process for repair manuals"
                )

        except Exception as e:
            print(f"✗ Failed to setup RAG for repair manuals: {e}")

        # 3. Setup RAG for Contracts
        contracts_table = "CONTRACTS_EMBEDDINGS_Demo"
        try:
            # Check if table already exists
            table_check_query = (
                f"SHOW TABLES LIKE '{contracts_table}' IN {database_name}.{demo_schema}"
            )
            existing_tables = session.sql(table_check_query).collect()

            if not existing_tables:
                # Create notification entry for contracts RAG process
                details = f"Creating vector embeddings from {contracts_stage} stage using {embedding_model}"
                notification_id = add_notification_entry(
                    session, "Create Embedding", "In-Progress", details
                )

                print(
                    f"🔄 Starting RAG process for contracts (Stage: {contracts_stage})"
                )
                print(f"   Output table: {contracts_table}")
                print(f"   Embedding model: {embedding_model}")

                # Refresh stage metadata and check if stage has files before processing
                print(f"🔄 Refreshing stage metadata for {contracts_stage}...")
                try:
                    # First, refresh the stage to update metadata
                    refresh_query = f"ALTER STAGE {database_name}.{demo_schema}.{contracts_stage} REFRESH"
                    session.sql(refresh_query).collect()
                    print(f"✓ Stage {contracts_stage} refreshed")
                except Exception as refresh_error:
                    print(
                        f"⚠️ Could not refresh stage {contracts_stage}: {refresh_error}"
                    )

                # Use LIST command instead of DIRECTORY() for more reliable file detection
                stage_files_query = (
                    f"LIST '@{database_name}.{demo_schema}.{contracts_stage}'"
                )
                try:
                    file_list = session.sql(stage_files_query).collect()
                    file_count = len(file_list)
                    print(f"   Files in stage: {file_count}")

                    if file_count > 0:
                        print("   Files found:")
                        for i, file_info in enumerate(
                            file_list[:5]
                        ):  # Show first 5 files
                            print(f"     - {file_info['name']}")
                        if file_count > 5:
                            print(f"     ... and {file_count - 5} more files")
                    else:
                        print(f"⚠️ Warning: No files found in stage {contracts_stage}")
                        print(
                            "⚠️ This might be because files were just uploaded and stage cache needs time to update"
                        )
                        print("🔄 Proceeding with embedding creation anyway...")

                except Exception as e:
                    print(f"⚠️ Could not list files in stage {contracts_stage}: {e}")
                    print("🔄 Proceeding with embedding creation anyway...")

                create_vector_embedding_from_stage(
                    session=session,
                    db=database_name,
                    schema=demo_schema,
                    stage=contracts_stage,
                    embedding_type=embedding_type,
                    embedding_model=embedding_model,
                    output_table=contracts_table,
                    # notification_id=notification_id
                )

                # Verify embeddings were created
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {database_name}.{demo_schema}.{contracts_table}"
                    embedding_count = session.sql(count_query).collect()[0]["COUNT"]
                    print(
                        f"✓ RAG process completed for contracts: {embedding_count} embeddings created (Notification ID: {notification_id})"
                    )
                except Exception as count_error:
                    print(
                        f"⚠️ Could not verify embedding count for contracts: {count_error}"
                    )
                    print(
                        f"✓ RAG process initiated for contracts (Notification ID: {notification_id})"
                    )
            else:
                print(
                    f"⏭ Table '{contracts_table}' already exists, skipping RAG process for contracts"
                )

        except Exception as e:
            print(f"✗ Failed to setup RAG for contracts: {e}")

        print("✅ Search service and RAG processes setup completed!")
        print("📋 Summary:")
        print(f"   • Search Service: {search_service_name} (on MEDICAL_NOTES)")
        print(f"   • RAG Tables: {repair_manual_table}, {contracts_table}")
        print(f"   • Embedding Model: {embedding_model}")
        print(
            "   • Note: RAG processes run asynchronously - check notifications for completion status"
        )

    except Exception as e:
        print(f"Error setting up search and RAG for demo: {e}")
        # Don't raise the exception to avoid breaking the application
        # but log it for debugging purposes


def create_starter_sql(session):
    """
    Execute the agent.sql script to set up sales intelligence demo data and infrastructure.
    This creates sales_conversations and sales_metrics tables with sample data,
    and sets up a Cortex search service for sales conversation analysis.

    Args:
        session: Snowflake session object

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Use demo configurations from config
        database_name = config["demo_database"]  # "SNOWFLAKE_AI_TOOLKIT"
        demo_schema = config["demo_schema"]  # "DEMO"

        # Quick check: if key components already exist, skip setup
        try:
            # Check if both main tables exist and have data
            tables_check_query = f"""
                SELECT COUNT(*) as table_count 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = '{demo_schema}' 
                AND TABLE_CATALOG = '{database_name}'
                AND TABLE_NAME IN ('SALES_CONVERSATIONS', 'SALES_METRICS')
            """
            table_count = session.sql(tables_check_query).collect()[0]["TABLE_COUNT"]

            if table_count == 2:
                # Both tables exist, check if they have data
                sales_conv_count = session.sql(
                    f"SELECT COUNT(*) as count FROM {database_name}.{demo_schema}.SALES_CONVERSATIONS"
                ).collect()[0]["COUNT"]
                sales_metrics_count = session.sql(
                    f"SELECT COUNT(*) as count FROM {database_name}.{demo_schema}.SALES_METRICS"
                ).collect()[0]["COUNT"]

                if sales_conv_count > 0 and sales_metrics_count > 0:
                    print("✓ Sales intelligence demo already exists and has data")
                    print(f"   • SALES_CONVERSATIONS: {sales_conv_count} records")
                    print(f"   • SALES_METRICS: {sales_metrics_count} records")
                    print("⏭ Skipping setup - demo environment already configured")
                    return True, "Sales intelligence demo already exists"
        except Exception:
            # If any check fails, proceed with setup
            pass

        # Read the agent.sql file
        agent_sql_path = Path("scripts/agent.sql")
        if not agent_sql_path.exists():
            return False, "agent.sql file not found in scripts directory"

        with open(agent_sql_path, "r") as f:
            sql_content = f.read()

        # Split SQL content into individual statements
        # Remove comments and empty lines, then split by semicolon
        sql_statements = []

        # Remove all comment lines first
        clean_lines = []
        for line in sql_content.split("\n"):
            # Skip comment lines that start with --
            if line.strip().startswith("--"):
                continue
            # Skip empty lines
            if line.strip():
                clean_lines.append(line)

        # Join back and split by semicolon to get individual statements
        clean_sql = "\n".join(clean_lines)
        raw_statements = clean_sql.split(";")

        # Clean up each statement
        for statement in raw_statements:
            cleaned = statement.strip()
            if cleaned:  # Only add non-empty statements
                sql_statements.append(cleaned)

        print("🚀 Starting sales intelligence demo setup...")
        print(f"📝 Found {len(sql_statements)} SQL statements to execute")

        # Execute each SQL statement
        for i, statement in enumerate(sql_statements):
            if not statement or statement.strip() == ";":
                continue

            try:
                print(f"📋 Executing statement {i + 1}/{len(sql_statements)}")

                # Execute the statement
                session.sql(statement).collect()

                # Show progress for major operations
                if "CREATE TABLE" in statement.upper():
                    if "sales_conversations" in statement:
                        print("✓ Created sales_conversations table")
                    elif "sales_metrics" in statement:
                        print("✓ Created sales_metrics table")
                elif "INSERT INTO" in statement.upper():
                    if "sales_conversations" in statement:
                        print("✓ Inserted sample conversation data")
                    elif "sales_metrics" in statement:
                        print("✓ Inserted sample metrics data")
                elif "CREATE OR REPLACE WAREHOUSE" in statement.upper():
                    print("✓ Created sales_intelligence_wh warehouse")
                elif "CREATE OR REPLACE CORTEX SEARCH SERVICE" in statement.upper():
                    print("✓ Created sales_conversation_search_Demo service")
                elif (
                    "ALTER TABLE" in statement.upper()
                    and "CHANGE_TRACKING" in statement.upper()
                ):
                    print("✓ Enabled change tracking on sales_conversations")

            except Exception as e:
                error_msg = str(e)
                # Handle specific common errors gracefully
                if "already exists" in error_msg.lower():
                    if "warehouse" in statement.lower():
                        print("⏭ Warehouse already exists, skipping...")
                    elif "table" in statement.lower():
                        print("⏭ Table already exists, skipping...")
                    elif "search service" in statement.lower():
                        print("⏭ Search service already exists, skipping...")
                    continue
                else:
                    print(f"✗ Error executing statement {i + 1}: {error_msg}")
                    return False, f"Error executing SQL statement {i + 1}: {error_msg}"

        print("🎉 Sales intelligence demo setup completed successfully!")
        print("📊 You can now use the sales conversation data for AI-powered analysis")

        return True, "Sales intelligence demo setup completed successfully"

    except FileNotFoundError:
        error_msg = "agent.sql file not found in scripts directory"
        print(f"✗ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error setting up sales intelligence demo: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg
