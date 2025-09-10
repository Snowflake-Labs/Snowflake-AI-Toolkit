from snowflake.snowpark.exceptions import SnowparkSQLException
import json
import streamlit as st
from src.notification import execute_cortex_with_logging, escape_sql_string as notification_escape_sql_string


def escape_sql_string(s):
    """Helper function to escape single quotes in SQL strings.
    
    Args:
        s (str): The SQL string to escape.
        
    Returns:
        str: The escaped SQL string with single quotes doubled.
    """
    return s.replace("'", "''")


def check_and_create_table(session, db, schema, table, columns):
    """
    Check if the table exists and if it does, truncate it.
    If the table does not exist, create it with the specified columns.

    Args:
        session: Snowflake session.
        db: Database name.
        schema: Schema name.
        table: Table name.
        columns: List of column definitions (e.g., ["col1 STRING", "col2 NUMBER"]).
    """
    full_table_name = f"{db}.{schema}.{table}"

    # Check if the table exists
    table_exists_query = f"SHOW TABLES IN SCHEMA {db}.{schema}"
    tables = session.sql(table_exists_query).collect()

    table_exists = any(t["name"] == table.upper() for t in tables)

    if table_exists:
        truncate_query = f"DROP TABLE {full_table_name}"
        session.sql(truncate_query).collect()
        print(f"Table {full_table_name} dropped successfully.")     


    # Create the table if it doesn't exist
    columns_definition = ", ".join(columns)
    create_table_query = f"CREATE TABLE {full_table_name} ({columns_definition})"
    session.sql(create_table_query).collect()
    print(f"Table {full_table_name} created successfully.")
    return



def get_complete_result(session, model, prompt, temperature, max_tokens, guardrails, system_prompt=None):
    """Handles the Complete functionality in playground mode.
    
    Args:
        session: Snowflake session.
        model (str): The model to use for completion.
        prompt (str): The user prompt text.
        temperature (float): Temperature parameter for generation.
        max_tokens (int): Maximum tokens to generate.
        guardrails (bool): Whether to enable guardrails.
        system_prompt (str, optional): System prompt to prepend.
        
    Returns:
        dict: The completion result from the model.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    # Use parameterized query to avoid JSON escaping issues
    query = """
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        ?,
        PARSE_JSON(?),
        OBJECT_CONSTRUCT(
            'temperature', ?,
            'max_tokens', ?,
            'guardrails', ?
        )
    );
    """
    
    # Parameters for the query
    params = [
        model,
        json.dumps(messages),  # JSON string without SQL escaping
        temperature,
        max_tokens,
        guardrails
    ]
    
    print("messages: ", messages)
    print("query parameters: ", params)
    try:
        result = session.sql(query, params).collect()[0][0]
        print("result: ", result)
        
        # Parse the result if it's a JSON string
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                return parsed_result
            except json.JSONDecodeError:
                # If parsing fails, return the raw string wrapped in a basic structure
                return {
                    "choices": [{"messages": result}],
                    "model": "unknown",
                    "usage": {}
                }
        else:
            # If result is already a dict/object, return as is
            return result
            
    except SnowparkSQLException as e:
        raise e

def get_complete_multimodal_result(session, model, prompt, stage, files):
    """Handles the Complete functionality in playground mode for multimodal inputs.
    
    Args:
        session: Snowflake session.
        model (str): The model to use for completion.
        prompt (str): The user prompt text.
        stage (str): Stage name containing files.
        files (list): List of file names to include in the prompt.
        
    Returns:
        dict: The completion result from the model.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    if len(files) == 1:
        q = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{prompt}',
            TO_FILE('@{stage}','{files[0]}'))
        """
        try:
            result = session.sql(q).collect()
            return list(result[0])[0]
        except SnowparkSQLException as e:
            raise e
    else:
        file_str = ",\n".join([f"TO_FILE('@{stage}', '{file}')" for file in files])
        q = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                '{model}',
                PROMPT('{prompt}',
                    {file_str}
                )
            )
            """
        try:
            result = session.sql(q).collect()
            return list(result[0])[0]
        except SnowparkSQLException as e:
            raise e

def get_translation(session, text, source_lang, target_lang):
    """Handles the Translate functionality in playground mode using SQL.
    
    Args:
        session: Snowflake session.
        text (str): Text to translate.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        
    Returns:
        str: The translated text.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    query = f"""
    SELECT SNOWFLAKE.CORTEX.TRANSLATE('{escape_sql_string(text)}', '{source_lang}', '{target_lang}');
    """
    try:
        result = session.sql(query).collect()[0][0]
        return result
    except SnowparkSQLException as e:
        raise e


def get_summary(session, text):
    """Handles the Summarize functionality in playground mode using SQL.
    
    Args:
        session: Snowflake session.
        text (str): Text to summarize.
        
    Returns:
        str: The summarized text.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    query = f"""
    SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{escape_sql_string(text)}');
    """
    try:
        result = session.sql(query).collect()[0][0]
        return result
    except SnowparkSQLException as e:
        raise e


def get_extraction(session, text, query_text):
    """Handles the Extract functionality in playground mode using SQL.
    
    Args:
        session: Snowflake session.
        text (str): Source text to extract from.
        query_text (str): Query text to guide extraction.
        
    Returns:
        str: The extracted answer.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    query_text_escaped = escape_sql_string(query_text)
    query = f"""
    SELECT SNOWFLAKE.CORTEX.EXTRACT_ANSWER('{escape_sql_string(text)}', '{query_text_escaped}');
    """
    try:
        result = session.sql(query).collect()[0][0]
        return result
    except SnowparkSQLException as e:
        raise e


def get_sentiment(session, text):
    """Handles the Sentiment functionality in playground mode using SQL.
    
    Args:
        session: Snowflake session.
        text (str): Text to analyze sentiment.
        
    Returns:
        str: The sentiment analysis result.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    query = f"""
    SELECT SNOWFLAKE.CORTEX.SENTIMENT('{escape_sql_string(text)}');
    """
    try:
        result = session.sql(query).collect()[0][0]
        return result
    except SnowparkSQLException as e:
        raise e
    
def get_classification(session, text, categories):
    """Handles the Classification functionality in playground mode using SQL.
    
    Args:
        session: Snowflake session.
        text (str): Text to classify.
        categories (list): List of categories for classification.
        
    Returns:
        str: The classification result.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    # Convert categories string to list of strings, categories are separated by commas
    categories = [f"'{category}'" for category in categories.split(",")]
    print("categories_str: ",categories)
    query = f"""
    SELECT SNOWFLAKE.CORTEX.CLASSIFY_TEXT('{text}', ARRAY_CONSTRUCT({','.join(categories)}));
    """
    print("query: ",query)
    try:
        result = session.sql(query).collect()[0][0]
        print("result: ",result)
        return result
    except SnowparkSQLException as e:
        raise e


def get_parse_document(session, stage, file, mode):
    """Handles the Parse Document functionality in playground mode using SQL.
    
    Args:
        session: Snowflake session.
        stage (str): Stage name containing files.
        file (str): File name to parse.
        mode (str): Mode for parsing (e.g., 'OCR', 'LAYOUT').
        
    Returns:
        str: The parsed document content.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    # Ensure stage and file are properly quoted for SQL
    stage = f"'{stage}'"
    file = f"'{file}'"
    
    print("stage: ", stage)
    print("file: ", file)
    print("mode: ", mode)
    
    # Format the mode parameter as a JSON-like string
    mode_param = "{'mode': 'OCR'}" if mode == "OCR" else "{'mode': 'LAYOUT'}"
    
    query = f"""
        SELECT TO_VARCHAR(
            SNOWFLAKE.CORTEX.PARSE_DOCUMENT(
                {stage},
                {file},
                {mode_param}
            )
        ) AS result;
    """
    
    print("query: ", query)
    try:
        result = session.sql(query).collect()[0][0]
        return result
    except SnowparkSQLException as e:
        raise e

def get_entity_sentiment(session, text, entities):
    """Handles the Entity Sentiment functionality in playground mode using SQL.
    
    Args:
        session: Snowflake session.
        text (str): Text to analyze entity sentiment.
        entities (list): List of entities to analyze.
        
    Returns:
        str: The entity sentiment analysis result.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    # Convert entities string to list of strings, entities are separated by commas
    entities = [f"'{entity}'" for entity in entities.split(",")]
    print("entities_str: ",entities)
    query = f"""
    SELECT SNOWFLAKE.CORTEX.ENTITY_SENTIMENT('{text}', ARRAY_CONSTRUCT({','.join(entities)}));
    """
    print("query: ",query)
    try:
        result = session.sql(query).collect()[0][0]
        print("result: ",result)
        return result
    except SnowparkSQLException as e:
        raise e

def get_complete_result_from_column(session, model, db, schema, table, input_column, temperature, max_tokens, guardrails, output_table, output_column, system_prompt=None, user_prompt=None):
    """Fetches content from a column and writes the completion result to an output table.
    
    Args:
        session: Snowflake session.
        model (str): The model to use for completion.
        db (str): Database name.
        schema (str): Schema name.
        table (str): Input table name.
        input_column (str): Column containing input text.
        temperature (float): Temperature parameter for generation.
        max_tokens (int): Maximum tokens to generate.
        guardrails (bool): Whether to enable guardrails.
        output_table (str): Table to write results to.
        output_column (str): Column to write results to.
        system_prompt (str, optional): System prompt to prepend.
        user_prompt (str, optional): User prompt template.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    from src.notification import log_cortex_usage
    from datetime import datetime
    
    start_time = datetime.now()
    
    # Check if the output table and column exist, and create/add the column if necessary
    columns = [
        f"{input_column} VARCHAR(16777216)",
        f"{output_column} VARCHAR(16777216)"
    ]
    # Ensure output table exists
    check_and_create_table(session, db, schema, output_table, columns)

    # Build full table names
    source_table_full = f"{db}.{schema}.{table}"
    output_table_full = f"{db}.{schema}.{output_table}"

    # Escape system prompt
    system_prompt_escaped = escape_sql_string(system_prompt) if system_prompt else ""
    user_prompt_escaped = escape_sql_string(user_prompt) if user_prompt else ""

    # Prepare user prompt by escaping special characters
    user_prompt_sql = f"'{user_prompt_escaped} <{table}>' || {input_column} || '</{table}>'"

    # Construct the SQL query without using PARSE_JSON incorrectly
    query = f"""
    INSERT INTO {output_table_full} ({input_column}, {output_column})
    SELECT {input_column},
        GET_PATH(
            SNOWFLAKE.CORTEX.COMPLETE(
                '{model}',
                ARRAY_CONSTRUCT(
                    OBJECT_CONSTRUCT('role', 'system', 'content', '{system_prompt_escaped}'),
                    OBJECT_CONSTRUCT('role', 'user', 'content', {user_prompt_sql})
                ),
                OBJECT_CONSTRUCT(
                    'temperature', {temperature},
                    'max_tokens', {max_tokens},
                    'guardrails', {str(guardrails).lower()}
                )
            ),
            'choices[0].messages'
        ) AS message_content
    FROM {source_table_full};
    """

    try:
        # Execute the batch operation
        session.sql(query).collect()
        end_time = datetime.now()
        
        # Count how many rows were processed
        row_count_query = f"SELECT COUNT(*) as count FROM {source_table_full}"
        row_count = session.sql(row_count_query).collect()[0]['COUNT']
        
        # Estimate tokens (rough approximation for batch operation)
        estimated_total_tokens = row_count * 100  # Rough estimate per row
        estimated_credits = estimated_total_tokens * 0.0001
        
        # Log the batch operation usage  
        log_cortex_usage(
            session=session,
            function_name="COMPLETE",
            model_name=model,
            total_tokens=estimated_total_tokens,
            estimated_credits=estimated_credits,
            start_time=start_time,
            end_time=end_time,
            details=f"Batch operation: {row_count} rows processed from {table}.{input_column} to {output_table}.{output_column}"
        )
        
    except SnowparkSQLException as e:
        end_time = datetime.now()
        # Log the failed attempt
        log_cortex_usage(
            session=session,
            function_name="COMPLETE",
            model_name=model,
            start_time=start_time,
            end_time=end_time,
            details=f"Batch operation failed: {str(e)}"
        )
        raise e


def get_translation_from_column(session, db, schema, table, input_column, source_lang, target_lang, output_table, output_column):
    """Fetches content from a column and writes the translation result to an output table.
    
    Args:
        session: Snowflake session.
        db (str): Database name.
        schema (str): Schema name.
        table (str): Input table name.
        input_column (str): Column containing text to translate.
        source_lang (str): Source language code.
        target_lang (str): Target language code.
        output_table (str): Table to write results to.
        output_column (str): Column to write results to.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    from src.notification import log_cortex_usage
    from datetime import datetime
    
    start_time = datetime.now()
    
    columns = [
        f"{input_column} VARCHAR(16777216)",
        f"{output_column} VARCHAR(16777216)"
    ]
    # Ensure output table exists
    check_and_create_table(session, db, schema, output_table, columns)

    source_table_full = f"{db}.{schema}.{table}"
    output_table_full = f"{db}.{schema}.{output_table}"

    query = f"""
    INSERT INTO {output_table_full} ({input_column}, {output_column})
    SELECT {input_column}, CAST(SNOWFLAKE.CORTEX.TRANSLATE({input_column}, '{source_lang}', '{target_lang}') AS STRING)
    FROM {source_table_full};
    """
    try:
        session.sql(query).collect()
        end_time = datetime.now()
        
        # Count processed rows and log usage
        row_count_query = f"SELECT COUNT(*) as count FROM {source_table_full}"
        row_count = session.sql(row_count_query).collect()[0]['COUNT']
        
        estimated_total_tokens = row_count * 50  # Rough estimate for translation
        estimated_credits = estimated_total_tokens * 0.0001
        
        log_cortex_usage(
            session=session,
            function_name="TRANSLATE",
            total_tokens=estimated_total_tokens,
            estimated_credits=estimated_credits,
            start_time=start_time,
            end_time=end_time,
            details=f"Batch translation: {row_count} rows from {source_lang} to {target_lang}"
        )
        
    except SnowparkSQLException as e:
        end_time = datetime.now()
        log_cortex_usage(
            session=session,
            function_name="TRANSLATE",
            start_time=start_time,
            end_time=end_time,
            details=f"Batch translation failed: {str(e)}"
        )
        raise e


def get_summary_from_column(session, db, schema, table, input_column, output_table, output_column):
    """Fetches content from a column and writes the summary result to an output table.
    
    Args:
        session: Snowflake session.
        db (str): Database name.
        schema (str): Schema name.
        table (str): Input table name.
        input_column (str): Column containing text to summarize.
        output_table (str): Table to write results to.
        output_column (str): Column to write results to.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    from src.notification import log_cortex_usage
    from datetime import datetime
    
    start_time = datetime.now()
    
    columns = [
        f"{input_column} VARCHAR(16777216)",
        f"{output_column} VARCHAR(16777216)"
    ]
    # Ensure output table exists
    check_and_create_table(session, db, schema, output_table, columns)

    source_table_full = f"{db}.{schema}.{table}"
    output_table_full = f"{db}.{schema}.{output_table}"

    query = f"""
    INSERT INTO {output_table_full} ({input_column}, {output_column})
    SELECT {input_column}, CAST(SNOWFLAKE.CORTEX.SUMMARIZE({input_column}) AS STRING)
    FROM {source_table_full};
    """
    try:
        session.sql(query).collect()
        end_time = datetime.now()
        
        # Count processed rows and log usage
        row_count_query = f"SELECT COUNT(*) as count FROM {source_table_full}"
        row_count = session.sql(row_count_query).collect()[0]['COUNT']
        
        estimated_total_tokens = row_count * 75  # Rough estimate for summarization
        estimated_credits = estimated_total_tokens * 0.0001
        
        log_cortex_usage(
            session=session,
            function_name="SUMMARIZE",
            total_tokens=estimated_total_tokens,
            estimated_credits=estimated_credits,
            start_time=start_time,
            end_time=end_time,
            details=f"Batch summarization: {row_count} rows processed"
        )
        
    except SnowparkSQLException as e:
        end_time = datetime.now()
        log_cortex_usage(
            session=session,
            function_name="SUMMARIZE",
            start_time=start_time,
            end_time=end_time,
            details=f"Batch summarization failed: {str(e)}"
        )
        raise e

def get_extraction_from_column(session, db, schema, table, input_column, query_text, output_table, output_column):
    """Fetches content from a column and writes the extracted answer to an output table.
    
    Args:
        session: Snowflake session.
        db (str): Database name.
        schema (str): Schema name.
        table (str): Input table name.
        input_column (str): Column containing source text.
        query_text (str): Query text to guide extraction.
        output_table (str): Table to write results to.
        output_column (str): Column to write results to.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    from src.notification import log_cortex_usage
    from datetime import datetime
    
    start_time = datetime.now()
    
    columns = [
        f"{input_column} VARCHAR(16777216)",
        f"{output_column} VARCHAR(16777216)"
    ]
    # Ensure output table exists
    check_and_create_table(session, db, schema, output_table, columns)

    source_table_full = f"{db}.{schema}.{table}"
    output_table_full = f"{db}.{schema}.{output_table}"

    query_text_escaped = escape_sql_string(query_text)

    query = f"""
    INSERT INTO {output_table_full} ({input_column}, {output_column})
    SELECT {input_column}, CAST(SNOWFLAKE.CORTEX.EXTRACT_ANSWER({input_column}, '{query_text_escaped}') AS STRING)
    FROM {source_table_full};
    """
    try:
        session.sql(query).collect()
        end_time = datetime.now()
        
        # Count processed rows and log usage
        row_count_query = f"SELECT COUNT(*) as count FROM {source_table_full}"
        row_count = session.sql(row_count_query).collect()[0]['COUNT']
        
        estimated_total_tokens = row_count * 60  # Rough estimate for extraction
        estimated_credits = estimated_total_tokens * 0.0001
        
        log_cortex_usage(
            session=session,
            function_name="EXTRACT_ANSWER",
            total_tokens=estimated_total_tokens,
            estimated_credits=estimated_credits,
            start_time=start_time,
            end_time=end_time,
            details=f"Batch extraction: {row_count} rows processed with query '{query_text}'"
        )
        
    except SnowparkSQLException as e:
        end_time = datetime.now()
        log_cortex_usage(
            session=session,
            function_name="EXTRACT_ANSWER",
            start_time=start_time,
            end_time=end_time,
            details=f"Batch extraction failed: {str(e)}"
        )
        raise e


def get_sentiment_from_column(session, db, schema, table, input_column, output_table, output_column):
    """Fetches content from a column and writes the sentiment analysis result to an output table.
    
    Args:
        session: Snowflake session.
        db (str): Database name.
        schema (str): Schema name.
        table (str): Input table name.
        input_column (str): Column containing text to analyze.
        output_table (str): Table to write results to.
        output_column (str): Column to write results to.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    from src.notification import log_cortex_usage
    from datetime import datetime
    
    start_time = datetime.now()
    
    columns = [
        f"{input_column} VARCHAR(16777216)",
        f"{output_column} VARCHAR(16777216)"
    ]
    # Ensure output table exists
    check_and_create_table(session, db, schema, output_table, columns)

    source_table_full = f"{db}.{schema}.{table}"
    output_table_full = f"{db}.{schema}.{output_table}"

    query = f"""
    INSERT INTO {output_table_full} ({input_column}, {output_column})
    SELECT {input_column}, CAST(SNOWFLAKE.CORTEX.SENTIMENT({input_column}) AS STRING)
    FROM {source_table_full};
    """
    try:
        session.sql(query).collect()
        end_time = datetime.now()
        
        # Count processed rows and log usage
        row_count_query = f"SELECT COUNT(*) as count FROM {source_table_full}"
        row_count = session.sql(row_count_query).collect()[0]['COUNT']
        
        estimated_total_tokens = row_count * 30  # Rough estimate for sentiment analysis
        estimated_credits = estimated_total_tokens * 0.0001
        
        log_cortex_usage(
            session=session,
            function_name="SENTIMENT",
            total_tokens=estimated_total_tokens,
            estimated_credits=estimated_credits,
            start_time=start_time,
            end_time=end_time,
            details=f"Batch sentiment analysis: {row_count} rows processed"
        )
        
    except SnowparkSQLException as e:
        end_time = datetime.now()
        log_cortex_usage(
            session=session,
            function_name="SENTIMENT",
            start_time=start_time,
            end_time=end_time,
            details=f"Batch sentiment analysis failed: {str(e)}"
        )
        raise e

    
def create_vector_embedding_from_stage(session, db, schema, stage, embedding_type, embedding_model,output_table):
    """Creates vector embeddings for all files in a selected stage.
    
    Args:
        session: Snowflake session.
        db (str): Database name.
        schema (str): Schema name.
        stage (str): Stage name containing files.
        embedding_type (str): Type of embedding to create.
        embedding_model (str): Model to use for embeddings.
        output_table (str): Table to write embeddings to.
        
    Raises:
        SnowparkSQLException: If the query fails.
    """
    from src.notification import log_cortex_usage
    from datetime import datetime
    
    start_time = datetime.now()
    
    stage_path = f"@{db}.{schema}.{stage}"
    output_table_full = f"{db}.{schema}.{output_table}"

    # Define the columns required in the output table
    columns = [
        "relative_path VARCHAR(16777216)","size NUMBER(38,0)","file_url VARCHAR(16777216)",
        "scoped_file_url VARCHAR(16777216)", "chunk VARCHAR(16777216)"
    ]
    if embedding_model == "EMBED_TEXT_768":
        columns.append("vector_embeddings VECTOR(FLOAT, 768)")
    else:
        columns.append("vector_embeddings VECTOR(FLOAT, 1024)")
    # Ensure the output table exists with the required columns
    print(f"📋 Creating/checking output table: {output_table_full}")
    check_and_create_table(session, db, schema, output_table, columns)

    # Construct the SQL query to process all files in the stage
    query = f"""
    INSERT INTO {output_table_full} (relative_path, size, file_url, scoped_file_url, chunk, vector_embeddings)
    SELECT 
        relative_path, 
        size,
        file_url,
        build_scoped_file_url('{stage_path}', relative_path) AS scoped_file_url,
        func.chunk AS chunk,
        SNOWFLAKE.CORTEX.{embedding_type}('{embedding_model}', func.chunk) AS vector_embeddings
    FROM 
        directory('{stage_path}') AS dir,
        TABLE(pdf_text_chunker(build_scoped_file_url('{stage_path}', relative_path))) AS func;
    """
    try:
        print(f"📝 Executing embedding query for stage {stage_path}:")
        print(query)
        session.sql(query).collect()
        end_time = datetime.now()
        
        # Count how many chunks were processed
        print(f"📊 Checking final record count in {output_table_full}...")
        chunk_count_query = f"SELECT COUNT(*) as count FROM {output_table_full}"
        chunk_count = session.sql(chunk_count_query).collect()[0]['COUNT']
        print(f"📊 Total records in output table: {chunk_count}")
        
        # Estimate tokens for embedding operation
        estimated_total_tokens = chunk_count * 200  # Rough estimate per chunk for embeddings
        estimated_credits = estimated_total_tokens * 0.0001
        
        log_cortex_usage(
            session=session,
            function_name=embedding_type,
            model_name=embedding_model,
            total_tokens=estimated_total_tokens,
            estimated_credits=estimated_credits,
            start_time=start_time,
            end_time=end_time,
            details=f"Vector embedding creation: {chunk_count} chunks processed from stage {stage}"
        )
        
        try:
            st.success(f"Vector embeddings created successfully for all files in {stage}. Results saved to {output_table}.")
        except:
            print(f"✓ Vector embeddings created successfully for all files in {stage}. Results saved to {output_table}.")
        
    except SnowparkSQLException as e:
        end_time = datetime.now()
        log_cortex_usage(
            session=session,
            function_name=embedding_type,
            model_name=embedding_model,
            start_time=start_time,
            end_time=end_time,
            details=f"Vector embedding creation failed: {str(e)}"
        )
        try:
            st.error(f"Failed to create embeddings: {e}")
        except:
            print(f"✗ Failed to create embeddings: {e}")
        raise e

def create_cortex_search_service(session, database, schema, table, column, attributes, service_name, embedding_model, warehouse):
    """Creates a Cortex Search Service for the specified table and column.

    Args:
        session: Snowflake session.
        database (str): Database name.
        schema (str): Schema name.
        table (str): Table name.
        columns (list[str]): Column to use for the search service.
        service_name (str): Name of the search service to create.
        embedding_model (str): Model to use for embeddings.
        warehouse (str): Snowflake warehouse to use.

    Raises:
        SnowparkSQLException: If the query fails.
    """

    attributes = " , ".join(attributes)

    print("column: ",column)
    print("attributes: ",attributes)

    query = f"""
        CREATE OR REPLACE CORTEX SEARCH SERVICE {database}.{schema}.{service_name}
        ON {column}
        ATTRIBUTES {attributes}
        WAREHOUSE = {warehouse}
        TARGET_LAG = '1 day'
        AS (
            SELECT * FROM {database}.{schema}.{table}
        );
    """
    try:
        print("query: ",query)
        session.sql(query).collect()
        st.success(f"Cortex Search Service {service_name} created successfully.")
    except Exception as e:
        st.error(f"Failed to create Cortex Search Service {service_name}: {e}")
        raise e
