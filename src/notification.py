import streamlit as st
from snowflake.snowpark import Session
from datetime import datetime, timedelta
import pandas as pd

def create_notification_table(session: Session):
    """
    Creates a notification table in Snowflake if it doesn't exist.
    
    Args:
        session (Session): Active Snowflake session object
        
    Raises:
        Exception: If table creation fails
    """
    try:
        # First, create the basic table if it doesn't exist
        session.sql("""
            CREATE TABLE IF NOT EXISTS notification (
                id INTEGER IDENTITY PRIMARY KEY,
                operation_type STRING,
                status STRING,
                created_at TIMESTAMP,
                completed_at TIMESTAMP,
                details STRING
            )
        """).collect()
        
        # Then, add new columns if they don't exist
        try:
            # Check if the new columns exist by trying to select them
            session.sql("SELECT source_type FROM notification LIMIT 1").collect()
        except:
            # If the columns don't exist, add them
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                source_type STRING DEFAULT 'USER'
            """).collect()
            
        try:
            session.sql("SELECT function_name FROM notification LIMIT 1").collect()
        except:
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                function_name STRING
            """).collect()
            
        try:
            session.sql("SELECT model_name FROM notification LIMIT 1").collect()
        except:
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                model_name STRING
            """).collect()
            
        try:
            session.sql("SELECT warehouse_id FROM notification LIMIT 1").collect()
        except:
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                warehouse_id STRING
            """).collect()
            
        try:
            session.sql("SELECT tokens FROM notification LIMIT 1").collect()
        except:
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                tokens INTEGER
            """).collect()
            
        try:
            session.sql("SELECT token_credits FROM notification LIMIT 1").collect()
        except:
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                token_credits FLOAT
            """).collect()
            
        try:
            session.sql("SELECT start_time FROM notification LIMIT 1").collect()
        except:
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                start_time TIMESTAMP
            """).collect()
            
        try:
            session.sql("SELECT end_time FROM notification LIMIT 1").collect()
        except:
            session.sql("""
                ALTER TABLE notification ADD COLUMN 
                end_time TIMESTAMP
            """).collect()
            
    except Exception as e:
        st.error(f"Failed to create notification table: {e}")
        print(f"Error creating notification table: {e}")
        raise Exception(f"Failed to create notification table: {e}")

def create_logs_table(session: Session):
    """
    Creates a logs table in Snowflake if it doesn't exist.
    
    Args:
        session (Session): Active Snowflake session object
    """
    session.sql("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER IDENTITY PRIMARY KEY,
            operation_type STRING,
            error_message STRING,
            created_at TIMESTAMP
        )
    """).collect()

def add_notification_entry(session: Session, operation_type: str, status: str, details: str, 
                          source_type: str = 'USER', function_name: str = None, model_name: str = None,
                          warehouse_id: str = None, tokens: int = None, token_credits: float = None,
                          start_time = None, end_time = None, completed_at = None) -> int:
    """
    Adds a new notification entry to the notification table.
    
    Args:
        session (Session): Active Snowflake session object
        operation_type (str): Type of operation being performed
        status (str): Current status of the operation
        details (str): Additional details about the operation
        source_type (str): Source of the notification ('USER', 'CORTEX_FUNCTIONS', etc.)
        function_name (str, optional): Name of the function (for Cortex data)
        model_name (str, optional): Model name used
        warehouse_id (str, optional): Warehouse ID
        tokens (int, optional): Number of tokens
        token_credits (float, optional): Credits consumed
        start_time (optional): Start time for the operation
        end_time (optional): End time for the operation
        completed_at (optional): Completion time for the operation
        
    Returns:
        int: ID of the newly created notification entry
        
    Raises:
        Exception: If notification ID retrieval fails
    """
    if not operation_type:
        operation_type = "Unknown Operation"
    if not status:
        status = "In-Progress"
    if not details:
        details = "No details provided"
    if not source_type:
        source_type = "USER"

    create_notification_table(session)
    
    # Build the insert query with all fields
    # Set completed_at based on provided value or status
    if completed_at:
        completed_at_value = f"'{completed_at}'"
    elif status.lower() in ['completed', 'success', 'finished']:
        completed_at_value = "CURRENT_TIMESTAMP"
    else:
        completed_at_value = 'NULL'
    
    insert_query = f"""
        INSERT INTO notification (
            operation_type, status, created_at, completed_at, details, source_type, 
            function_name, model_name, warehouse_id, tokens, token_credits, 
            start_time, end_time
        )
        VALUES (
            '{operation_type}', '{status}', CURRENT_TIMESTAMP, {completed_at_value}, '{details}', '{source_type}',
            {f"'{function_name}'" if function_name else 'NULL'},
            {f"'{model_name}'" if model_name else 'NULL'},
            {f"'{warehouse_id}'" if warehouse_id else 'NULL'},
            {tokens if tokens is not None else 'NULL'},
            {token_credits if token_credits is not None else 'NULL'},
            {f"'{start_time}'" if start_time else 'NULL'},
            {f"'{end_time}'" if end_time else 'NULL'}
        )
    """
    session.sql(insert_query).collect()

    # Fetch the last inserted notification ID based on created_at
    id_query = """
        SELECT id FROM notification
        WHERE created_at = (
            SELECT MAX(created_at) FROM notification
        )
    """
    
    result = session.sql(id_query).collect()

    # Return the notification ID
    if result:
        notification_id = result[0]["ID"]
        print(f"Inserted notification ID: {notification_id}")
        return notification_id
    else:
        raise Exception("Failed to retrieve notification ID")

def update_notification_entry(session: Session, notification_id: int, status: str):
    """
    Updates the status and completion time of an existing notification entry.
    
    Args:
        session (Session): Active Snowflake session object
        notification_id (int): ID of the notification to update
        status (str): New status to set for the notification
    """
    if not status:
        status = 'Unknown Status'

    query = f"""
        UPDATE notification
        SET status = '{status}', completed_at = CURRENT_TIMESTAMP
        WHERE id = {notification_id}
    """
    session.sql(query).collect()

def update_notification_fine_tune_entry(session: Session, notification_id: int, details: str):
    """
    Updates the status and completion time of an existing notification entry.
    
    Args:
        session (Session): Active Snowflake session object
        notification_id (int): ID of the notification to update
        status (str): New status to set for the notification
    """
    if not details:
        status = 'Unknown details'

    query = f"""
        UPDATE notification
        SET details = '{details}', completed_at = CURRENT_TIMESTAMP
        WHERE id = {notification_id}
    """
    session.sql(query).collect()

def escape_sql_string(value: str) -> str:
    """
    Escapes single quotes in SQL strings to prevent SQL injection.
    
    Args:
        value (str): String to escape
        
    Returns:
        str: Escaped string with single quotes doubled
    """
    return value.replace("'", "''") if value else value

def log_cortex_usage(session: Session, function_name: str, model_name: str = None, 
                    input_tokens: int = None, output_tokens: int = None, 
                    total_tokens: int = None, estimated_credits: float = None,
                    start_time: datetime = None, end_time: datetime = None,
                    details: str = None, warehouse_id: str = None) -> int:
    """
    Logs Cortex function usage to the notification table with detailed usage information.
    
    Args:
        session (Session): Active Snowflake session object
        function_name (str): Name of the Cortex function (COMPLETE, TRANSLATE, etc.)
        model_name (str, optional): Model name used
        input_tokens (int, optional): Number of input tokens
        output_tokens (int, optional): Number of output tokens  
        total_tokens (int, optional): Total tokens (input + output)
        estimated_credits (float, optional): Estimated credits consumed
        start_time (datetime, optional): Start time of the operation
        end_time (datetime, optional): End time of the operation
        details (str, optional): Additional details about the operation
        warehouse_id (str, optional): Warehouse ID used
        
    Returns:
        int: ID of the newly created notification entry
    """
    if not start_time:
        start_time = datetime.now()
    if not end_time:
        end_time = datetime.now()
    
    # Calculate total tokens if not provided
    if not total_tokens and input_tokens and output_tokens:
        total_tokens = input_tokens + output_tokens
    elif not total_tokens:
        total_tokens = input_tokens or output_tokens
    
    # Build detailed description
    detail_parts = []
    if model_name:
        detail_parts.append(f"Model: {model_name}")
    if input_tokens:
        detail_parts.append(f"Input Tokens: {input_tokens}")
    if output_tokens:
        detail_parts.append(f"Output Tokens: {output_tokens}")
    if total_tokens:
        detail_parts.append(f"Total Tokens: {total_tokens}")
    if estimated_credits:
        detail_parts.append(f"Estimated Credits: {estimated_credits:.6f}")
    if details:
        detail_parts.append(f"Details: {details}")
    
    combined_details = ", ".join(detail_parts) if detail_parts else f"Cortex {function_name} operation completed"
    
    return add_notification_entry(
        session=session,
        operation_type=function_name,
        status="Completed",
        details=combined_details,
        source_type="CORTEX_FUNCTIONS",
        function_name=function_name,
        model_name=model_name,
        warehouse_id=warehouse_id,
        tokens=total_tokens,
        token_credits=estimated_credits,
        start_time=start_time,
        end_time=end_time,
        completed_at=end_time  # Use end_time as completed_at for accuracy
    )

def execute_cortex_with_logging(session: Session, cortex_query: str, function_name: str = None, 
                               model_name: str = None, input_text: str = None, 
                               warehouse_id: str = None, query_params=None, **kwargs) -> tuple:
    """
    Universal wrapper for any Cortex function that automatically logs usage details.
    
    Args:
        session (Session): Active Snowflake session object
    cortex_query (str): The Cortex SQL query to execute (can include '?' placeholders)
        function_name (str, optional): Name of the Cortex function (auto-detected if not provided)
        model_name (str, optional): Model name used (for functions that use models)
        input_text (str, optional): Input text for token estimation
        warehouse_id (str, optional): Warehouse ID
    query_params (list | tuple | None): Optional parameters to bind to the query
        **kwargs: Additional parameters for logging
        
    Returns:
        tuple: (result, notification_id) - The query result and notification ID
        
    Examples:
        # Complete function
        result, nid = execute_cortex_with_logging(
            session, 
            "SELECT SNOWFLAKE.CORTEX.COMPLETE('llama2-7b-chat', 'Hello world')",
            function_name="COMPLETE",
            model_name="llama2-7b-chat",
            input_text="Hello world"
        )
        
        # Translate function
        result, nid = execute_cortex_with_logging(
            session,
            "SELECT SNOWFLAKE.CORTEX.TRANSLATE('Hello', 'en', 'es')",
            function_name="TRANSLATE",
            input_text="Hello"
        )
        
        # Any custom Cortex query
        result, nid = execute_cortex_with_logging(
            session,
            "SELECT SNOWFLAKE.CORTEX.SUMMARIZE('Long text here...')",
            input_text="Long text here..."
        )
    """
    start_time = datetime.now()
    
    # Auto-detect function name if not provided
    if not function_name:
        cortex_query_upper = cortex_query.upper()
        if 'CORTEX.COMPLETE' in cortex_query_upper:
            function_name = "COMPLETE"
        elif 'CORTEX.TRANSLATE' in cortex_query_upper:
            function_name = "TRANSLATE"
        elif 'CORTEX.SUMMARIZE' in cortex_query_upper:
            function_name = "SUMMARIZE"
        elif 'CORTEX.SENTIMENT' in cortex_query_upper:
            function_name = "SENTIMENT"
        elif 'CORTEX.EXTRACT_ANSWER' in cortex_query_upper:
            function_name = "EXTRACT_ANSWER"
        elif 'CORTEX.CLASSIFY_TEXT' in cortex_query_upper:
            function_name = "CLASSIFY_TEXT"
        else:
            function_name = "CORTEX_FUNCTION"
    
    # Estimate input tokens if input_text provided
    input_tokens = None
    if input_text:
        input_tokens = max(1, len(input_text) // 4)
    
    try:
        # Execute the Cortex query (bind parameters if provided)
        if query_params is not None:
            result = session.sql(cortex_query, query_params).collect()
        else:
            result = session.sql(cortex_query).collect()
        end_time = datetime.now()
        
        if result:
            # Get the result value (assuming single column result)
            result_value = result[0][0]
            if hasattr(result_value, 'values'):
                response_text = str(list(result_value.values())[0])
            else:
                response_text = str(result_value)
            
            # Estimate output tokens
            output_tokens = None
            if response_text and response_text != 'None':
                output_tokens = max(1, len(response_text) // 4)
            
            # Calculate total tokens
            total_tokens = 0
            if input_tokens:
                total_tokens += input_tokens
            if output_tokens:
                total_tokens += output_tokens
            
            # Estimate credits (rough approximation - adjust based on actual pricing)
            estimated_credits = None
            if total_tokens > 0:
                estimated_credits = total_tokens * 0.0001  # Placeholder rate
            
            # Build details
            detail_parts = []
            if input_text:
                detail_parts.append(f"Input length: {len(input_text)} chars")
            if response_text and response_text != 'None':
                detail_parts.append(f"Output length: {len(response_text)} chars")
            if kwargs:
                for key, value in kwargs.items():
                    detail_parts.append(f"{key}: {value}")
            
            details = ", ".join(detail_parts) if detail_parts else f"{function_name} operation completed"
            
            # Log the usage
            notification_id = log_cortex_usage(
                session=session,
                function_name=function_name,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_credits=estimated_credits,
                start_time=start_time,
                end_time=end_time,
                details=details,
                warehouse_id=warehouse_id
            )
            
            return response_text, notification_id
        else:
            raise Exception(f"No result returned from Cortex {function_name}")
            
    except Exception as e:
        end_time = datetime.now()
        # Log the error
        add_log_entry(session, f"CORTEX_{function_name}", str(e))
        
        # Still log the attempt with error details
        error_details = f"Error: {str(e)}"
        if input_text:
            error_details += f" | Input: {len(input_text)} chars"
        if model_name:
            error_details += f" | Model: {model_name}"
            
        notification_id = log_cortex_usage(
            session=session,
            function_name=function_name,
            model_name=model_name,
            input_tokens=input_tokens,
            start_time=start_time,
            end_time=end_time,
            details=error_details,
            warehouse_id=warehouse_id
        )
        raise e

# Convenience helper functions for common use cases
def cortex_complete(session: Session, model: str, prompt: str, **options):
    """
    Convenience wrapper for Cortex COMPLETE with logging.
    
    Args:
        session: Snowflake session
        model: Model name
        prompt: Input prompt
        **options: Additional options like max_tokens, temperature
    
    Returns:
        tuple: (result, notification_id)
    """
    # Build parameterized query to prevent SQL injection.
    # Support common options: temperature, max_tokens, guardrails. Others remain for logging only.
    temperature = options.get("temperature")
    max_tokens = options.get("max_tokens")
    guardrails = options.get("guardrails")

    params = [model, prompt]
    object_pairs = []
    if temperature is not None:
        object_pairs.append("'temperature', ?")
        params.append(temperature)
    if max_tokens is not None:
        object_pairs.append("'max_tokens', ?")
        params.append(max_tokens)
    if guardrails is not None:
        object_pairs.append("'guardrails', ?")
        params.append(guardrails)

    if object_pairs:
        options_object = f"OBJECT_CONSTRUCT({', '.join(object_pairs)})"
        query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?, {options_object}) as result"
    else:
        query = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as result"

    return execute_cortex_with_logging(
        session=session,
        cortex_query=query,
        query_params=params,
        function_name="COMPLETE",
        model_name=model,
        input_text=prompt,
        **options
    )

def cortex_translate(session: Session, text: str, source_lang: str, target_lang: str):
    """
    Convenience wrapper for Cortex TRANSLATE with logging.
    """
    query = f"SELECT SNOWFLAKE.CORTEX.TRANSLATE('{escape_sql_string(text)}', '{source_lang}', '{target_lang}') as result"
    return execute_cortex_with_logging(
        session=session,
        cortex_query=query,
        function_name="TRANSLATE",
        input_text=text,
        source_language=source_lang,
        target_language=target_lang
    )

def cortex_summarize(session: Session, text: str):
    """
    Convenience wrapper for Cortex SUMMARIZE with logging.
    """
    query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{escape_sql_string(text)}') as result"
    return execute_cortex_with_logging(
        session=session,
        cortex_query=query,
        function_name="SUMMARIZE",
        input_text=text
    )

def cortex_sentiment(session: Session, text: str):
    """
    Convenience wrapper for Cortex SENTIMENT with logging.
    """
    query = f"SELECT SNOWFLAKE.CORTEX.SENTIMENT('{escape_sql_string(text)}') as result"
    return execute_cortex_with_logging(
        session=session,
        cortex_query=query,
        function_name="SENTIMENT",
        input_text=text
    )

def add_log_entry(session: Session, operation_type: str, error_message: str):
    """
    Adds a new error log entry to the logs table.
    
    Args:
        session (Session): Active Snowflake session object
        operation_type (str): Type of operation that generated the error
        error_message (str): Description of the error that occurred
    """
    if not operation_type:
        operation_type = "Unknown Operation"
    if not error_message:
        error_message = "No error message provided"

    # Escape special characters to prevent SQL issues
    operation_type_escaped = escape_sql_string(operation_type)
    error_message_escaped = escape_sql_string(error_message)
    create_logs_table(session)
    # Construct the SQL query with escaped values
    query = f"""
        INSERT INTO logs (operation_type, error_message, created_at)
        VALUES ('{operation_type_escaped}', '{error_message_escaped}', CURRENT_TIMESTAMP)
    """
    
    # Execute the query
    session.sql(query).collect()

def fetch_notifications(session: Session, start_date=None, end_date=None):
    """
    Retrieves notification entries filtered by date range.
    
    Args:
        session (Session): Active Snowflake session object
        start_date (datetime, optional): Start date for filtering notifications
        end_date (datetime, optional): End date for filtering notifications
        
    Returns:
        pandas.DataFrame: DataFrame containing filtered notification entries
    """
    create_notification_table(session)
    
    # Try to select all columns, falling back to basic columns if new ones don't exist
    try:
        if start_date and end_date:
            # Format the dates properly for Snowflake SQL
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            query = f"""
                SELECT id, operation_type, status, created_at, completed_at, details,
                       COALESCE(source_type, 'USER') as source_type,
                       function_name, model_name, warehouse_id, tokens, token_credits,
                       start_time, end_time
                FROM notification
                WHERE created_at BETWEEN '{start_date_str}' AND '{end_date_str}'
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT id, operation_type, status, created_at, completed_at, details,
                       COALESCE(source_type, 'USER') as source_type,
                       function_name, model_name, warehouse_id, tokens, token_credits,
                       start_time, end_time
                FROM notification 
                ORDER BY created_at DESC
            """
        
        return session.sql(query).to_pandas()
    except Exception as e:
        # Fallback to basic columns if new columns don't exist
        print(f"Warning: Using fallback query for notifications: {e}")
        if start_date and end_date:
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            query = f"""
                SELECT id, operation_type, status, created_at, completed_at, details,
                       'USER' as source_type,
                       NULL as function_name, NULL as model_name, NULL as warehouse_id, 
                       NULL as tokens, NULL as token_credits,
                       NULL as start_time, NULL as end_time
                FROM notification
                WHERE created_at BETWEEN '{start_date_str}' AND '{end_date_str}'
                ORDER BY created_at DESC
            """
        else:
            query = """
                SELECT id, operation_type, status, created_at, completed_at, details,
                       'USER' as source_type,
                       NULL as function_name, NULL as model_name, NULL as warehouse_id, 
                       NULL as tokens, NULL as token_credits,
                       NULL as start_time, NULL as end_time
                FROM notification 
                ORDER BY created_at DESC
            """
        
        return session.sql(query).to_pandas()

def fetch_logs(session: Session, start_date=None, end_date=None):
    """
    Retrieves log entries filtered by date range.
    
    Args:
        session (Session): Active Snowflake session object
        start_date (datetime, optional): Start date for filtering logs
        end_date (datetime, optional): End date for filtering logs
        
    Returns:
        pandas.DataFrame: DataFrame containing filtered log entries
    """
    create_logs_table(session)
    if start_date and end_date:
        # Format the dates properly for Snowflake SQL
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT * FROM logs
            WHERE created_at BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY created_at DESC
        """
    else:
        query = "SELECT * FROM logs ORDER BY created_at DESC"
    
    return session.sql(query).to_pandas()

def fetch_unified_notifications(session: Session, start_date=None, end_date=None):
    """
    Retrieves unified notification data including both user notifications and Cortex usage data.
    
    Args:
        session (Session): Active Snowflake session object
        start_date (datetime, optional): Start date for filtering notifications
        end_date (datetime, optional): End date for filtering notifications
        
    Returns:
        pandas.DataFrame: DataFrame containing unified notification entries
    """
    import pandas as pd
    
    create_notification_table(session)
    
    # Get user notifications
    user_notifications = fetch_notifications(session, start_date, end_date)
    
    # Prepare list to collect all data
    all_data = []
    
    # Add user notifications to the list
    if not user_notifications.empty:
        for _, row in user_notifications.iterrows():
            all_data.append({
                'ID': row.get('ID'),
                'OPERATION_TYPE': row.get('OPERATION_TYPE'),
                'STATUS': row.get('STATUS'),
                'CREATED_AT': row.get('CREATED_AT'),
                'COMPLETED_AT': row.get('COMPLETED_AT'),
                'DETAILS': row.get('DETAILS'),
                'SOURCE_TYPE': row.get('SOURCE_TYPE', 'USER'),
                'FUNCTION_NAME': row.get('FUNCTION_NAME'),
                'MODEL_NAME': row.get('MODEL_NAME'),
                'WAREHOUSE_ID': row.get('WAREHOUSE_ID'),
                'TOKENS': row.get('TOKENS'),
                'TOKEN_CREDITS': row.get('TOKEN_CREDITS'),
                'START_TIME': row.get('START_TIME'),
                'END_TIME': row.get('END_TIME')
            })
    
    # Convert start_date and end_date to strings for Snowflake queries
    if start_date and end_date:
        start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Fetch and add Cortex Functions Usage data
    try:
        if start_date and end_date:
            query = f"""
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
                WHERE START_TIME BETWEEN '{start_date_str}' AND '{end_date_str}'
                ORDER BY START_TIME DESC
            """
        else:
            query = """
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
                ORDER BY START_TIME DESC
                LIMIT 1000
            """
        cortex_functions = session.sql(query).to_pandas()
        
        if not cortex_functions.empty:
            for index, row in cortex_functions.iterrows():
                all_data.append({
                    'ID': 1000000 + index,  # Use large base number to avoid conflicts with user notifications
                    'OPERATION_TYPE': row.get('FUNCTION_NAME', 'CORTEX_FUNCTION'),
                    'STATUS': 'COMPLETED',
                    'CREATED_AT': row.get('START_TIME'),
                    'COMPLETED_AT': row.get('END_TIME'),
                    'DETAILS': f"Model: {row.get('MODEL_NAME', 'N/A')}, Tokens: {row.get('TOKENS', 0)}, Credits: {row.get('TOKEN_CREDITS', 0)}",
                    'SOURCE_TYPE': 'CORTEX_FUNCTIONS',
                    'FUNCTION_NAME': row.get('FUNCTION_NAME'),
                    'MODEL_NAME': row.get('MODEL_NAME'),
                    'WAREHOUSE_ID': str(row.get('WAREHOUSE_ID', '')),
                    'TOKENS': row.get('TOKENS'),
                    'TOKEN_CREDITS': row.get('TOKEN_CREDITS'),
                    'START_TIME': row.get('START_TIME'),
                    'END_TIME': row.get('END_TIME')
                })
    except Exception as e:
        print(f"Failed to fetch Cortex Functions data: {e}")
    
    # Fetch and add Cortex Functions Query Usage data (query-level granularity)
    try:
        if start_date and end_date:
            query = f"""
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY
                WHERE QUERY_ID IN (
                    SELECT QUERY_ID FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
                    WHERE START_TIME BETWEEN '{start_date_str}' AND '{end_date_str}'
                )
                ORDER BY QUERY_ID DESC
            """
        else:
            query = """
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY
                ORDER BY QUERY_ID DESC
                LIMIT 1000
            """
        cortex_query_functions = session.sql(query).to_pandas()
        
        if not cortex_query_functions.empty:
            for index, row in cortex_query_functions.iterrows():
                all_data.append({
                    'ID': 2000000 + index,  # Use different base number for query functions
                    'OPERATION_TYPE': row.get('FUNCTION_NAME', 'CORTEX_QUERY'),
                    'STATUS': 'COMPLETED',
                    'CREATED_AT': None,  # No timestamp in query usage table
                    'COMPLETED_AT': None,
                    'DETAILS': f"Query-level: Model: {row.get('MODEL_NAME', 'N/A')}, Tokens: {row.get('TOKENS', 0)}, Credits: {row.get('TOKEN_CREDITS', 0)}",
                    'SOURCE_TYPE': 'CORTEX_FUNCTIONS_QUERY',
                    'FUNCTION_NAME': row.get('FUNCTION_NAME'),
                    'MODEL_NAME': row.get('MODEL_NAME'),
                    'WAREHOUSE_ID': str(row.get('WAREHOUSE_ID', '')),
                    'TOKENS': row.get('TOKENS'),
                    'TOKEN_CREDITS': row.get('TOKEN_CREDITS'),
                    'START_TIME': None,
                    'END_TIME': None
                })
    except Exception as e:
        print(f"Failed to fetch Cortex Functions Query data: {e}")
    
    # Fetch and add Cortex Search Usage data (using correct table name)
    try:
        if start_date and end_date:
            start_date_only = start_date.strftime('%Y-%m-%d')
            end_date_only = end_date.strftime('%Y-%m-%d')
            query = f"""
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_DAILY_USAGE_HISTORY
                WHERE USAGE_DATE BETWEEN '{start_date_only}' AND '{end_date_only}'
                ORDER BY USAGE_DATE DESC
            """
        else:
            query = """
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_DAILY_USAGE_HISTORY
                ORDER BY USAGE_DATE DESC
                LIMIT 1000
            """
        cortex_search = session.sql(query).to_pandas()
        
        if not cortex_search.empty:
            for index, row in cortex_search.iterrows():
                all_data.append({
                    'ID': 3000000 + index,  # Use different base number for search
                    'OPERATION_TYPE': 'CORTEX_SEARCH',
                    'STATUS': 'COMPLETED',
                    'CREATED_AT': row.get('USAGE_DATE'),
                    'COMPLETED_AT': row.get('USAGE_DATE'),
                    'DETAILS': f"Service: {row.get('SERVICE_NAME', 'N/A')}, Credits: {row.get('CREDITS', 0)}",
                    'SOURCE_TYPE': 'CORTEX_SEARCH',
                    'FUNCTION_NAME': row.get('SERVICE_NAME'),
                    'MODEL_NAME': row.get('MODEL_NAME'),
                    'WAREHOUSE_ID': None,
                    'TOKENS': row.get('TOKENS'),
                    'TOKEN_CREDITS': row.get('CREDITS'),
                    'START_TIME': row.get('USAGE_DATE'),
                    'END_TIME': row.get('USAGE_DATE')
                })
    except Exception as e:
        print(f"Failed to fetch Cortex Search data: {e}")
    
    # Fetch and add Cortex Fine-tuning Usage data
    try:
        if start_date and end_date:
            query = f"""
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FINE_TUNING_USAGE_HISTORY
                WHERE START_TIME BETWEEN '{start_date_str}' AND '{end_date_str}'
                ORDER BY START_TIME DESC
            """
        else:
            query = """
                SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FINE_TUNING_USAGE_HISTORY
                ORDER BY START_TIME DESC
                LIMIT 1000
            """
        cortex_fine_tuning = session.sql(query).to_pandas()
        
        if not cortex_fine_tuning.empty:
            for index, row in cortex_fine_tuning.iterrows():
                all_data.append({
                    'ID': 4000000 + index,  # Use different base number for fine-tuning
                    'OPERATION_TYPE': 'CORTEX_FINE_TUNING',
                    'STATUS': 'COMPLETED',
                    'CREATED_AT': row.get('START_TIME'),
                    'COMPLETED_AT': row.get('END_TIME'),
                    'DETAILS': f"Base Model: {row.get('MODEL_NAME', 'N/A')}, Tokens: {row.get('TOKENS', 0)}, Credits: {row.get('TOKEN_CREDITS', 0)}",
                    'SOURCE_TYPE': 'CORTEX_FINE_TUNING',
                    'FUNCTION_NAME': 'FINE_TUNE',
                    'MODEL_NAME': row.get('MODEL_NAME'),
                    'WAREHOUSE_ID': str(row.get('WAREHOUSE_ID', '')),
                    'TOKENS': row.get('TOKENS'),
                    'TOKEN_CREDITS': row.get('TOKEN_CREDITS'),
                    'START_TIME': row.get('START_TIME'),
                    'END_TIME': row.get('END_TIME')
                })
    except Exception as e:
        print(f"Failed to fetch Cortex Fine-tuning data: {e}")
    
    # Convert to DataFrame and sort by created_at
    if all_data:
        unified_df = pd.DataFrame(all_data)
        # Convert CREATED_AT to datetime with proper timezone handling
        try:
            # First, handle timezone-aware datetimes by converting to UTC
            unified_df['CREATED_AT'] = pd.to_datetime(unified_df['CREATED_AT'], utc=True)
            # Also handle COMPLETED_AT for consistency
            unified_df['COMPLETED_AT'] = pd.to_datetime(unified_df['COMPLETED_AT'], utc=True)
            # Handle START_TIME and END_TIME as well
            unified_df['START_TIME'] = pd.to_datetime(unified_df['START_TIME'], utc=True)
            unified_df['END_TIME'] = pd.to_datetime(unified_df['END_TIME'], utc=True)
            unified_df = unified_df.sort_values('CREATED_AT', ascending=False)
        except Exception as e:
            print(f"Warning: Could not sort by CREATED_AT: {e}")
            # Fallback: try without timezone conversion
            try:
                unified_df['CREATED_AT'] = pd.to_datetime(unified_df['CREATED_AT'])
                unified_df['COMPLETED_AT'] = pd.to_datetime(unified_df['COMPLETED_AT'])
                unified_df['START_TIME'] = pd.to_datetime(unified_df['START_TIME'])
                unified_df['END_TIME'] = pd.to_datetime(unified_df['END_TIME'])
                unified_df = unified_df.sort_values('CREATED_AT', ascending=False)
            except Exception as e2:
                print(f"Warning: Could not sort by CREATED_AT even with fallback: {e2}")
        return unified_df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'ID', 'OPERATION_TYPE', 'STATUS', 'CREATED_AT', 'COMPLETED_AT', 'DETAILS',
            'SOURCE_TYPE', 'FUNCTION_NAME', 'MODEL_NAME', 'WAREHOUSE_ID', 'TOKENS', 
            'TOKEN_CREDITS', 'START_TIME', 'END_TIME'
        ])

def display_notification(session: Session):
    """
    Displays a Streamlit interface for viewing unified notifications and logs.
    
    Creates an interactive UI with:
    - Default view showing unified notifications (user + Cortex usage data)
    - Checkbox to switch to logs view
    - Date range filtering
    - Refresh button
    - Data display in tabular format
    
    Args:
        session (Session): Active Snowflake session object
    """
    # Create notification and logs tables if they don't exist
    create_notification_table(session)
    create_logs_table(session)

    # Title with refresh button
    col1, col2 = st.columns([9, 1])
    with col1:
        st.title("Data Viewer")
    with col2:
        if st.button("↻", help="Refresh data"):
            st.rerun()  # Trigger a re-run for refreshing

    # Checkbox for view selection
    show_logs = st.checkbox("Show Logs Instead", key="show_logs_checkbox")

    # Date filter
    col_date1, col_date2 = st.columns(2)
    today = datetime.today() + timedelta(days=1)
    last_day = today - timedelta(days=7)
    with col_date1:
        start_date = st.date_input("Start Date", last_day)
    with col_date2:
        end_date = st.date_input("End Date", today)

    # Convert date inputs to datetime for Snowflake queries
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

    # Display the selected view
    if show_logs:
        st.subheader("Logs")
        logs = fetch_logs(session, start_date, end_date)
        if logs.empty:
            st.write("No logs available.")
        else:
            st.dataframe(logs, use_container_width=True)
    else:
        st.subheader("Notifications")
        try:
            unified_notifications = fetch_unified_notifications(session, start_date, end_date)
            if unified_notifications.empty:
                st.write("No notifications available.")
            else:
                # Display with better formatting
                st.dataframe(unified_notifications, use_container_width=True)
                
                # Display summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(unified_notifications))
                with col2:
                    user_count = len(unified_notifications[unified_notifications['SOURCE_TYPE'] == 'USER'])
                    st.metric("User Notifications", user_count)
                with col3:
                    cortex_count = len(unified_notifications[unified_notifications['SOURCE_TYPE'].str.startswith('CORTEX', na=False)])
                    st.metric("Cortex Usage Records", cortex_count)
                with col4:
                    total_credits = unified_notifications['TOKEN_CREDITS'].sum()
                    st.metric("Total Credits", f"{total_credits:.4f}" if pd.notnull(total_credits) else "0")
        except Exception as e:
            st.error(f"Failed to fetch unified notifications: {e}")
            print(f"Error fetching unified notifications: {e}")