import streamlit as st
from src.utils import *  # Utility functions for database operations
from src.cortex_functions import *  # Cortex-specific helper functions
from src.query_result_builder import *  # Query result formatting utilities
from src.notification import *  # Notification/toast message utilities
from src.cortex_functions import get_complete_result  # Import the complete function
import json
from snowflake.snowpark.files import SnowflakeFile
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import requests
import jwt
import re

try:
    import _snowflake
except ImportError:
    pass

# Load configuration from JSON file
config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Get Snowflake configuration
snowflake_config = config.get("snowflake_connection", {})
RSA_PRIVATE_KEY = snowflake_config.get("rsa_private_key", "")

# Default settings template for Cortex agents
SETTINGS_TEMPLATE = {
    "model": "",  # AI model to use
    "tools": [],  # List of tool specifications
    "tool_resources": {},  # Configuration for each tool
    "response_instruction": "You will always maintain a friendly tone and provide concise response. When a user asks you a question, you will also be given excerpts from different sources provided by a search engine.",
    "starter_questions": [],  # List of starter questions for the agent
}


def run_snowflake_query(session, query):
    """Execute a Snowflake SQL query and return results as a pandas DataFrame.

    Args:
        session: Snowflake session object
        query (str): SQL query to execute

    Returns:
        pandas.DataFrame: Query results, or None if an error occurs
    """
    try:
        df = session.sql(query.replace(";", ""))  # Remove trailing semicolon
        return df.to_pandas()  # Convert Snowflake result to pandas DataFrame
    except Exception as e:
        st.error(f"Error executing SQL: {str(e)}")
        return None


class StarterQuestionGenerator:
    """Class for generating and caching starter questions for Cortex agents."""

    def __init__(self, session):
        """Initialize the starter question generator.

        Args:
            session: Snowflake session object
        """
        self.session = session
        self.cache_table = "agent_starter_questions"
        self.ensure_cache_table_exists()

    def ensure_cache_table_exists(self):
        """Create the cache table for storing starter questions if it doesn't exist."""
        try:
            db = snowflake_config.get("database")
            self.session.sql(f"""
                CREATE TABLE IF NOT EXISTS {db}.PUBLIC.{self.cache_table} (
                    agent_name VARCHAR,
                    tool_signature VARCHAR,  -- Hash of tool configuration for cache invalidation
                    questions ARRAY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (agent_name, tool_signature)
                )
            """).collect()
        except Exception as e:
            st.error(f"Error creating cache table {self.cache_table}: {str(e)}")

    def _generate_tool_signature(self, tools, tool_resources):
        """Generate a signature for the tool configuration to use for caching.

        Args:
            tools (list): List of tool specifications
            tool_resources (dict): Tool resources configuration

        Returns:
            str: Hash signature of the configuration
        """
        import hashlib

        config_str = json.dumps(
            {"tools": tools, "tool_resources": tool_resources}, sort_keys=True
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def _read_semantic_model_file(self, semantic_model_path):
        """Read semantic model YAML file from Snowflake stage.

        Args:
            semantic_model_path (str): Path to semantic model file (e.g., @DB.SCHEMA.STAGE/file.yaml)

        Returns:
            str: Content of the semantic model file
        """
        try:
            # Parse the stage path: @DB.SCHEMA.STAGE/file.yaml
            # Need to separate stage name from file path for GET_PRESIGNED_URL
            if "/" in semantic_model_path:
                stage_name, file_path = semantic_model_path.rsplit("/", 1)
            else:
                # If no file path, assume it's just the stage
                stage_name = semantic_model_path
                file_path = ""

            # Get presigned URL for the file
            presigned_query = (
                f"SELECT GET_PRESIGNED_URL('{stage_name}', '{file_path}', 3600) as url"
            )
            presigned_result = self.session.sql(presigned_query).collect()

            if presigned_result and len(presigned_result) > 0:
                presigned_url = presigned_result[0]["URL"]

                # Use requests to fetch the actual file content
                response = requests.get(presigned_url)

                if response.status_code == 200:
                    content_str = response.text

                    print("----------------------------------------")
                    print("YAML content preview:", content_str[:500])

                    # Debug: Show first 500 characters of the semantic model content
                    if config.get("mode") == "debug":
                        st.info(f"📄 Reading semantic model from {semantic_model_path}")
                        st.text(
                            f"Content preview (first 500 chars): {content_str[:500]}..."
                        )

                    return content_str
                else:
                    st.warning(
                        f"Failed to fetch file content. HTTP Status: {response.status_code}"
                    )
                    return None
            else:
                st.warning(
                    f"Could not get presigned URL for semantic model file: {semantic_model_path}"
                )
                return None

        except Exception as e:
            st.warning(
                f"Error reading semantic model file {semantic_model_path}: {str(e)}"
            )
            # Return None so we can use default questions
            return None

    def _generate_questions_for_sql_tool(self, semantic_model_content):
        """Generate starter questions for Cortex Analyst SQL tool using semantic model.

        Args:
            semantic_model_content (str): Content of the semantic model YAML file

        Returns:
            list: List of generated starter questions
        """
        if not semantic_model_content:
            return [
                "What data is available in this database?",
                "Show me the key metrics and dimensions",
                "Generate a summary of the data model",
                "What are the main tables and relationships?",
                "Create a sample analysis query",
            ]

        system_prompt = """You are a data analyst expert. Analyze the provided semantic model YAML file and generate 4 highly specific business questions that users would ask about this exact dataset.

        Instructions:
        1. Carefully examine the YAML structure for:
           - Table names and their business purpose
           - Column names that represent metrics, dimensions, dates
           - Relationships between tables
           - Time-based columns for trend analysis
           - Categorical columns for grouping/filtering
        
        2. Generate questions that are:
           - Specific to the actual table/column names found
           - Business-focused (not technical)
           - Actionable and insightful
           - Using the exact terminology from the data model
           - Mix of aggregations, trends, comparisons, and drill-downs
        
        3. Question types to include:
           - Top/bottom rankings using actual metric names
           - Time-based trends using actual date columns
           - Comparisons across actual dimension values
           - Performance analysis using real KPIs
           - Distribution analysis of key metrics
        
        4. Format as a JSON array of strings only, no other text
        
        Example: ["What are the top 5 products by total_revenue?", "Show monthly sales_amount trends over the last year", "Which store_regions have the highest customer_satisfaction_score?"]"""

        user_prompt = f"""Analyze this semantic model YAML and extract specific table names, column names, and business concepts to create targeted starter questions:

SEMANTIC MODEL:
{semantic_model_content}

Generate 4 specific business questions using the EXACT table names, column names, and metrics found in this YAML. Make the questions actionable and focused on the actual data structure provided."""

        try:
            result = get_complete_result(
                session=self.session,
                model="llama3.1-70b",
                prompt=user_prompt,
                temperature=0.7,
                max_tokens=1000,
                guardrails=True,
                system_prompt=system_prompt,
            )

            # Extract the generated text
            if isinstance(result, dict) and "choices" in result:
                generated_text = result["choices"][0].get("messages", "")
            else:
                generated_text = str(result)

            # Debug: Show what the LLM generated
            if config.get("mode") == "debug":
                st.info("🤖 LLM Response for starter questions:")
                st.text(f"Generated text: {generated_text}")

            # Try to parse as JSON array
            try:
                import re

                # Find JSON array in the response - handle multiline arrays better
                json_match = re.search(r"\[[\s\S]*?\]", generated_text)
                if json_match:
                    json_str = json_match.group().strip()
                    questions = json.loads(json_str)
                    if isinstance(questions, list):
                        # Filter out empty or very short questions
                        valid_questions = [
                            q.strip(" \"'")
                            for q in questions
                            if isinstance(q, str) and len(q.strip(" \"'")) > 15
                        ]
                        return valid_questions[:4]  # Limit to 4 questions
            except Exception as json_error:
                print(f"JSON parsing error: {json_error}")
                pass

            # Fallback: split by lines and clean
            lines = [
                line.strip() for line in generated_text.split("\n") if line.strip()
            ]
            questions = []
            for line in lines:
                # Remove bullets, numbers, quotes, and common prefixes
                clean_line = re.sub(r'^[\d\.\-\*\+"]*\s*', "", line).strip(" \"'")
                # Remove question numbering patterns like "1.", "Q1:", etc.
                clean_line = re.sub(
                    r"^(Q\d+[:.]?\s*|\d+[\.\)]\s*)", "", clean_line, flags=re.IGNORECASE
                )

                if clean_line and len(clean_line) > 15 and "?" in clean_line:
                    # Ensure it's actually a question
                    questions.append(clean_line)

            # If we still don't have good questions, try extracting from any text that looks like questions
            if not questions:
                question_pattern = r'["\']([^"\']*\?[^"\']*)["\'"]'
                potential_questions = re.findall(question_pattern, generated_text)
                questions = [
                    q.strip() for q in potential_questions if len(q.strip()) > 15
                ]

            return questions[:7] if questions else self._get_default_sql_questions()

        except Exception as e:
            st.error(f"Error generating questions from semantic model: {str(e)}")
            return self._get_default_sql_questions()

    def _generate_questions_for_search_tool(self, search_service_name):
        """Generate starter questions for Cortex Search tool.

        Args:
            search_service_name (str): Name of the search service

        Returns:
            list: List of generated starter questions
        """
        return [
            "What documents are available in this collection?",
            "Search for information about key topics",
            "Find relevant content related to my query",
            "What are the main themes in the documents?",
            "Search for specific policies or procedures",
        ]

    def _get_default_sql_questions(self):
        """Get default SQL questions when generation fails or semantic model is unavailable."""
        return [
            "What data is available in this database?",
            "Show me the key metrics and KPIs",
            "What are the main dimensions I can analyze by?",
            "Generate a summary of recent trends",
            "What are the top performing categories?",
            "Show me data distribution across different time periods",
            "What patterns can you identify in the data?",
        ]

    def generate_starter_questions(self, agent_name, tools, tool_resources):
        """Generate starter questions for an agent based on its tools.

        Args:
            agent_name (str): Name of the agent
            tools (list): List of tool specifications
            tool_resources (dict): Tool resources configuration

        Returns:
            list: List of starter questions
        """
        # Check cache first
        tool_signature = self._generate_tool_signature(tools, tool_resources)
        cached_questions = self._get_cached_questions(agent_name, tool_signature)

        if cached_questions:
            return cached_questions

        all_questions = []

        for tool in tools:
            tool_name = tool["tool_spec"]["name"]
            tool_type = tool["tool_spec"]["type"]

            if tool_type == "cortex_analyst_text_to_sql":
                # Get semantic model content and generate questions
                if tool_name in tool_resources:
                    semantic_model_path = tool_resources[tool_name].get(
                        "semantic_model_file", ""
                    )
                    if semantic_model_path:
                        semantic_content = self._read_semantic_model_file(
                            semantic_model_path
                        )
                        questions = self._generate_questions_for_sql_tool(
                            semantic_content
                        )
                        all_questions.extend(questions)

            elif tool_type == "cortex_search":
                # Generate search-related questions
                if tool_name in tool_resources:
                    search_service = tool_resources[tool_name].get("name", "")
                    questions = self._generate_questions_for_search_tool(search_service)
                    all_questions.extend(questions)

        # Remove duplicates while preserving order
        unique_questions = []
        seen = set()
        for q in all_questions:
            if q not in seen:
                unique_questions.append(q)
                seen.add(q)

        # Limit to 7 questions max
        final_questions = unique_questions[:7]

        # Cache the results
        self._cache_questions(agent_name, tool_signature, final_questions)

        return final_questions

    def _get_cached_questions(self, agent_name, tool_signature):
        """Retrieve cached questions for an agent.

        Args:
            agent_name (str): Name of the agent
            tool_signature (str): Tool configuration signature

        Returns:
            list: Cached questions or None if not found
        """
        try:
            db = snowflake_config.get("database")
            # Escape single quotes in parameters to prevent SQL injection
            safe_agent_name = agent_name.replace("'", "''")
            safe_tool_signature = tool_signature.replace("'", "''")

            query = f"""
                SELECT questions FROM {db}.PUBLIC.{self.cache_table} 
                WHERE agent_name = '{safe_agent_name}' AND tool_signature = '{safe_tool_signature}'
            """
            result = self.session.sql(query).collect()
            if result:
                # Parse the array from Snowflake
                questions_array = result[0]["QUESTIONS"]
                if isinstance(questions_array, list):
                    return questions_array
                elif isinstance(questions_array, str):
                    try:
                        return json.loads(questions_array)
                    except:
                        pass
            return None
        except Exception as e:
            st.error(f"Error retrieving cached questions: {str(e)}")
            return None

    def _cache_questions(self, agent_name, tool_signature, questions):
        """Cache generated questions for an agent.

        Args:
            agent_name (str): Name of the agent
            tool_signature (str): Tool configuration signature
            questions (list): Questions to cache
        """
        try:
            db = snowflake_config.get("database")
            questions_json = json.dumps(questions)
            # Escape single quotes in the JSON to prevent SQL injection
            safe_questions_json = questions_json.replace("'", "''")
            # Also escape single quotes in agent_name and tool_signature
            safe_agent_name = agent_name.replace("'", "''")
            safe_tool_signature = tool_signature.replace("'", "''")

            query = f"""
                MERGE INTO {db}.PUBLIC.{self.cache_table} AS target
                USING (SELECT 
                    '{safe_agent_name}' as agent_name,
                    '{safe_tool_signature}' as tool_signature,
                    PARSE_JSON('{safe_questions_json}') as questions,
                    CURRENT_TIMESTAMP as created_at
                ) AS source
                ON target.agent_name = source.agent_name AND target.tool_signature = source.tool_signature
                WHEN MATCHED THEN
                    UPDATE SET questions = source.questions, created_at = source.created_at
                WHEN NOT MATCHED THEN
                    INSERT (agent_name, tool_signature, questions, created_at)
                    VALUES (source.agent_name, source.tool_signature, source.questions, source.created_at)
            """
            self.session.sql(query).collect()
        except Exception as e:
            st.error(f"Error caching questions: {str(e)}")


class CortexAgent:
    """Class representing a Cortex AI agent with chat capabilities."""

    def __init__(self, name=None, settings=None):
        """Initialize a new Cortex agent.

        Args:
            name (str, optional): Agent name
            settings (dict, optional): Agent configuration settings
        """
        self.name = name
        self.settings = settings or {}  # Default to empty dict if None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self):
        """Convert agent to dictionary for storage.

        Returns:
            dict: Agent data in dictionary format
        """
        return {
            "name": self.name,
            "settings": json.dumps(self.settings, ensure_ascii=True),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _generate_jwt_token(self, rsa_public_key_fp):
        """Generate a new JWT token using RS256 algorithm with iat and exp in seconds, 1-hour gap."""
        if not RSA_PRIVATE_KEY:
            st.error("RSA private key not available for JWT generation.")
            return None, None

        issued_at = datetime.now(timezone.utc)
        expiration = issued_at + timedelta(hours=1)

        snowflake_account = snowflake_config["account"]
        if not snowflake_account:
            st.error("Snowflake account name not found in secrets.")
            return None, None
        user = snowflake_config["user"].upper()
        if not user:
            st.error("Snowflake user name not found in secrets.")
            return None, None

        payload = {
            "iss": f"{snowflake_account}.{user}.{rsa_public_key_fp}",
            "sub": f"{snowflake_account}.{user}",
            "iat": int(issued_at.timestamp()),
            "exp": int(expiration.timestamp()),
        }

        # print("payload: ", json.dumps(payload, indent=2))

        try:
            token = jwt.encode(payload, RSA_PRIVATE_KEY, algorithm="RS256")
            # print("token: ", token)
            return token, expiration
        except Exception as e:
            print(e)
            st.error(f"Error generating JWT token: {str(e)}")
            return None, None

    def _get_valid_jwt_token(self, session):
        """Get a valid JWT token, generating a new one if necessary."""

        rsa_public_key_fp = snowflake_config["rsa_public_key_fp"]

        if not rsa_public_key_fp:
            st.error("Failed to fetch RSA public key fingerprint.")
            return None

        token, expiry = self._generate_jwt_token(rsa_public_key_fp)
        if token:
            return token
        return None

    def chat(self, session, message, jwt_token=None):
        """Process a chat message and return response and SQL (if any).

        Args:
            session: Snowflake session object
            message (str): User input message
            jwt_token (str, optional): JWT token for authentication

        Returns:
            tuple: (response text, generated SQL) or (None, None) on error
        """

        jwt_token = self._get_valid_jwt_token(session)
        if not jwt_token:
            show_toast_message("Failed to obtain a valid JWT token", toast_type="error")
            return None, None

        # print("from chat: ", jwt_token)

        API_ENDPOINT = "/api/v2/cortex/agent:run"
        SNOWFLAKE_ACCOUNT_BASE_URL = (
            snowflake_config.get("account_base_url", "").strip().rstrip("/")
        )

        # Validate required configuration
        if not SNOWFLAKE_ACCOUNT_BASE_URL or not (
            jwt_token or self.settings.get("jwt_token")
        ):
            show_toast_message("Required configuration missing", toast_type="error")
            return None, None

        jwt_token = jwt_token or self.settings.get("jwt_token")
        payload = self._prepare_payload(message)
        headers = self._prepare_headers(jwt_token)

        try:
            url = f"{SNOWFLAKE_ACCOUNT_BASE_URL}{API_ENDPOINT}"
            if config["mode"] == "debug":
                print("started")
                with requests.post(
                    url, headers=headers, json=payload, stream=True
                ) as response:
                    response.raise_for_status()
                    print("response: ", response)
                    return self._process_stream_response(response)
            else:
                resp = _snowflake.send_snow_api_request(
                    "POST",  # method
                    API_ENDPOINT,  # path
                    {},  # headers
                    {},  # params
                    payload,  # body
                    None,  # request_guid
                    # API_TIMEOUT,  # timeout in milliseconds,
                )
                return self._process_server_stream_response(resp)
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling Cortex Agent API: {str(e)}")
            return None, None

    def _prepare_payload(self, message):
        """Prepare the API payload for chat requests.

        Args:
            message (str): User input message

        Returns:
            dict: Payload for API request
        """
        filtered_tool_resources = {}
        for tool in self.settings.get("tools", []):
            tool_name = tool["tool_spec"]["name"]
            tool_type = tool["tool_spec"]["type"]
            if tool_name in self.settings.get("tool_resources", {}):
                filtered_tool_resources[tool_name] = self._filter_tool_resources(
                    tool_name, tool_type
                )

        return {
            "model": self.settings.get("model", "llama3.1-70b"),  # Default model
            "response_instruction": self.settings.get("response_instruction", ""),
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": str(message)}]}
            ],
            "tool_choice": {"type": "auto"},
            "experimental": {},
            "tools": self.settings.get("tools", []),
            "tool_resources": filtered_tool_resources,
        }

    def _filter_tool_resources(self, tool_name, tool_type):
        """Filter tool resources based on tool type.

        Args:
            tool_name (str): Name of the tool
            tool_type (str): Type of the tool

        Returns:
            dict: Filtered tool resources
        """
        resources = self.settings["tool_resources"][tool_name]
        if tool_type == "cortex_analyst_text_to_sql":
            return {"semantic_model_file": resources.get("semantic_model_file", "")}
        elif tool_type == "cortex_search":
            return {
                "name": resources.get("name", ""),
                "max_results": resources.get("max_results", 1),
                # "id_column": "DOCUMENT",
                # "title_column": "TEXT",
                "database": resources.get("database", ""),
                "schema": resources.get("schema", ""),
            }
        return {}

    def _prepare_headers(self, jwt_token):
        """Prepare HTTP headers for API requests.

        Args:
            jwt_token (str): JWT token for authentication

        Returns:
            dict: HTTP headers
        """
        return {
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {jwt_token}",
        }

    def _process_stream_response(self, response):
        """Process streaming response from the API.

        Args:
            response: HTTP response object

        Returns:
            tuple: (response text, generated SQL)
        """
        text = ""
        sql = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                print("decoded_line: ", decoded_line)
                if decoded_line.startswith("data: ") and decoded_line[6:] != "[DONE]":
                    try:
                        json_data = json.loads(decoded_line[6:])
                        if "delta" in json_data and "content" in json_data["delta"]:
                            for content_item in json_data["delta"]["content"]:
                                if content_item["type"] == "text":
                                    text += content_item.get("text", "")
                                elif content_item["type"] == "tool_results":
                                    tool_results = content_item.get("tool_results", {})
                                    if "content" in tool_results:
                                        for result in tool_results["content"]:
                                            if result.get("type") == "json":
                                                text += result.get("json", {}).get(
                                                    "text", ""
                                                )
                                                sql = result.get("json", {}).get(
                                                    "sql", ""
                                                )
                    except json.JSONDecodeError:
                        text += decoded_line[6:]  # Fallback to raw data
        return text, sql

    def _process_server_stream_response(self, response):
        """Process server-side streaming response from the Snowflake API.

        Args:
            response: Response object from _snowflake.send_snow_api_request

        Returns:
            tuple: (response text, generated SQL) or (None, None) on error
        """
        text = ""
        sql = ""

        if not response:
            st.error("No response received from the server.")
            return None, None

        # Check for HTTP status
        if response.get("status") != 200:
            st.error(
                f"❌ HTTP Error: {response.get('status')} - {response.get('reason', 'Unknown reason')}"
            )
            st.error(f"Response details: {response}")
            return None, None

        # Parse the response content
        try:
            response_content = json.loads(response["content"])
        except json.JSONDecodeError:
            st.error(
                "❌ Failed to parse API response. The server may have returned an invalid JSON format."
            )
            st.error(f"Raw response: {response['content'][:200]}...")
            return None, None

        # Process the SSE-like response content
        try:
            for event in response_content:
                if event.get("event") == "message.delta":
                    data = event.get("data", {})
                    delta = data.get("delta", {})

                    for content_item in delta.get("content", []):
                        content_type = content_item.get("type")
                        if content_type == "text":
                            text += content_item.get("text", "")
                        elif content_type == "tool_results":
                            tool_results = content_item.get("tool_results", {})
                            if "content" in tool_results:
                                for result in tool_results["content"]:
                                    if result.get("type") == "json":
                                        text += result.get("json", {}).get("text", "")
                                        sql = result.get("json", {}).get("sql", "")
        except Exception as e:
            st.error(f"Error processing server stream response: {str(e)}")
            return None, None

        return text, sql

    def edit(self, session, name=None, settings=None):
        """Update agent properties.

        Args:
            session: Snowflake session object
            name (str, optional): New name for the agent
            settings (dict, optional): New settings to update

        Returns:
            CortexAgent: Self reference for method chaining
        """
        if name:
            self.name = name
        if settings:
            self.settings.update(settings)
        self.updated_at = datetime.now()
        return self

    @classmethod
    def from_dict(cls, data):
        """Create an agent instance from dictionary data.

        Args:
            data (dict): Agent data from database

        Returns:
            CortexAgent: New agent instance
        """
        agent = cls(
            name=data["NAME"], settings=json.loads(data["SETTINGS"], strict=False)
        )
        agent.created_at = (
            data["CREATED_AT"]
            if isinstance(data["CREATED_AT"], datetime)
            else datetime.strptime(data["CREATED_AT"], "%Y-%m-%d %H:%M:%S")
        )
        agent.updated_at = (
            data["UPDATED_AT"]
            if isinstance(data["UPDATED_AT"], datetime)
            else datetime.strptime(data["UPDATED_AT"], "%Y-%m-%d %H:%M:%S")
        )
        return agent


class CortexAgentManager:
    """Manager class for handling Cortex agent persistence."""

    def __init__(self, session):
        """Initialize the agent manager.

        Args:
            session: Snowflake session object
        """
        self.session = session
        self.table_name = "cortex_agents"
        self.ensure_table_exists()

    def ensure_table_exists(self):
        """Create the agents table in Snowflake if it doesn't exist."""
        try:
            db = snowflake_config.get("database")
            self.session.sql(f"""
                CREATE TABLE IF NOT EXISTS {db}.PUBLIC.{self.table_name} (
                    name VARCHAR,
                    settings VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (name)
                )
            """).collect()
        except Exception as e:
            st.error(f"Error creating table {self.table_name}: {str(e)}")

    def save_agent(self, agent):
        """Save or update an agent in the database.

        Args:
            agent (CortexAgent): Agent to save

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            agent_data = agent.to_dict()
            # Escape single quotes in all string fields to prevent SQL injection
            safe_name = agent_data["name"].replace("'", "''")
            safe_settings = agent_data["settings"].replace("'", "''")
            safe_created_at = str(agent_data["created_at"]).replace("'", "''")
            safe_updated_at = str(agent_data["updated_at"]).replace("'", "''")

            query = f"""
                MERGE INTO {self.table_name} AS target
                USING (SELECT 
                    '{safe_name}' as name,
                    '{safe_settings}' as settings,
                    '{safe_created_at}' as created_at,
                    '{safe_updated_at}' as updated_at
                ) AS source
                ON target.name = source.name
                WHEN MATCHED THEN
                    UPDATE SET 
                        name = source.name,
                        settings = source.settings,
                        updated_at = source.updated_at
                WHEN NOT MATCHED THEN
                    INSERT (name, settings, created_at, updated_at)
                    VALUES (source.name, source.settings, source.created_at, source.updated_at)
            """
            self.session.sql(query).collect()
            return True
        except Exception as e:
            st.error(f"Error saving agent: {str(e)}")
            return False

    def get_agent(self, name):
        """Retrieve an agent by name from the database.

        Args:
            name (str): Agent name

        Returns:
            CortexAgent: Agent instance or None if not found
        """
        try:
            # Escape single quotes in name to prevent SQL injection
            safe_name = name.replace("'", "''")
            query = f"SELECT * FROM {self.table_name} WHERE NAME = '{safe_name}'"
            result = self.session.sql(query).collect()
            return CortexAgent.from_dict(result[0]) if result else None
        except Exception as e:
            st.error(f"Error getting agent: {str(e)}")
            return None

    def get_all_agents(self):
        """Retrieve all agents from the database.

        Returns:
            list: List of CortexAgent instances
        """
        try:
            query = f"SELECT * FROM {self.table_name}"
            result = self.session.sql(query).collect()
            return [CortexAgent.from_dict(row) for row in result] if result else []
        except Exception as e:
            st.error(f"Error retrieving agents: {str(e)}")
            return []


def render_dynamic_settings_form(
    session,
    settings_template: Dict[str, Any],
    prefix: str = "",
    prefilled_values: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Render and process a dynamic form for agent settings.

    Args:
        session: Snowflake session object
        settings_template (dict): Template for settings structure
        prefix (str): Prefix for UI element keys
        prefilled_values (dict, optional): Pre-existing values to populate

    Returns:
        dict: Updated settings dictionary
    """
    if prefilled_values is None:
        prefilled_values = settings_template.copy()
    filled_settings = settings_template.copy()

    databases = list_databases(session)  # Get available databases

    # Initialize session state for tools if not present
    if f"{prefix}_tools_list" not in st.session_state:
        st.session_state[f"{prefix}_tools_list"] = (
            prefilled_values["tools"].copy() if prefilled_values["tools"] else []
        )
    if f"{prefix}_tool_resources" not in st.session_state:
        st.session_state[f"{prefix}_tool_resources"] = (
            prefilled_values["tool_resources"].copy()
            if prefilled_values["tool_resources"]
            else {}
        )

    def update_tool_resources(tool_name, tool_type, tool_resources, i):
        """Update tool resources based on type and user input.

        Args:
            tool_name (str): Name of the tool
            tool_type (str): Type of the tool
            tool_resources (dict): Current tool resources
            i (int): Index for unique key generation
        """
        if tool_name not in tool_resources:
            tool_resources[tool_name] = (
                {"semantic_model_file": ""}
                if tool_type == "cortex_analyst_text_to_sql"
                else {"name": "", "max_results": 1, "database": "", "schema": ""}
            )

        if tool_type == "cortex_analyst_text_to_sql":
            col1, col2 = st.columns(2)
            with col1:
                database = st.selectbox(
                    f"Database",
                    databases,
                    index=databases.index(tool_resources[tool_name].get("database", ""))
                    if tool_resources[tool_name].get("database", "") in databases
                    else 0,
                    key=f"{prefix.lower()}database_{i}",
                )
                tool_resources[tool_name]["database"] = database

            with col2:
                if database:
                    schemas = list_schemas(session, database)
                    schema = st.selectbox(
                        f"Schema",
                        schemas,
                        index=schemas.index(tool_resources[tool_name].get("schema", ""))
                        if tool_resources[tool_name].get("schema", "") in schemas
                        else 0,
                        key=f"{prefix.lower()}schema_{i}",
                    )
                    tool_resources[tool_name]["schema"] = schema

            with col1:
                if database and schema:
                    stages = list_stages(session, database, schema)
                    stage = st.selectbox(
                        f"Stage",
                        stages,
                        index=stages.index(tool_resources[tool_name].get("stage", ""))
                        if tool_resources[tool_name].get("stage", "") in stages
                        else 0,
                        key=f"{prefix.lower()}stage_{i}",
                    )
                    tool_resources[tool_name]["stage"] = stage

            with col2:
                if database and schema and stage:
                    semantic_models = list_files_in_stage(
                        session, database, schema, stage
                    )
                    semantic_models = [
                        model for model in semantic_models if model.endswith(".yaml")
                    ]
                    semantic_models = [
                        model.split("/")[-1] for model in semantic_models
                    ]
                    semantic_model_file = st.selectbox(
                        f"Semantic Model File",
                        semantic_models,
                        index=semantic_models.index(
                            tool_resources[tool_name].get("semantic_model_file", "")
                        )
                        if tool_resources[tool_name].get("semantic_model_file", "")
                        in semantic_models
                        else 0,
                        key=f"{prefix.lower()}semantic_model_{i}",
                    )

            if database and schema and stage and semantic_model_file:
                model_file = f"@{database}.{schema}.{stage}/{semantic_model_file}"
                tool_resources[tool_name]["semantic_model_file"] = model_file

        elif tool_type == "cortex_search":
            col3, col4 = st.columns(2)
            with col3:
                database = st.selectbox(
                    f"Database",
                    databases,
                    index=databases.index(tool_resources[tool_name].get("database", ""))
                    if tool_resources[tool_name].get("database", "") in databases
                    else 0,
                    key=f"{prefix.lower()}database_{i}",
                )
                tool_resources[tool_name]["database"] = database

            with col4:
                if database:
                    schemas = list_schemas(session, database)
                    schema = st.selectbox(
                        f"Schema",
                        schemas,
                        index=schemas.index(tool_resources[tool_name].get("schema", ""))
                        if tool_resources[tool_name].get("schema", "") in schemas
                        else 0,
                        key=f"{prefix.lower()}schema_{i}",
                    )
                    tool_resources[tool_name]["schema"] = schema

            with col3:
                if database and schema:
                    cortex_search_services = (
                        list_cortex_services(session, database, schema) or []
                    )
                    name = st.selectbox(
                        f"Search Service Name",
                        cortex_search_services,
                        index=cortex_search_services.index(
                            tool_resources[tool_name]["name"].split(".")[-1]
                        )
                        if tool_resources[tool_name]["name"]
                        and tool_resources[tool_name]["name"].split(".")[-1]
                        in cortex_search_services
                        else 0,
                        key=f"{prefix.lower()}service_name_{i}",
                    )
                    tool_resources[tool_name]["name"] = f"{database}.{schema}.{name}"

            with col4:
                if database and schema and name:
                    tool_resources[tool_name]["max_results"] = st.number_input(
                        f"Max Search Results",
                        min_value=1,
                        max_value=100,
                        value=tool_resources[tool_name].get("max_results", 1),
                        key=f"{prefix.lower()}max_results_{i}",
                    )

            with col3:
                if database and schema and name:
                    filter_operator = st.selectbox(
                        "Filter Operator",
                        ["@eq", "@contains", "@gte", "@lte"],
                        key=f"{prefix.lower()}filter_operator_{i}",
                    )
                    tool_resources[tool_name]["filter_operator"] = filter_operator

            with col4:
                if database and schema and name and filter_operator:
                    filter_value = st.text_input(
                        "Filter Value", key=f"{prefix.lower()}filter_value_{i}"
                    )
                    tool_resources[tool_name]["filter_value"] = filter_value

    st.subheader(f"Tools and Resources")
    with st.expander("", expanded=True):
        # Manage tools
        tools_list = st.session_state[f"{prefix}_tools_list"]
        tool_resources = st.session_state[f"{prefix}_tool_resources"]

        for i, tool in enumerate(tools_list):
            with st.container():
                tool_name = tool["tool_spec"]["name"]
                tool_type = tool["tool_spec"]["type"]

                col1, col2 = st.columns(2)
                with col1:
                    new_tool_name = st.text_input(
                        f"Tool Name",
                        value=tool_name,
                        key=f"{prefix.lower()}tool_name_{i}",
                    )
                with col2:
                    new_tool_type = st.selectbox(
                        f"Tool Type",
                        ["cortex_search", "cortex_analyst_text_to_sql"],
                        index=0 if tool_type == "cortex_search" else 1,
                        key=f"{prefix.lower()}tool_type_{i}",
                    )

                tools_list[i] = {
                    "tool_spec": {"type": new_tool_type, "name": new_tool_name}
                }
                update_tool_resources(new_tool_name, new_tool_type, tool_resources, i)

                if st.button(f"Remove", key=f"{prefix.lower()}remove_tool_{i}"):
                    del tools_list[i]
                    if new_tool_name in tool_resources:
                        del tool_resources[new_tool_name]
                    st.rerun()

                st.divider()

        # Response instruction input
        filled_settings["response_instruction"] = st.text_area(
            f"Response Instruction",
            value=prefilled_values["response_instruction"],
            key=f"{prefix.lower()}response_instruction",
        )

        if st.button(f"Add", key=f"{prefix.lower()}add_tool"):
            new_tool_name = f"tool_{len(tools_list) + 1}"
            tools_list.append(
                {"tool_spec": {"type": "cortex_search", "name": new_tool_name}}
            )
            st.rerun()

        # Update settings with tools data
        filled_settings["tools"] = tools_list
        filled_settings["tool_resources"] = tool_resources
        st.session_state[f"{prefix}_tools_list"] = tools_list
        st.session_state[f"{prefix}_tool_resources"] = tool_resources

    return filled_settings


def display_cortex_agent(session):
    """Main function to display the Cortex Agent interface.

    Args:
        session: Snowflake session object
    """
    st.title("Cortex Agent")
    agent_manager = CortexAgentManager(session)

    option = st.selectbox(
        "Choose Functionality", ["Create", "Edit"], key="agent_option"
    )

    if option == "Create":
        # Create Agent Tab
        st.subheader("Create Agent")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Agent Name", key="create_agent_name")
        with col2:
            model = st.selectbox(
                "Model",
                ["llama3.1-70b", "llama3.3-70b", "mistral-large2", "claude-3-5-sonnet"],
                key="create_model_select",
            )

        new_settings = render_dynamic_settings_form(
            session, SETTINGS_TEMPLATE, prefix="Create"
        )
        new_settings["model"] = model

        if st.button("Create", key="create_agent"):
            if name.strip():
                # Generate starter questions based on tools
                question_generator = StarterQuestionGenerator(session)
                starter_questions = question_generator.generate_starter_questions(
                    name,
                    new_settings.get("tools", []),
                    new_settings.get("tool_resources", {}),
                )
                new_settings["starter_questions"] = starter_questions

                new_agent = CortexAgent(name=name, settings=new_settings)
                new_settings["response_instruction"] = (
                    new_settings.get("response_instruction", "")
                    + "Use that information to provide a summary that addresses the user's question. Question: {{.Question}}\n\nContext: {{.Context}}"
                )
                response_instruction = new_settings.get("response_instruction", "")
                if (
                    "{{.Question}}" not in response_instruction
                    or "{{.Context}}" not in response_instruction
                ):
                    show_toast_message(
                        "Response instruction must include {{.Question}} and {{.Context}}",
                        toast_type="error",
                    )
                    return

                if agent_manager.save_agent(new_agent):
                    st.success(
                        f"Agent {name} created successfully with {len(starter_questions)} starter questions"
                    )
                    # Clean up session state
                    for key in list(st.session_state.keys()):
                        if key.startswith("Create"):
                            del st.session_state[key]
                else:
                    st.error(f"Error creating agent {name}")
            else:
                st.error("Agent name cannot be empty")

    elif option == "Edit":
        # Edit Agent Tab
        st.subheader("Edit Existing Agent")
        agents = agent_manager.get_all_agents()
        agent_names = [agent.name for agent in agents]

        col1, col2 = st.columns(2)
        with col1:
            selected_agent = st.selectbox(
                "Select Agent to Edit", agent_names, key="edit_select"
            )

        if selected_agent:
            agent = next(a for a in agents if a.name == selected_agent)
            new_name = agent.name
            with col2:
                new_model = st.selectbox(
                    "Model",
                    [
                        "llama3.1-70b",
                        "llama3.3-70b",
                        "mistral-large2",
                        "claude-3-5-sonnet",
                    ],
                    index=[
                        "llama3.1-70b",
                        "llama3.3-70b",
                        "mistral-large2",
                        "claude-3-5-sonnet",
                    ].index(agent.settings["model"])
                    if agent.settings["model"]
                    in [
                        "llama3.1-70b",
                        "llama3.3-70b",
                        "mistral-large2",
                        "claude-3-5-sonnet",
                    ]
                    else 0,
                    key="edit_model_select",
                )

            # Initialize session state when agent changes
            if st.session_state.get("edit_selected_agent") != selected_agent:
                st.session_state["Edit_tools_list"] = agent.settings["tools"].copy()
                st.session_state["Edit_tool_resources"] = agent.settings[
                    "tool_resources"
                ].copy()
                st.session_state["edit_selected_agent"] = selected_agent

            # Render the rest of the settings form below the columns
            updated_settings = render_dynamic_settings_form(
                session,
                SETTINGS_TEMPLATE,
                prefix="Edit",
                prefilled_values=agent.settings,
            )
            # Override the model and jwt_token from column inputs
            updated_settings["model"] = new_model

            # print("settings: " , json.dumps(updated_settings, indent=2))

            if st.button("Update", key="edit_button"):
                if new_name.strip():
                    # Regenerate starter questions if tools have changed
                    question_generator = StarterQuestionGenerator(session)
                    starter_questions = question_generator.generate_starter_questions(
                        new_name,
                        updated_settings.get("tools", []),
                        updated_settings.get("tool_resources", {}),
                    )
                    updated_settings["starter_questions"] = starter_questions

                    updated_agent = agent.edit(session, new_name, updated_settings)
                    if agent_manager.save_agent(updated_agent):
                        st.success(
                            f"Agent '{new_name}' updated successfully with {len(starter_questions)} starter questions!"
                        )
                        # Clean up session state
                        for key in list(st.session_state.keys()):
                            if key.startswith("Edit") or key == "edit_selected_agent":
                                del st.session_state[key]
                    else:
                        st.error("Failed to update agent")
                else:
                    st.error(
                        "Agent name and RSA public key fingerprint cannot be empty"
                    )
