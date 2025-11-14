import streamlit as st
from src.utils import *
from src.cortex_functions import *
from src.query_result_builder import *
from src.notification import *
import asyncio
import threading
import json
from snowflake.core import Root
import os

config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

def display_search(session):
    
    st.title("Cortex Search")
    st.subheader("Use Your Tables As Context To Search")

    # Display "Create or Use Knowledge Source" dropdown
    create_or_use = st.selectbox("Select Action", ("Create","Display"), key="create_or_use")

    # Display "Create Cortex Search Service" form
    # if st.button("show cortex search service"):
    #     st.write("Cortex Search Service")
    #     q = f"SHOW CORTEX SEARCH SERVICES"
    #     res = session.sql(q).collect()
    #     st.write(res)


    warehouse = config["snowflake_connection"]["warehouse"]

    if create_or_use == "Create":

        service_name = st.text_input("Enter Service Name", key="service_name")
        # Row 1: Database and Schema Selection
        col1, col2 = st.columns(2)
        with col1:
            selected_db = st.selectbox("Database", list_databases(session))
        with col2:
            selected_schema = st.selectbox("Schema", list_schemas(session, selected_db))

        # Row 2: Stage Selection and File Upload
        col1, col2 = st.columns(2)
        required_columns = []
        with col1:
            tables = list_tables(session, selected_db, selected_schema)
            selected_table = st.selectbox("Table", tables or [])
        with col2:
            if selected_table:
                required_columns = list_columns(session, selected_db, selected_schema, selected_table)
                selected_column = st.selectbox("Search Column", required_columns or [])      
        
        attributes_cols = [col for col in required_columns if col != selected_column]

        # Embedding Options
        col1, col2 = st.columns(2)
        with col1:
            selected_attributes = st.multiselect("Attributes", attributes_cols or [])
        with col2:
            embedding_model = st.selectbox("Model", config["default_settings"]["embeddings"]["CORTEX_SUPPORTED"])

        # Create Cortex Search Service
        if st.button("Create"):
            # Add notification for process tracking
            details = f"Creating cortex search service in table {selected_table} with the selected columns."
            print("coming to notification")
            notification_id = add_notification_entry(session, "Create Cortex Search Service", "In-Progress", details)
            print("added to notification")
            try:
                # Trigger async cortex search service creation
                trigger_async_search_process(
                    session, selected_db, selected_schema, selected_table, selected_column, selected_attributes,service_name, embedding_model, warehouse, notification_id
                )
                st.success("Cortex search service creation initiated. Check notifications for updates.")
            except Exception as e:
                # Update notification to Failed and log the error
                update_notification_entry(session, notification_id, "Failed")
                add_log_entry(session, "Create Cortex Search Service", str(e))
                st.error(f"Failed to initiate cortex search service creation: {e}")
        
    elif create_or_use == "Use":
        
            # for name in cortex_services:
            #     q = f"DROP CORTEX SEARCH SERVICE {name.lower()}"
            #     session.sql(q).collect()
            # print(cortex_services)
            st.subheader("Choose Your Search Service")
            col1, col2 = st.columns(2)
            with col1:
                selected_db = st.selectbox("Database", list_databases(session))
            with col2:
                selected_schema = st.selectbox("Schema", list_schemas(session, selected_db))

            col1, col2 = st.columns(2)
            with col1:
                cortex_services = list_cortex_services(session,selected_db,selected_schema)
                selected_service = st.selectbox("Service", cortex_services or [])

            with col2:
                data = fetch_cortex_service(session, selected_service,selected_db,selected_schema)
                row = data[0]
                cols = row.columns.split(",")
                attributes = row.attribute_columns.split(",")
                
                columns = st.multiselect("Display Columns", cols)
            
            st.subheader("Create Filter & Limits")
            col3, col4 = st.columns(2)
            with col3:
                filter_column = st.selectbox("Filter Columns", attributes)
            with col4:
                filter_operator = st.selectbox("Filter Operator", ["@eq", "@contains", "@gte", "@lte"])
            filter_value = st.text_input(f"Enter value for {filter_operator} on {filter_column}")

            if filter_column and filter_operator and filter_value:
                if filter_operator == "@eq":
                    filter = { "@eq": { filter_column: filter_value } }
                elif filter_operator == "@contains":
                    filter = { "@contains": { filter_column: filter_value} }
                elif filter_operator == "@gte":
                    filter = { "@gte": { filter_column: filter_value } }
                elif filter_operator == "@lte":
                    filter = { "@lte": { filter_column: filter_value } }
                
                st.write(f"Generated Filter: {filter}")

            else:
                filter = {}
            limit = st.slider("Limit Results", min_value=1, max_value=10, value=1)

            st.subheader("Choose Your Model")
            col5, col6 = st.columns(2)
            with col5:
                model_type = st.selectbox("Model Type", ["Base","Fine Tuned", "Private Preview"])
            
            with col6:
                if model_type == "Base":
                    selected_model = st.selectbox("Model", config["default_settings"]["model"])
                elif model_type == "Private Preview":
                    selected_model = st.selectbox("Model", config["default_settings"]["private_preview_models"])
                else:
                    fine_tuned_models = fetch_fine_tuned_models(session)
                    print(fine_tuned_models)
                    selected_model = st.selectbox("Model", fine_tuned_models)
        
            question = st.text_input("Enter Question", key="question")

            if st.button("Search"):
                try:
                    root = Root(session)
                    service = (root
                            .databases[selected_db]
                            .schemas[selected_schema]
                            .cortex_search_services[selected_service.lower()]
                            )
                    print("service: ",service)
                    print("query: ",question)
                    print("columns: ",columns)
                    if not columns:
                        show_toast_message("Please select columns to display.")   
                        return
                    columns = [col.lower() for col in columns]
                    # print("filter: ",filter)
                    print("limit: ",limit)
                    resp = service.search(
                        query=question,
                        columns=columns,
                        filter=filter,  # Use the dynamically generated filter
                        limit=int(limit)
                    )
                    # print("resp: ",resp)
                    
                    import pandas as pd
                    if isinstance(resp.results, list) and len(resp.results) > 0:
                        df = pd.DataFrame(resp.results)
                        st.dataframe(df)  # Display as a table
                    else:
                        st.write("No results found.")
                except Exception as e:
                    add_log_entry(session, "Generate Search Response", str(e))
                    print(e)
                    st.error("An error occurred. Please check logs for details.")

    elif create_or_use == "Display":
        col1, col2,col3 = st.columns(3)
        with col1:
            selected_db = st.selectbox("Database", list_databases(session))
        with col2:
            selected_schema = st.selectbox("Schema", list_schemas(session, selected_db))
        with col3:  
            selected_service = st.selectbox("Service", list_cortex_services(session, selected_db, selected_schema))
            

        if st.button("Display"):
            data = cortex_search_service_display(session, selected_db,selected_schema,selected_service)
            st.write(data)
        

def trigger_async_search_process(session, database, schema, table, column, attributes, service_name, embedding_model, warehouse, notification_id):
    """
    Triggers the asynchronous process to create a Cortex Search Service.

    Args:
        session (snowflake.connector.connection.SnowflakeConnection): Active Snowflake session.
        database (str): The database containing the table.
        schema (str): The schema containing the table.
        table (str): The table to be used for the Cortex Search Service.
        columns (list[str]): The column to be used for the Cortex Search Service.
        embedding_model (str): The embedding model to use for the process.
        warehouse (str): The Snowflake warehouse to use for the process.
        notification_id (int): The ID of the notification entry to update.
    """
    async def async_rag_process():
        try:
            # Simulate asynchronous processing
            await asyncio.sleep(1)
            
            # Create the Cortex Search Service
            create_cortex_search_service(session, database, schema, table, column, attributes,service_name,embedding_model, warehouse)
            
            # Update notification status to Success
            update_notification_entry(session, notification_id, "Success")
            st.success(f"Cortex Search Service created successfully. Check the notification screen for details.")
        except Exception as e:
            # Log the error and update notification status to Failed
            update_notification_entry(session, notification_id, "Failed")
            add_log_entry(session, "Create Cortex Search Service", str(e))
            st.error(f"An error occurred: {e}")
            raise e

    # Trigger async process using threading
    thread = threading.Thread(target=asyncio.run, args=(async_rag_process(),))
    thread.start()