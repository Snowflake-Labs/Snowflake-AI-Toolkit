from src.settings import *
import json
from pathlib import Path

config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

default_params = {
    "Complete": {"prompt": "What is RAG is all about and explain it like a 5 year old"},
    "Complete Multimodal": {
        "database": config["demo_database"],
        "schema": config["demo_schema"],
        "stage": config["my_images"],
        "image": "image.jpg",
        "prompt": "Explain the image in detail and provide a summary of its content",
    },
    "Translate": {
        "text": "Good morning! It’s a beautiful day outside, and I’m planning to go for a walk in the park."
    },
    "Summarize": {
        "text": "The city recently introduced a new bike-sharing program that allows residents to rent bicycles using a mobile app. The initiative aims to reduce traffic congestion, improve air quality, and promote a healthier lifestyle for citizens."
    },
    "Extract": {
        "text": "The report highlighted a 10% increase in renewable energy usage last year. It also mentioned the introduction of stricter emission policies and incentives for electric vehicle adoption.",
        "query": "whats the percent inrease?",
    },
    "Sentiment": {
        "text": "The hotel room was spacious and clean, the staff was friendly, but the check-in process was slow and frustrating.",
        "entities": "Room, Staff, Check-in",
    },
    "Classify Text": {
        "text": "The new smartphone includes a high-resolution display, longer battery life, and advanced photo editing features.",
        "categories": "Technology, Sports, Travel, Food",
    },
    "Parse Document": {
        "database": config["demo_database"],
        "schema": config["demo_schema"],
        "stage": config["contracts"],
        "file": "file.pdf",
    },
    "AI Parse Document": {
        "database": config["demo_database"],
        "schema": config["demo_schema"],
        "stage": config["contracts"],
        "file": "nsa.pdf",
    },
    "AI Transcribe": {
        "database": config["demo_database"],
        "schema": config["demo_schema"],
        "stage": config["my_audios"],
        "file": "demo.mp3",
    },
    "AI Complete": {
        "Text": {"prompt": "What is RAG is all about and explain it like a 5 year old"},
        "Image": {
            "database": config["demo_database"],
            "schema": config["demo_schema"],
            "stage": config["my_images"],
            "image": "tiger_in_forest.jpg",
            "prompt": "Explain the image in detail and provide a summary of its content",
        },
        "Prompt Object": {
            "database": config["demo_database"],
            "schema": config["demo_schema"],
            "stage": config["my_images"],
            "prompt_column": "prompt",
            "multi_images": ["tiger_in_forest.jpg", "image_of_pizza.png"],
            "prompt_text": "Are both image {0} and image {1} pictures of cats?",
        },
    },
    "AI Similarity": {
        "Text": {
            "first text": "Explain photosynthesis in a simple way that a child could understand.",
            "second text": "How do plants make food using sunlight? Explain it like I am five years old.",
        },
        "Image": {
            "database": config["demo_database"],
            "schema": config["demo_schema"],
            "stage": config["my_images"],
            "first_image": "image1.jpg",
            "second_image": "image2.jpg",
        },
    },
    "AI Classify": {
        "Text": {
            "text": "Eating fresh fruits and vegetables daily contributes to a healthy lifestyle, while consuming excessive sugary drinks can be unhealthy.",
            "categories": "healthy, unhealthy",
        },
        "Image": {
            "database": config["demo_database"],
            "schema": config["demo_schema"],
            "stage": config["my_images"],
            "image": "image_of_pizza.png",
            "prompt": "classify the food in the image as healthy or unhealthy",
            "categories": "healthy, unhealthy",
        },
        # note - the defaults are not yet taken
    },
    "AI Filter": {
        "Text": {"first text": "Is Paris the capital of France?"},
        "Image": {
            "database": config["demo_database"],
            "schema": config["demo_schema"],
            "stage": config["my_images"],
            "first_image": "tiger_in_forest.jpg",
            "prompt": "is tiger in the image?",
        },
        # note - by default image should be selected
    },
    "AI Agg": {
        "Text": {
            "expression": '["The phone battery lasts all day and charges quickly.", "The camera quality is average and struggles in low light.", "The design feels premium and lightweight.", "The screen sometimes freezes while using multiple apps."]',
            "prompt": "Summarize the product reviews into a short statement for consumers",
        },
        "Table": {
            "database": config["demo_database"],
            "schema": config["demo_schema"],
            "table": config["reason_view"],
            "text_column": "R_REASON_DESC",
            "group_by": "None",
            "prompt_text": "Summarize the cancel reasons given by customers",
        },
    },
    "AI Summarize Agg": {
        "Text": {
            "expression": "The vacation was wonderful. I enjoyed the beach and local food, though the hotel service could have been better."
        },
        "Table": {
            "database": config["demo_database"],
            "schema": config["demo_schema"],
            "table": config["reason_view"],
            "text_column": "R_REASON_DESC",
            "group_by": "None",
        },
    },
    "AI Translate": {
        "text": "Good morning! It’s a beautiful day outside, and I’m planning to go for a walk in the park."
    },
    "RAG": {
        "database": config["demo_database"],
        "schema": config["demo_schema"],
        "table": "REPAIR_MANUAL_EMBEDDINGS_Demo",
        "vector_column": "EMBEDDING",
        "text_column": "CONTENT",
        "embeddings": "EMBED_TEXT_1024",
        "embedding_model": "snowflake-arctic-embed-l-v2.0",
        "prompt": "Explain the concept of RAG in detail and how it is used in AI applications",
    },
    "Search": {
        "database": config["demo_database"],
        "schema": config["demo_schema"],
        "service": "MEDNOTES_SEARCH_SERVICE_Demo",
        "display_column": "TEXT",
    },
}
