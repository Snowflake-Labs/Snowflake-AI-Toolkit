# Snowflake AI Toolkit

Snowflake AI Toolkit is an AI Accelerator and Playground for enabling AI in Snowflake. It is a Plug and Play Streamlit-based Native App that lets you explore, learn, and rapidly prototype AI solutions powered by Snowflake's Cortex and AI Functions. In addition to these capabilities, the toolkit automatically sets up your environment and provides prebuilt use cases to try all Cortex features—from simple prompting to advanced agentic solutions.

# Features

### Playground

An interactive environment where users can chat and experiment with Snowflake Cortex functions, test prompts, and explore Cortex capabilities. Supports both manual text entry and voice-to-text input for prompts across multiple functions—including Complete, Translate, Extract, Sentiment Analysis, and AI Complete (debug mode only). Additionally, audio output is available for responses, enabling users to listen to results as well as view them.

### Build

A dedicated section for constructing and deploying data pipelines and workflows using Snowflake Cortex's powerful AI capabilities, enabling seamless integration with your Snowflake databases and tables

- Text completion and generation using the COMPLETE function
- Retrieval-Augmented Generation (RAG) for question answering with your own data
- Fine-tuning large language models on your custom datasets

### Search

Get a hybrid (vector and keyword) search engine on your text data in minutes,

- Create and manage Cortex Search Service
- Use Cortex Search for your RAG Application
- Use Cortex Search Powered Chat

### Agent

Cortex Agents orchestrate across both structured and unstructured data sources to deliver insights.

- Create and Manage Cortex Agent Instances
- Test the Agent against the Data
- Utilize Cortex Search and Cortex Analyst through Agents

### Audio Input Integration

The Snowflake AI Toolkit now supports voice input across multiple functionalities, making it easier to interact with AI functions through speech (available in debug mode only):

- **Voice-to-Text Conversion**: Convert spoken words directly into text inputs for various AI functions
- **Multi-Function Support**: Available for Complete, Complete Multimodal, Translate, Extract, Sentiment Analysis, AI Classify (Image), and AI Complete functions
- **Real-Time Processing**: Instant speech recognition with immediate text population in input fields
- **User-Friendly Interface**: Microphone button (🎙️) positioned conveniently next to the Run button for seamless workflow
- **Debug Mode Only**: Audio input feature is enabled only when `mode` is set to `"debug"` in `src/settings_config.json`

### Notification

- **Real-Time Updates**: Receive instant notifications for every function call and completion.
- **Synchronous & Asynchronous Support**: Stay informed about both immediate and long-running processes, including RAG and fine-tuning tasks.
- **Computation Details**: View token usage and token credit consumption for each operation.
- **In-App Delivery**: All notifications appear directly within the application interface.
- **Enhanced Transparency**: Get clear, timely feedback throughout all workflows.


## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Snowflake Account**: Ensure you have an active Snowflake account in a region where Cortex functionalities are supported. For detailed information, refer to the [Availability Region documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability).
- **Private Preview Access**: To try private preview models in Snowflake, you need to contact your Snowflake account team to request access to the private preview feature.
- **Role Permissions**: ACCOUNTADMIN or equivalent role with permissions to create:
  - Stages
  - Databases
  - Other resources
- **Python**: Version 3.7 or higher.
- **Streamlit**: Installed on your local machine.
- **Snowflake-CLI** : Install snowflake-cli through pip
- **Model Availability**: `claude-4-sonnet` and `claude-4-opus` is available only in AWS US (cross-region).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sgsshankar/Snowflake-AI-Toolkit.git
   cd snowflake-ai-toolkit
   ```

2. Install dependencies (only required for local development):

   ```bash
   pip install -r requirements.txt
   ```

   All necessary libraries are listed in `requirements.txt`.

## Setup

The application will automatically handle:

- Database creation
- Stage setup
- Required resource initialization

Make sure your Snowflake role has the necessary privileges for these operations.

To configure the application mode, update the `mode` parameter in `src/settings_config.json`:

- Set to `"debug"` for local development and editing.
- Set to `"native"` for running natively in Snowflake.

### Authentication for Cortex Agent – JWT Token Support Only

This app currently supports **only JWT-based authentication** using RSA key pairs for Cortex Agents. OAuth is **not supported** at this time.

#### Step 1: Generate RSA Key Pair

```bash
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_private_key.p8 -nocrypt
openssl rsa -in rsa_private_key.p8 -pubout -out rsa_public_key.pub
```

#### Step 2: Upload Public Key to Snowflake

1. Copy the content from `rsa_public_key.pub` (exclude `-----BEGIN PUBLIC KEY-----` headers).
2. In Snowflake, run:

```sql
alter user <your_username> set rsa_public_key='<paste_public_key_here>';
```

3. Retrieve the public key fingerprint:

```sql
desc user <your_username>;
```

Use the fingerprint in `rsa_public_key_fp`.

#### Step 3: Add to `settings_config.json`

- Paste full contents of `rsa_private_key.p8` into `rsa_private_key` (add \n).
- Paste fingerprint into `rsa_public_key_fp`.

For more details, refer to the [Snowflake documentation](https://docs.snowflake.com/en/user-guide/key-pair-auth.html#using-key-pair-authentication).

## Running the App

### Locally

Launch the application locally using:

```bash
streamlit run streamlit_app.py
```

### Natively on Snowflake

To deploy the application natively in Snowflake, use the following command:

```bash
snow streamlit deploy --account "<your_account>" --user "<your_username>" --password "<your_password>" --role "<your_role>" --warehouse "<your_warehouse>" --database "<your_database>" --replace
```

Replace the placeholders (`<your_account>`, `<your_username>`, etc.) with your actual Snowflake account details. When running natively in Snowflake, installing dependencies from `requirements.txt` is not needed.

## Project Structure

| File/Directory                                             | Description                          |
| ---------------------------------------------------------- | ------------------------------------ |
| [src/](src/)                                               | Source code for setup and styling    |
| [src/setup.py](src/setup.py)                               | Setup and initialization code        |
| [src/styles.css](src/styles.css)                           | Custom styling                       |
| [src/build.py](src/build.py)                               | Build mode functionality             |
| [src/cortex_functions.py](src/cortex_functions.py)         | Core functions for Cortex operations |
| [src/query_result_builder.py](src/query_result_builder.py) | Query result handling and display    |
| [src/playground.py](src/playground.py)                     | Playground mode functionality        |
| [src/rag.py](src/rag.py)                                   | RAG mode functionality               |
| [src/fine_tune.py](src/fine_tune.py)                       | Fine-tuning functionality            |
| [src/search.py](src/search.py)                             | Cortex Search Functionality          |
| [.gitignore](.gitignore)                                   | Git ignore file                      |
| [requirements.txt](requirements.txt)                       | Project dependencies                 |
| [streamlit_app.py](streamlit_app.py)                       | Main application entry point         |

## Troubleshooting

Common issues and solutions:

1. **Snowflake Permission Errors**

   - Verify role privileges for database and stage creation
   - Ensure Cortex functionalities are enabled

2. **Dependency Issues**

   - If pandas/pyarrow installation fails:
     ```bash
     pip install pandas pyarrow
     ```

3. **Connection Issues**
   - Verify Snowflake account credentials
   - Check network connectivity
   - Ensure proper role assignments

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For assistance:

- Open an issue in the GitHub repository
- Contact project maintainers
- Check documentation for common solutions

## Warranty

The Software is provided as Open Source. This software is provided “as is” and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Legal

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this tool except in compliance with the License. You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

This is an Open Source repository and not an official Snowflake offering. This tool is not endorsed by Snowflake or any of the previous or current employers of the developers.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. SNOWFLAKE is a trademark of Snowflake Computing, Inc in the United States and/or other countries. Any use of third-party trademarks or logos are subject to those third-party's policies.
