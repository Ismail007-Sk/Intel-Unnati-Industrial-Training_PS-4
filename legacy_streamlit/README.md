Need of File and Folder in Client

.venv → Virtual environment folder; isolates Python packages so they don’t conflict with system or other projects.

.env → Stores environment variables (API keys, credentials).

.python-version → Specifies the Python version for this project (used by pyenv).

main.py → Usually the entry point of the Streamlit app; contains the code that runs the frontend (UI, inputs, outputs, calling backend functions).

pyproject.toml → Defines project metadata & dependencies (modern Python packaging).

README.txt → Project instructions or description.

requirements.txt → Lists Python packages to install (pip install -r requirements.txt).




Client Workflow (Streamlit frontend):

Start App: main.py runs when you execute streamlit run main.py.

Load Config: .env loads API keys or settings.

User Input: Streamlit widgets (text box, buttons) take user queries.

Send Request: Client sends query to server (backend) via API or function call.

Receive Response: Server returns the generated answer.

Display Output: Streamlit shows the answer in the app interface.

Supporting files:

.venv → ensures dependencies work correctly.

requirements.txt → installs needed packages.

README.txt → instructions for running client.

pyproject.toml & .python-version → manage project environment.
