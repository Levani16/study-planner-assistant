# Study Planner Assistant (OpenAI Edition)

## Brief Description

The Study Planner Assistant is a Python-based interactive command-line application designed to help users manage their study schedule, query their study materials, and test their knowledge. It leverages Langchain with an OpenAI language model (e.g., GPT-3.5-turbo, GPT-4o-mini) to understand user requests and perform actions.

Key features include:
* **Study Material Management (RAG):** Upload your study notes (PDFs, TXT files) and the assistant can search and retrieve information from them to answer your questions or provide summaries.
* **Calendar Integration:** Schedule study sessions, check your upcoming schedule, and get an overview of your study commitments. Calendar events are stored locally in a text file.
* **Quiz Generation:** Ask the assistant to generate quiz questions on topics covered in your study materials to test your understanding.
* **Date Information:** Get the current date.
* **General Web Search:** For queries not covered by your study materials, the assistant can perform a general web search.

## How to Run

### Prerequisites

1.  **Python:** Python 3.7 or newer installed. You can download it from [python.org](https://www.python.org/downloads/).
2.  **OpenAI API Key:** You need an API key from OpenAI. You can get this by creating an account and setting up billing at [platform.openai.com](https://platform.openai.com/).
3.  **Git (Optional, for version control):** If you plan to manage your project with Git and upload it to GitHub. Download from [git-scm.com](https://git-scm.com/).

### Setup Steps

1.  **Save the Script:**
    * Download or save the Python script (e.g., `study_planner_script.py`) to a dedicated project folder (e.g., `MyStudyPlanner`).

2.  **Install Required Libraries:**
    * Open your terminal or command prompt.
    * Navigate to your project folder (e.g., `cd path/to/MyStudyPlanner`).
    * Run the following command to install all necessary Python packages:
        ```bash
        pip install langchain langchain-openai sentence-transformers faiss-cpu pypdf python-dateutil duckduckgo-search python-dotenv
        ```

3.  **Create `.env` File for API Key:**
    * In the root of your project folder (same directory as your Python script), create a new file named exactly `.env`.
    * Open this `.env` file with a text editor and add your OpenAI API key:
        ```env
        OPENAI_API_KEY='sk-YOUR_ACTUAL_OPENAI_API_KEY_HERE'
        ```
        Replace `sk-YOUR_ACTUAL_OPENAI_API_KEY_HERE` with your real OpenAI API key.
    * (Optional) You can also specify a preferred OpenAI model in the `.env` file:
        ```env
        OPENAI_MODEL_NAME='gpt-4o-mini'
        ```
        If not set, the script defaults to "gpt-3.5-turbo".

4.  **Prepare Study Materials:**
    * In your project folder, create a subfolder named `study_materials`.
    * Place your study files (e.g., `chapter1.pdf`, `lecture_notes.txt`) into this `study_materials` folder. The script will load these when it starts.

5.  **Directory Structure:**
    Your project directory should ideally look like this:
    ```
    MyStudyPlanner/
    ├── .env                      # Stores your API key
    ├── study_planner_script.py   # The main Python script
    ├── study_materials/          # Folder for your notes
    │   ├── your_notes.pdf
    │   └── another_file.txt
    ├── data/                     # Will be created automatically by the script
    │   └── study_calendar.txt    # Stores calendar events
    └── README.md                 # This documentation file
    ```
    If you're using Git, also add a `.gitignore` file (see GitHub upload instructions for its content, especially to ignore `.env`).

### Running the Script

1.  **Navigate to Project Directory:**
    * Open your terminal or command prompt.
    * Use the `cd` command to go into your project folder (e.g., `cd path/to/MyStudyPlanner`).

2.  **Execute the Script:**
    * Run the script using Python:
        ```bash
        python study_planner_script.py
        ```
        (Or `python3 study_planner_script.py` if `python` defaults to Python 2 on your system).

3.  **Interact with the Assistant:**
    * The script will start and print some initialization messages (loading materials, model being used).
    * You will then see a prompt like: `🧑‍🎓 You:`.
    * Type your requests or questions here. For example:
        * "What are the main topics in `your_notes.pdf`?"
        * "Schedule a study session for 'Calculus Review' next Friday at 2 PM for 2 hours."
        * "Generate a quiz on 'Chapter 1 concepts'."
        * "What is today's date?"
        * "Summarize what I studied last week on 'History'."
    * The assistant uses a ReAct agent, so you will see its "Thought" process (which tools it's choosing, etc.) followed by the final answer. This verbose output is helpful for understanding its actions and for debugging.

4.  **Stopping the Assistant:**
    * At the `🧑‍🎓 You:` prompt, type `quit`, `exit`, or `q` and press Enter.
    * Alternatively, you can press `Ctrl+C` in the terminal.

## Key Functionalities Overview

* **Query Study Material (`QueryStudyMaterial` tool):** Ask questions about the content of your PDFs and TXT files in the `study_materials` folder.
    * *Example:* "Explain the concept of [topic] from my notes."
* **Schedule Study Sessions (`ScheduleStudySession` tool):** Add study events to a local calendar file (`data/study_calendar.txt`). Can understand specific and relative dates (e.g., "tomorrow", "next Monday").
    * *Example:* "Schedule 'Biology Chapter 5 reading' for tomorrow at 4 PM for 1 hour."
* **Check Study Schedule (`CheckStudySchedule` tool):** Retrieve and view your scheduled study sessions.
    * *Example:* "What am I studying next week?" or "Do I have anything scheduled for 2025-06-10?"
* **Generate Quizzes (`GenerateQuiz` tool):** Create quiz questions based on topics found within your study materials.
    * *Example:* "Generate a 3-question quiz on [topic from your notes]."
* **Check Current Date (`CheckCurrentDate` tool):** Useful for the assistant (and you) to know today's date, especially when scheduling with relative terms.
    * *Example:* "What is today's date?"
* **General Web Search (`GeneralSearch` tool):** For questions not covered in your study materials, the assistant can perform a web search using DuckDuckGo.
    * *Example:* "What is the latest news on AI?"
