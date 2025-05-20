import sys
import os
import datetime
import re
import time
# import numpy as np
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv # Import for .env file handling

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from dateutil import parser
from dateutil.relativedelta import relativedelta
from langchain_community.tools import DuckDuckGoSearchRun
import pypdf

# --- Load Environment Variables from .env file ---
load_dotenv() # This will load variables from a .env file in the same directory

# --- Model and Embeddings Initialization ---
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key: # Check if the key was loaded
        raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")

    model = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
        temperature=0.1,
    )
    print(f"Using OpenAI model: {model.model_name}")

except ValueError as ve:
    print(f"Error: {ve}")
    print("Please ensure your OPENAI_API_KEY is set in a .env file or as an environment variable.")
    print("Example .env file content: OPENAI_API_KEY='your-sk-xxxx'")
    sys.exit(1)
except Exception as e:
    print(f"Error initializing ChatOpenAI: {e}")
    sys.exit(1)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

# --- Study Corpus ---
class StudyCorpus:
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.material_store = None
        self.all_materials = []

    def add_study_material(self, content: str, source: str = "unknown"):
        self.all_materials.append({"content": content, "source": source})
        documents = [Document(page_content=content, metadata={"type": "study_material", "source": source})]
        
        if self.material_store is None:
            if documents:
                self.material_store = FAISS.from_documents(documents, self.embeddings)
        else:
            if documents:
                self.material_store.add_documents(documents)
    
    def search_materials(self, query: str, k: int = 5) -> List[str]:
        if not self.material_store or not self.all_materials:
            return []
            
        results = self.material_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

def initialize_study_corpus(study_dir: str = "study_materials") -> StudyCorpus:
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
        print(f"Created directory: {study_dir}. Please add your study materials (PDF, TXT) there.")

    corpus = StudyCorpus(embeddings)
    
    print(f"Loading study materials from: {study_dir}")
    loaded_files = 0
    if not os.path.isdir(study_dir): # Check if study_dir is actually a directory
        print(f"Error: {study_dir} is not a directory. Cannot load materials.")
        return corpus

    for filename in os.listdir(study_dir):
        file_path = os.path.join(study_dir, filename)
        content = None
        try:
            if filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                print(f"Successfully loaded: {filename}")
            elif filename.lower().endswith(".pdf"):
                text_parts = []
                try:
                    with open(file_path, "rb") as f:
                        pdf_reader = pypdf.PdfReader(f)
                        if not pdf_reader.pages:
                            print(f"Warning: PDF {filename} has no pages or could not be read.")
                            continue
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            extracted_text = page.extract_text()
                            if extracted_text:
                                text_parts.append(extracted_text)
                    content = "\n".join(text_parts)
                    if content.strip():
                         print(f"Successfully extracted text from PDF: {filename}")
                    else:
                        print(f"Warning: No text extracted from PDF (it might be image-based or empty): {filename}")
                except pypdf.errors.PdfReadError:
                    print(f"Error: Could not read PDF {filename}. It might be corrupted or password-protected.")
                except Exception as e:
                    print(f"Error processing PDF {filename}: {e}")
            
            if content and content.strip():
                corpus.add_study_material(content, source=filename)
                loaded_files +=1
            elif not content and filename.lower().endswith((".txt", ".pdf")):
                 print(f"No content loaded from {filename}.")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    if loaded_files == 0:
        print(f"No study materials were loaded. Ensure files are in '{study_dir}' and are readable.")
    else:
        print(f"Loaded {loaded_files} file(s) into the study corpus.")
    return corpus

study_corpus = initialize_study_corpus()

# --- Calendar Management (Persistent) ---
CALENDAR_FILE = "data/study_calendar.txt"

def ensure_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(CALENDAR_FILE):
        with open(CALENDAR_FILE, "w", encoding="utf-8") as f:
            pass

ensure_data_dir()

# --- Tools for the Study Planner Assistant ---

class StudyCalendarTool:
    def __init__(self, corpus: StudyCorpus):
        self.corpus = corpus

    def schedule_study_session(self, event_details: str) -> str:
        event_details = event_details.strip()
        
        topic_match = re.search(r"Topic:\s*([^-]+)", event_details, re.IGNORECASE)
        date_match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", event_details, re.IGNORECASE)
        time_match = re.search(r"Time:\s*(\d{2}:\d{2})", event_details, re.IGNORECASE)
        duration_match = re.search(r"Duration:\s*([^-]+)", event_details, re.IGNORECASE)

        topic = topic_match.group(1).strip() if topic_match else "Unspecified Topic"
        date_str = date_match.group(1).strip() if date_match else "Unspecified Date"
        time_str = time_match.group(1).strip() if time_match else "Unspecified Time"
        duration_str = duration_match.group(1).strip() if duration_match else "Unspecified Duration"

        event_entry = f"Scheduled Study: {topic} - Date: {date_str} - Time: {time_str} - Duration: {duration_str}"
        
        try:
            with open(CALENDAR_FILE, "a", encoding="utf-8") as f:
                f.write(event_entry + "\n")
            return f"Scheduled: '{topic}' on {date_str} at {time_str} for {duration_str}."
        except Exception as e:
            return f"Error scheduling event: {e}"

    def check_study_schedule(self, query: str) -> str:
        try:
            with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
                events = f.read().splitlines()
        except FileNotFoundError:
            return "No study schedule found. Please schedule some sessions first."
        
        if not events:
            return "Your study schedule is currently empty."
        
        relevant_events = []
        try:
            parsed_query_date = parser.parse(query, fuzzy=True)
            query_date_str = parsed_query_date.strftime("%Y-%m-%d")
            relevant_events = [event for event in events if query_date_str in event]
            if not relevant_events:
                relevant_events = [event for event in events if query.lower() in event.lower()]
        except (ValueError, TypeError, OverflowError): # Added OverflowError for very large date strings
             relevant_events = [event for event in events if query.lower() in event.lower()]

        if not relevant_events:
            return f"No study sessions found matching '{query}'."
        
        return "Found the following study sessions:\n" + "\n".join(relevant_events)

class CorpusQueryTool:
    def __init__(self, corpus: StudyCorpus):
        self.corpus = corpus
    
    def query_material(self, query: str) -> str:
        results = self.corpus.search_materials(query, k=3)
        if not results:
            return "I couldn't find specific information on that topic in your study materials."
        return "Here's what I found in your study materials related to your query:\n\n" + "\n\n---\n\n".join(results)

class QuizGeneratorTool:
    def __init__(self, llm, corpus_query_tool: CorpusQueryTool):
        self.llm = llm
        self.corpus_query_tool = corpus_query_tool

    def generate_quiz(self, topic: str, num_questions: int = 3) -> str:
        try:
            match_num_questions = re.search(r"with (\d+) questions", topic, re.IGNORECASE)
            if match_num_questions:
                num_questions = int(match_num_questions.group(1))
                topic = re.sub(r"with (\d+) questions", "", topic, flags=re.IGNORECASE).strip()

            context_data = self.corpus_query_tool.query_material(topic)
            if "I couldn't find specific information" in context_data:
                return f"Sorry, I can't generate a quiz on '{topic}' as I couldn't find enough information in your study materials."

            prompt_template = PromptTemplate(
                input_variables=["topic", "num_questions", "context"],
                template="""Based on the following context about {topic}, generate {num_questions} quiz questions.
                The questions should test understanding of key concepts.
                For each question, also provide a brief answer.
                Format:
                Q1: [Question]
                A1: [Answer]

                Context:
                {context}

                Quiz:"""
            )
            quiz_chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = quiz_chain.invoke({"topic":topic, "num_questions":num_questions, "context":context_data})
            return f"Here's a quiz on '{topic}':\n{response['text'] if isinstance(response, dict) else response}"
        except Exception as e:
            return f"Error generating quiz: {e}"

class DateTool:    
    def get_current_date(self, _: str = "") -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d")

# --- Initialize Tools ---
date_tool = DateTool()
study_calendar_tool = StudyCalendarTool(study_corpus)
corpus_query_tool = CorpusQueryTool(study_corpus)
quiz_generator_tool = QuizGeneratorTool(model, corpus_query_tool)
search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(name="CheckCurrentDate", func=date_tool.get_current_date, description="Use to check today's date. Returns current date in YYYY-MM-DD format. Input can be empty."),
    Tool(name="ScheduleStudySession", func=study_calendar_tool.schedule_study_session, description="Use to schedule a new study session. Input should include topic, date (YYYY-MM-DD), time (HH:MM), and duration (e.g., '1 hour'). Example: 'Topic: Photosynthesis - Date: 2025-06-15 - Time: 14:00 - Duration: 2 hours'. If date/time is relative (e.g., 'tomorrow', 'next Monday at 2pm'), first use CheckCurrentDate to get today's date, then calculate the absolute date and time before calling this tool."),
    Tool(name="CheckStudySchedule", func=study_calendar_tool.check_study_schedule, description="Use to check the study schedule for upcoming or past events. Input can be a date (YYYY-MM-DD), a topic, or keywords like 'next week', 'what am I studying on Monday?'. For relative dates like 'next week', first determine the date range using CheckCurrentDate if necessary."),
    Tool(name="QueryStudyMaterial", func=corpus_query_tool.query_material, description="Use to search and retrieve information from your uploaded study materials (PDFs, notes). Input should be a topic, specific question, or keywords you want to find information about. Example: 'What are the stages of mitosis?' or 'Summarize the key points of chapter 3 on thermodynamics.'"),
    Tool(name="GenerateQuiz", func=quiz_generator_tool.generate_quiz, description="Use to generate quiz questions on a specific topic based on your study materials. Input should be the topic name. You can optionally specify the number of questions. Example: 'Generate a quiz on Quantum Physics' or 'Create 5 quiz questions about the French Revolution'."),
    Tool(name="GeneralSearch", func=search_tool.run, description="Use for general knowledge questions or topics not covered in the study materials.")
]

# --- Agent Creation ---
def create_study_agent():
    if model is None:
        raise ValueError("LLM (model) is not initialized.")

    # This is the new, corrected prefix.
    # It contains your custom instructions but stops before the tool listing and format instructions.
    agent_instructions_prefix = """You are a helpful Study Planner Assistant. Your goal is to help the user manage their study schedule, understand their study materials, and test their knowledge.

    When scheduling study sessions:
    - If a relative date/time is given (e.g., "tomorrow", "next Monday"), ALWAYS use "CheckCurrentDate" first to get today's date.
    - Then, calculate the target absolute date (YYYY-MM-DD) and time (HH:MM).
    - Finally, use "ScheduleStudySession" with the full details: "Topic: [topic] - Date: [YYYY-MM-DD] - Time: [HH:MM] - Duration: [duration]".

    When checking the study schedule for queries like "next week" or "this month":
    - First use "CheckCurrentDate" to understand the current date.
    - Then formulate your query for "CheckStudySchedule" based on the date range you derive.

    When asked to "Summarize what I've already covered":
    1. Think about how to determine what has been "covered". This might involve checking past events in the study schedule.
    2. Use "CheckStudySchedule" to find past study sessions or relevant topics.
    3. For each covered topic/session, use "QueryStudyMaterial" to get the relevant text from the study materials.
    4. Synthesize the information from these materials to provide a summary. Do not just list the topics, but briefly summarize their content.

    When asked to generate a quiz:
    - Use the "GenerateQuiz" tool. Provide the topic clearly.

    If a user asks a general question that might not be in their study materials, consider using "GeneralSearch".
    Always try to use your specialized tools first for study-related tasks.
    Be polite and helpful!

    You have access to the following tools:""" # Standard lead-in to the tool list

    # The 'format_instructions' will be the default from Langchain for REACT agents,
    # which correctly handles {tool_names}. We don't need to specify it here
    # unless we want to significantly change the "Thought/Action/Observation" flow.

    return initialize_agent(
        tools,
        model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors="Check your output and try again. Ensure action input is correctly formatted. Stick to the provided tool names.",
        max_iterations=7,
        agent_kwargs={
            "prefix": agent_instructions_prefix,
            # "format_instructions": "..." # Only if you want to override the default ReAct format instructions
            # "suffix": "..." # Only if you want to override the default suffix ("Begin!\n\nQuestion: {input}\n{agent_scratchpad}")
        }
    )
    
# --- Main Interaction Logic ---
def get_study_response(query: str, agent):
    try:
        response = agent.invoke({"input": query})
        return response.get('output', "No output from agent.") if isinstance(response, dict) else str(response)
    except Exception as e:
        print(f"Error during agent execution: {e}")
        return "I encountered an issue trying to process that. Could you please rephrase or try a different request?"

def parse_input(user_input: str) -> str | None:
    if user_input.lower() in ['quit', 'exit', 'q']:
        return None
    return user_input

def run_study_planner():
    print("📚 Study Planner Assistant (OpenAI Edition with .env) 📚")
    print("----------------------------------------------------")
    print("Ensure your study materials are in the 'study_materials' folder.")
    print("Ensure your OPENAI_API_KEY is in a '.env' file in this directory.")
    print("Type 'quit' or 'exit' to leave.")
    
    try:
        study_agent = create_study_agent()
    except ValueError as ve:
        print(f"Failed to create agent: {ve}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during agent creation: {e}")
        return

    while True:
        try:
            user_input = input("\n🧑‍🎓 You: ")
            query = parse_input(user_input)
            if query is None:
                print("\n🤖 Assistant: Happy studying! Goodbye!")
                sys.exit()
            if not query.strip():
                print("\n🤖 Assistant: Please enter a request or type 'quit' to exit.")
                continue
            print("\n🤖 Assistant: Processing your request...")
            response = get_study_response(query, study_agent)
            print(f"\n🤖 Assistant: {response}\n")
        except KeyboardInterrupt:
            print("\n🤖 Assistant: Happy studying! Goodbye!")
            sys.exit()
        except Exception as e:
            print(f"\n🤖 Assistant: An error occurred: {e}\nPlease try again.\n")

if __name__ == "__main__":
    # The load_dotenv() call at the top handles loading the key.
    # The check for openai_api_key during model initialization handles the error message.
    run_study_planner()