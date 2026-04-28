import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv(".env")

def sanitize_path(path):
    """Convert Windows paths to Python-safe format"""
    return path.replace('\\', '/')

python_repl = PythonREPL()

def save_report(content, input_path):
    """Save report to text file"""
    directory = os.path.dirname(input_path)
    report_path = os.path.join(directory, "ml_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(str(content))  # Ensure content is string
    return report_path

# Define agents with proper tool configuration
eda_agent = Agent(
    role='Data Analyst',
    goal='Perform efficient EDA',
    backstory="Expert in quick data analysis and visualization",
    verbose=True,
    tools=[Tool.from_function(
        func=lambda cmd: python_repl.run(cmd),
        name="python_repl",
        description="Executes Python code"
    )]
)

ml_engineer = Agent(
    role='ML Engineer',
    goal='Select best model',
    backstory="Expert in model selection",
    verbose=True,
    allow_delegation=False
)

trainer = Agent(
    role='Trainer',
    goal='Train models efficiently',
    backstory="Expert in efficient model training",
    verbose=True,
    tools=[Tool.from_function(
        func=lambda cmd: python_repl.run(cmd),
        name="python_repl",
        description="Executes Python code"
    )]
)

tuning_specialist = Agent(
    role='Tuning Specialist',
    goal='Improve model accuracy by fine-tuning hyperparameters',
    backstory="Expert in hyperparameter tuning, optimization, and boosting model performance",
    verbose=True,
    tools=[Tool.from_function(
        func=lambda cmd: python_repl.run(cmd),

        
        name="python_repl",
        description="Executes Python code"
    )]
)

reporter = Agent(
    role='Reporter',
    goal='Generate concise report',
    backstory="Technical writer expert",
    verbose=True
)

def ml_pipeline(input_path):
    sanitized_path = sanitize_path(input_path)
    
    # File loading task
    load_task = Task(
        description=f"Load data from {sanitized_path}",
        agent=eda_agent,
        expected_output="Data loaded successfully with basic validation",
        config={'path': sanitized_path}
    )

    # EDA Task
    eda_task = Task(
        description="Perform quick data analysis",
        agent=eda_agent,
        context=[load_task],
        expected_output="Key statistics and data overview",
        config={'max_columns': 10}
    )

    # Model Selection Task
    model_task = Task(
        description="Select best model type",
        agent=ml_engineer,
        context=[eda_task],
        expected_output="Recommended model with justification"
    )

    # Training Task
    train_task = Task(
        description="Train model and generate metrics",
        agent=trainer,
        context=[model_task],
        expected_output="Trained model with evaluation metrics",
        config={'max_iter': 100}
    )

    # Fine-tuning Task
    tuning_task = Task(
        description="Fine-tune the trained model to improve accuracy",
        agent=tuning_specialist,
        context=[train_task],
        expected_output="Fine-tuned model with improved evaluation metrics",
        config={'tuning_iterations': 50}
    )

    # Report Generation Task
    report_task = Task(
        description="Generate final report with code and results",
        agent=reporter,
        context=[tuning_task],
        expected_output="Complete report file in markdown format"
    )

    crew = Crew(
        agents=[eda_agent, ml_engineer, trainer, tuning_specialist, reporter],
        tasks=[load_task, eda_task, model_task, train_task, tuning_task, report_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = crew.kickoff()
    return save_report(result, input_path)

# STREAMLIT APP
st.set_page_config(page_title="ML Pipeline Automation", page_icon="🤖", layout="centered")

st.title("🛠️ Machine Learning Pipeline Automation")
st.write("Upload your dataset (CSV) and get a full EDA, Model Selection, Training, Fine-Tuning and Report Generation.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary path
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully! 🚀")
    
    if st.button("Run ML Pipeline"):
        with st.spinner("Running the ML pipeline... This might take a few minutes..."):
            report_path = ml_pipeline(temp_path)
        
        st.success("ML Pipeline completed! 🎉")
        st.write(f"Report generated at: `{report_path}`")
        
        # Display the report
        with open(report_path, "r", encoding='utf-8') as f:
            report_content = f.read()
        
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name="ml_report.txt",
            mime="text/plain"
        )
        
        st.subheader("📄 Report Preview")
        st.text(report_content)
else:
    st.info("Please upload a CSV file to proceed.")
