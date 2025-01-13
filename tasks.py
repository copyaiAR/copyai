from crewai import Task
from agents import programer

# Writing task with language model configuration
programer_task = Task(
  description=(
    "Using this description: {description}"
    "Create accurate HTML code that matches the description."
  ),
  expected_output='One HTML file that matches the description exactly.',
  agent=programer,
  async_execution=False,
  output_file='index.html'  # Example of output customization
)