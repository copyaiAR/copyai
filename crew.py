# crew.py
from custom_agent import get_gemini_response

file_path_to_describe = "Capture.jpg"
response_from_gemini = get_gemini_response(file_path_to_describe)

# تمرير النص فقط بدلاً من كائن الاستجابة بالكامل
from crewai import Crew,Process
from tasks import programer_task
from agents import programer

## Forming the tech focused crew with some enhanced configuration
crew=Crew(
    agents=[programer],
    tasks=[programer_task],
    process=Process.sequential,
    verbose=True  # or verbose=False
)

## starting the task execution process wiht enhanced feedback

# هنا يتم تمرير النص من الاستجابة
result=crew.kickoff(inputs={'description':response_from_gemini.text})
print(result)