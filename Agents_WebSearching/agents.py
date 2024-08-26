################################################################
#                        AGENTS                                #
################################################################

#-> python agents.py

######################## Setup #################################
## for data
import base64
import os
from tqdm import tqdm

## for llm
from langchain_community.llms import Ollama #0.0.38
vision_llm = Ollama(model="llava")
llm = Ollama(model="phi3")

## for agents
from langchain_community.tools import DuckDuckGoSearchRun ##6.1.7
from crewai_tools import tool #0.4.0
import crewai #0.35.0


######################## Data ##################################
def encode_image(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

path = 'data/'
folder = [x for x in os.listdir(path) if x.endswith(('.png','.jpg','.jpeg'))]
lst_imgs = [encode_image(path+i) for i in folder]
vision_llm = Ollama(model="llava")

des = ""
for n,img in tqdm(enumerate(lst_imgs)):
    res = vision_llm.invoke(input=["Describe the image accurately"], images=[img])
    des = des.strip() + "\n\n" + f"image{n+1}: "+res.replace('\n',' ')
print("\n---Data---\n"+des)


######################## Tools ##################################
@tool("browser")
def tool_browser(q: str) -> str:
    """DuckDuckGo browser"""
    return DuckDuckGoSearchRun().run(q)

@tool("instagram")
def tool_instagram(q: str) -> str:
    '''Search Instagram'''
    return DuckDuckGoSearchRun().run(f"site:instagram.com {q}")

def callback_function(output):
    print(f"Task completed: {output.raw_output}")


######################## 1-Photographer #########################
prompt = '''Choose the picture from {images} that would get more likes on Instagram.'''

## Agent
agent_photograper = crewai.Agent(
    role="Photographer",
    goal=prompt,
    backstory='''As the Photographer, you need to understand which picture would get more likes on Instagram,
     make more people interact with the post, and maximize the conversion rate.
     Search about the current season, today's date, any particular events of this month.
     ''',
    tools=[tool_instagram], 
    max_iter=3,
    llm=llm,
    allow_delegation=False, verbose=True)

## Task
task_photograper = crewai.Task(
    description=prompt,
    agent=agent_photograper,
    callback=callback_function,
    expected_output='''Image that you chose and explain why you think is the best''')


######################## 2-Social Media Manager ##################
prompt = '''Write a caption for the post that would maximize the conversion rate on Instagram based on the image.'''

## Agent
agent_social = crewai.Agent(
    role="Social Media Manager",
    goal=prompt,
    backstory='''As the Social Media Manager, you must generate a short caption based on the output from the Photographer
     that would get more likes on Instagram, make more people interact with the post, and maximize the conversion rate. 
     Search about trending topics, hashtags and emojis. 
     ''',
    tools=[tool_instagram], 
    max_iter=3,
    llm=llm,
    allow_delegation=False, verbose=True)

## Task
task_social = crewai.Task(
    description=prompt,
    agent=agent_social,
    expected_output='''Short caption for Instagram post''')


######################## 3-Manager of the other Agents #############
prompt = '''Oversee the post creation process, choose the best picture that that maximizes the likes of the post,
            and write the best caption that maximizes the conversion rate for the post.'''

## Agent
agent_manager = crewai.Agent(
    role="Manager of the other Agents",
    goal=prompt,
    backstory='''As the manager of the process, you follow every step to create the perfect Instagram post:
     1-Choose the picture that would get more likes on Instagram with the Photograper.
     2-Write a caption for the post that would maximize the conversion rate on Instagram based on the image with the Social Media Manager.
     At the end of the process, you MUST ask the human for final approval, use the human input tool. 
     ''',
    max_iter=3,
    llm=llm,
    allow_delegation=True, verbose=True)

## Task
task_manager = crewai.Task(
    description=prompt, agent=agent_manager,
    human_input=True,
    expected_output='''Best image and short caption, basically the whole Instagram post''')


######################## Run ####################################
crew = crewai.Crew(agents=[agent_photograper, agent_social], 
                   tasks=[task_photograper, task_social, task_manager], 
                   process=crewai.Process.hierarchical,
                   manager_agent=agent_manager,
                   verbose=True)

res = crew.kickoff(inputs={"images":des})

print("\n---Res---\n"+res)

print(f'''\n---Debug---
    - task_photograper Output: {task_photograper.output.raw_output}
    - task_social Output: {task_social.output.raw_output}
    - task_manager Output: {task_manager.output.raw_output}
    ''')
