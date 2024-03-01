#############################################################
#                                                           #
# langchain and fireworks require signup to use their APIs  #
#                                                           #
# %pip install -U --quiet  langchain langgraph              #
# %pip install -U --quiet tavily-python                     #
# %pip install -U --quiet fireworks-ai                      #
#                                                           #
#############################################################

import os

# Get org ID & API keys from files
def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the script
langchain_api_key = os.path.join(script_dir, "langchain_api_key.txt")
fireworks_api_key = os.path.join(script_dir, "fireworks_api_key.txt")

os.environ["LANGCHAIN_API_KEY"] = read_file(langchain_api_key) #set your own Langchain API key
os.environ["FIREWORKS_API_KEY"] = read_file(fireworks_api_key) #set your own Fireworks API key


from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            "Generate the best essay possible for the user's request."
            "If you receive feedback or a critique of the essay, incorporate the feedback and rewite the essay."
            "If you rewrite the essay, preface it with 'Revised Essay:' to make it clear that what follows is a revision. "
            "If you have no feedback or critiques, you do not need to rewrite the essay.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",   #fireworks has a significant number of LLMs to chose from, test drive diffferent ones
    model_kwargs={"max_tokens": 32768},
)
generate = prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            "Provide detailed recommendations, including requests for length, depth, style, etc."
            "Make sure to include an overall score from 0 - 100 prefaced by the indicator 'Score:' for the essay prominently as either the first line or title of the critique.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm

def initialize_reflector():
  request = ""
  essay = ""
  reflection = ""
  s_score = ""
  score = 0
  revised_essay = ""
  revision_count = 0

  #write initial essay
  request = HumanMessage(
  content="Write an essay on the history of Canada." #time to learn a bit about your northern neighbor
    )
  for chunk in generate.stream({"messages": [request]}):
      #print(chunk.content, end="")
      essay += chunk.content

  #get intial reflection
  for chunk in reflect.stream({"messages": [request, HumanMessage(content=essay)]}):
      #print(chunk.content, end="")
      reflection += chunk.content

  #get initial score
  first_pass = reflection.split("Score: ")
  s_score = str(first_pass[1]).split("\n")[0]
  score = int(s_score.split("/")[0])

  return request, essay, reflection, score

def iterate(request, essay, reflection, score):
  revised_essay = essay
  s_score = ""
  revision_count = 0
  target_score = 95 #adjust this as needed
    
  while score < target_score: #
      # Initialize a variable to hold the latest revision
      latest_revision = ""

      # Improve essay
      for chunk in generate.stream({"messages": [request, AIMessage(content=revised_essay), HumanMessage(content=reflection)]}):
          # Accumulate chunks in latest_revision
          latest_revision += chunk.content

      # After accumulating all chunks, update revised_essay with the latest revision
      revised_essay = latest_revision

      revision_count += 1
      # Split the string into words, removing any leading or trailing whitespace
      words = revised_essay.strip().split()
      essay_length = len(words)


      print("Revision count: ", revision_count)
      print('Essay length: ', essay_length)
      #print(revised_essay)

      # Reflect on improved essay
      reflection = ""
      for chunk in reflect.stream({"messages": [request, HumanMessage(content=revised_essay)]}):
          reflection += chunk.content
          # Extract score from reflection
          if "Score: " in reflection:
              s_score = reflection.split("Score: ")[1].split("\n")[0]
              score = int(s_score.split("/")[0])

      print('Essay score: ', score, "\n")

  return request, revised_essay, reflection, score


def main():
  score=0
  essay=""
  revised_essay=""
  reflection=""

  request, essay, reflection, score = initialize_reflector()
  request, revised_essay, reflection, score = iterate(request, essay, reflection, score)

  print(request, '\n')
  print(revised_essay)

if __name__ == "__main__":
    main()
