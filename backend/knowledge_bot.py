import requests, time, json, re, os, sys
from io import StringIO
from google.oauth2.service_account import Credentials
import google.auth
import google.auth.transport.requests
from matching_engine_bigtable import MatchingEngine
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.agents import ConversationalAgent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.agents import ZeroShotAgent
from langchain.chains import LLMChain


native_PROMPT = """You are an ai assistant of a story named ReFantasy. Your name is Gegewu.

Here are some functions you can call to meet user's needs:
functions:
$functions$


respond format:
    respond in json with three fields:
    is_current_context_enough: boolean. sometimes, the context has some info for user's question, but not very match, in this case, set it false, and user function call to get more context.
    need_call_function: boolean. if you need function call to solve the question, set it true, else false.
    function_call: string. if need_call_function is true, set the function call, such as "query_story_context(question='who is dasao')", else null
    is_same_function_call_in_user_history: boolean. if the same function call including same params has been called in the user_history, set true, else false.
    should_answer: boolean. when need funcion call and not called before, set false, else true.
    answer: string. if should answer, answer here. if you can not get enough context to answer after function call, anwser "sorry, I don't know"

respond example:

{
  "is_current_context_enough": false,
  "need_call_function": true,
  "function_call": "query_story_context(question='who is dasao')",
  "is_same_function_call_in_user_history": false,
  "should_answer": false,
  "answer": null
}
{
  "is_current_context_enough": false,
  "need_call_function": true,
  "function_call": "query_story_context(question='who is dasao')",
  "is_same_function_call_in_user_history": true,
  "should_answer": true,
  "answer": "answer content"
}


user_history:
$user_history$
"""

conv_PREFIX = """Assistant is a large language model trained by Google. Assistant name is Gegewu.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant must use tools for every input.

TOOLS:
------

Assistant has access to the following tools:"""
conv_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]ß
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response here]
```"""

conv_SUFFIX = """Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

react_PREFIX = """Answer the following questions as best you can. The answer should be as detailed  as possible. You have access to the following tools:"""
react_FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times. )
Thought: I now know the final answer
Final Answer: the final answer to the original input question. The final answer should be very detailed."""
react_SUFFIX = """Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

class change_model():
    def __init__(self, model_name: str = "palm", prompt_type: str = "native", vectordb: MatchingEngine = None, collection: str = ""):
        if model_name == "palm":
            project_id = os.environ.get('ME_PROJECT_ID')
            region = os.environ.get('ME_LOCATION')
            vertexai.init(project=project_id, location=region)

            self.llm_model = VertexAI()
            self.embedding_model = VertexAIEmbeddings()
            self.llm_model_native = TextGenerationModel.from_pretrained("text-bison@001")
            self.collection = collection
            if prompt_type == "native":
                self.knowledge_bot = knowledge_bot_me(vectordb,  self.llm_model_native, self.embedding_model, self.collection)
            elif prompt_type == "react":
                self.knowledge_bot = knowledge_bot_langchain_react_me( vectordb, self.llm_model, self.embedding_model, self.collection)
            elif prompt_type == "conversational":
                self.knowledge_bot = knowledge_bot_langchain_conv_me( vectordb, self.llm_model, self.embedding_model, self.collection)
            else:
                self.knowledge_bot = knowledge_bot_me(vectordb, self.llm_model_native, self.embedding_model, self.collection)
        
        elif model_name == "enterprise_search":
            project_id = os.environ.get('ES_PROJECT_ID')
            region = os.environ.get('ES_LOCATION')
            model_name = "enterprise_search"
            search_engine_id = os.environ.get('DATA_STORE_ID')
            self.knowledge_bot = knowledge_bot_es(project_id, search_engine_id)
        else:
            print("Not defined yet")
    
    def get_bot(self):
        return self.knowledge_bot

    def send_message(self, query):
        result = self.knowledge_bot.run(query)
        return result



class knowledge_bot_es():
    def __init__(self, project_id, search_engine_id):
        self.project_id = project_id
        self.search_engine_id = search_engine_id

    def get_token(self):
        credentials, _ = google.auth.default()
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        token = credentials.token
        return token
 
    def get_es_result(self, query):
        url = f"https://discoveryengine.googleapis.com/v1beta/projects/{self.project_id}/locations/global/collections/default_collection/dataStores/{self.search_engine_id}/servingConfigs/default_search:search"
        headers = {
                'Authorization': 'Bearer ' + self.get_token(),
                'Content-Type': 'application/json'
        }
        params = {
                    "query": query,
                    "pageSize": 5,
                    "queryExpansionSpec": { "condition": "AUTO"},
                    "contentSearchSpec": {
                        "snippetSpec": {"maxSnippetCount":3},
                        "summarySpec": {
                            "summaryResultCount": 1,
                            "includeCitations": False,
                            "ignoreAdversarialQuery": False,
                            "ignoreNonSummarySeekingQuery": False
                        },
                        "extractiveContentSpec": {
                            "maxExtractiveAnswerCount": 2,
                            "maxExtractiveSegmentCount": 1
                        }
                    }
                }

        response = requests.post(url,headers=headers,json=params).content
        result = json.loads(response.decode("utf-8"))
        for item in result['results'][0]['document']['derivedStructData']['extractive_segments']:
            print(item['content'])
        return result['summary']['summaryText']

    def run(self, query):
        std_out = sys.stdout
        io_out = StringIO()
        sys.stdout = io_out
        result = self.get_es_result(query)
        debug_log = io_out.getvalue()
        sys.stdout = std_out  
        return result, debug_log

class knowledge_bot_me():
    def __init__(self, vectordb_instance: MatchingEngine, llm_model, embedding_model, collection):
        #init embedding model
        self.embeddings_model = embedding_model#VertexAIEmbeddings()
        #init llm model
        self.llm_model = llm_model#TextGenerationModel.from_pretrained("text-bison@001")
        #init vectordb
        self.me_instance = vectordb_instance
        #init vectordb filter
        self.collection = collection
        self.functions = self.add_tools_in_prompt()
        self.prompt = native_PROMPT.replace("$functions$", str(self.functions))
        self.user_history = []

    def reset_user_history(self):
        self.user_history = []

    def get_user_history(self):
        return self.user_history
    
    def add_tools_in_prompt(self):
        properties = {}
        properties['question'] = {}
        properties['question']['type'] = 'string'
        properties['question']['description'] = 'clear question to query'

        story_tool = self.format_tool("query_story_context", "query ReFantasy context about the parameters: question. Please refrain from repeatedly asking the same question.",
                "question",
                properties)
        return [story_tool]

    def query_story_context(self, question, k=2):
        found = self.me_instance.similarity_search(question, k=k, collection=self.collection)
        item_list = ""
        for item in found:
            item_list = item_list + item.page_content + "\n\n"
        return json.dumps({
            'status': "success",
            'context': item_list
        })
    
    def format_tool(self, name, description, required, propertities):
        data = {}
        data['name'] = name
        data['description'] = description
        data['parameters'] = {}
        data['parameters']['type'] = 'obejct'
        data['parameters']['propertities'] = propertities
        data['parameters']['required'] = [required]
        return data

    def user_history_to_str(self, user_history):
        result = ""
        roles = ['user', 'respond']
        i=0
        for h in user_history:
            role = roles[i%2]
            result = result + f"{role}: {h}\n"
            i = i + 1
        return result
    
    def parse_function_str(self, function_string):
        s = function_string
        match = re.match(r"(\w+)\((.*)\)", s)
        if match:
            function_name = match.group(1)
            params_string = match.group(2)
            params = {}

            for param in params_string.split(','):
                key, value = param.split('=')
                # 去除参数名和参数值的空格和引号
                key = key.strip()
                value = value.strip().strip('\'"')
                params[key] = value
            return function_name, params
        return None
    
    def call_function(self, function_string):
        function_name, params = self.parse_function_str(function_string)
        if function_name == 'query_story_context':
            return self.query_story_context(params['question'])
    
    def json_fixer(self, query, temperature=0):
        PROMPT = """The input json has incorrect format, please output correct json format

input: {
  \"is_current_context_enough\": true,
  \"need_call_function\": false,
  \"function_call\":  \"abc
output: {
  \"is_current_context_enough\": true,
  \"need_call_function\": false,
  \"function_call\": \"abc\"
}

input: $input$

output:
"""
        prompt = PROMPT.replace("$input$", query)
        response = self.llm_model.predict(
                    prompt,
                    temperature=temperature,
                    max_output_tokens=1024,
                    top_k=40,
                    top_p=0.8,
                )
        #print(f"Response from Model: {response.text}")
        return response.text
    
    def text_task(self, query, temperature=0):
        response = self.llm_model.predict(
            query,
            temperature=temperature,
            max_output_tokens=1024,
            top_k=40,
            top_p=0.8,
        )
        return response.text
    
    def get_native_result(self, user_input):
        print("I'm here!")
        self.user_history.append(user_input)
        while True:
            user_history_str = self.user_history_to_str(self.user_history)
            current_prompt = self.prompt.replace("$user_history$", f"{user_history_str}respond: ")
            print(current_prompt)
            respond = self.text_task(current_prompt)
            print(respond)
            self.user_history.append(respond)
            try:
                respond = json.loads(respond)
            except:
                respond = self.json_fixer(respond)
                respond = json.loads(respond)
            if respond['need_call_function'] and respond['is_same_function_call_in_user_history'] != True:
                fun_res = self.call_function(respond['function_call'])
                print(f"call: {respond['function_call']}, return: {fun_res}")
                self.user_history.append(f"{respond['function_call']} return:{fun_res}")
                continue
            else:
                answer = respond['answer']
                print(f"answer: {answer}")
                return answer
    
    def run(self, user_query):
        std_out = sys.stdout
        io_out = StringIO()
        sys.stdout = io_out
        result = self.get_native_result(user_query)
        debug_log = io_out.getvalue()
        sys.stdout = std_out  
        return result, debug_log

class knowledge_bot_langchain_conv_me():
    def __init__(self, vectordb_instance: MatchingEngine, llm_model, embedding_model, collection):
        #init llm model
        self.llm_model = llm_model#VertexAI()
        #init embedding model
        self.embeddings_model = embedding_model#VertexAIEmbeddings()
        #init vectordb
        self.me_instance = vectordb_instance
        #init vectordb filter
        self.collection = collection
        
        def get_me_in_langchain(query, k=2):
            item_list = ""
            for item in self.me_instance.similarity_search(query, k=k, collection=self.collection):
                item_list = item_list + item.page_content + "\n\n"
            return item_list

        tools = [
            Tool.from_function(
                func=get_me_in_langchain,
                name = "Matching Engine",
                description="Search for a query"
            )
        ]
        memory = ConversationBufferMemory(memory_key="chat_history")
        chat_react_agent = ConversationalAgent.from_llm_and_tools(llm=self.llm_model,
                                            tools=tools,
                                            prefix=conv_PREFIX,
                                            suffix=conv_SUFFIX,
                                            format_instructions=conv_FORMAT_INSTRUCTIONS,
                                            input_variables=["input", "chat_history", "agent_scratchpad"],
                                            #memory=memory,
                                            verbose=True                
        )
        self.chat_react_agent_executor = AgentExecutor.from_agent_and_tools(
            agent=chat_react_agent, tools=tools, verbose=True, memory=memory
        )
    
    def run(self, query):
        std_out = sys.stdout
        io_out = StringIO()
        sys.stdout = io_out
        result = self.chat_react_agent_executor.run(query)
        debug_log = io_out.getvalue()
        sys.stdout = std_out  
        return result, debug_log

class knowledge_bot_langchain_react_me():
    def __init__(self, vectordb_instance: MatchingEngine, llm_model, embedding_model, collection):
        #init llm model
        self.llm_model = llm_model#VertexAI()
        #init embedding model
        self.embeddings_model = embedding_model#VertexAIEmbeddings()
        #init vectordb
        self.me_instance = vectordb_instance
        #init vectordb filter
        self.collection = collection
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm_model, chain_type="stuff", retriever=self.me_instance.as_retriever(collection=self.collection), return_source_documents=False)
        tools = [
                Tool(
                name="Matching Engine",
                func=qa_chain.run,
                description="search for a query",
                ),
        ]

        react_prompt = ZeroShotAgent.create_prompt(
            tools,
            react_PREFIX,
            react_SUFFIX,
            react_FORMAT_INSTRUCTIONS,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        memory = ConversationBufferMemory(memory_key="chat_history")
        react_chain = LLMChain(llm=self.llm_model, prompt=react_prompt, verbose=True)
        react_agent = ZeroShotAgent(llm_chain=react_chain, tools=tools, verbose=True)
        self.react_agent_executor= AgentExecutor.from_agent_and_tools(
            agent=react_agent, tools=tools, verbose=True, memory=memory
        )

    def run(self, query):
        std_out = sys.stdout
        io_out = StringIO()
        sys.stdout = io_out
        result = self.react_agent_executor.run(query)
        debug_log = io_out.getvalue()
        sys.stdout = std_out  
        return result, debug_log
