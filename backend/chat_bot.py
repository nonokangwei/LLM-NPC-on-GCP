import json
import random
import re
import os
import sys
from io import StringIO

import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair, ChatMessage
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatMessage

extract_name_template = """
You are a name and entity extraction service. You can extract all the names and entities from the input query.
Example:
Lord: Do you like Itis?
Blake: Sure. I admire her. Do you know that Dawn loves Itis?
Lord: Oh, I don't know.
Lord: Yeah, that's a secret.
Output: Itis,Dawn,Lord,Blake

Query:
$query$
Output:"""

describe_query_prompt = """
You are an question explanation service. You can generate a detailed question from the original question, based the attached conversation history.
The output should be seperated by comma, no space.

Example:
Conversaion:
Human: Do you know Thrall?
Bot: Yes, I know. He's a hero in the World of Warcraft.

Question: What's his father's name?
Output: What's Thrall's father's name?

Conversation:
$conversation$

Question: $question$
Output:"""

summary_template = """Summarize the below conversation in detail. The summary must be very descriptive.

$conversation_string$

summary:"""

chat_system_template = """You are acting as $character_name$ in the game $game_name$. Your response is based on the game background, character's descrption, story context, and chat history.Stay in the character in every response.

Background of $game_name$:
$world_string$

Description of $character_name$:
$bio_string$

Additional Information:
1. story context:
$public_data_string$

$character_data_string$

2. other characters' descriptions:
$related_hero_bios$

Human is acting as Lord. Human's name is $player_name$. $character_name$ and $player_name$(Current Emotion: $emotion$) are talking now"""


text_template = """You are acting as $character_name$ in the game $game_name$. Your response is based on the game background, character's descrption, story context, and chat history.Stay in the character in every response, and generate response like the talking style.

Background of $game_name$:
$world_string$

Description of $character_name$:
$bio_string$

$character_name$'s Talking Style:
$pre_conversation_string$

Additional Information
1. story context:
$public_data_string$
$character_data_string$

2. other characters' descriptions:
$related_hero_bios$

Human is acting as Lord. Human's name is $player_name$. $character_name$ and $player_name$(Current Emotion: $emotion$) are talking now.

$conversation_string$
$player_name$: $text$
$character_name$:"""

text_template_backup = """You are acting as $character_name$ in the game $game_name$. Your response is based on the game background, character's descrption, story context, and chat history.Stay in the character in every response, and generate response like the talking style.

Background of $game_name$:
$world_string$

Description of $character_name$:
$bio_string$

Additional Information
1. story context:
$public_data_string$

$character_data_string$

2. other characters' descriptions:
$related_hero_bios$

Human is acting as Lord. Human's name is $player_name$. $character_name$ and $player_name$(Current Emotion: $emotion$) are talking now.

$pre_conversation_string$
$conversation_string$
$player_name$: $text$
$character_name$:"""

class chat_bot():
      def __init__(self, model_name, matching_engine_instance, data_folder='./'):
        self.model_name = model_name
        self.me_instance = matching_engine_instance
        self.data_folder = data_folder

      def change_model(self, model_name):
        self.model_name = model_name
        return True

      def list_role(self):
        hero_name_list_txt = open(f"{self.data_folder}/hero_name.txt").readlines()
        hero_name_list = [hero_name.replace('\n','') for hero_name in hero_name_list_txt]
        return hero_name_list

      def token_len(self, text):
          return len(re.findall(r'\w+', text))

      def get_world(self):
          world_string = open(f"{self.data_folder}/world.txt").read()
          return world_string

      def get_preconversation_string(self, character_name):
          # read from pre_conversation_chat folder
          # pre_conversation_string = ""
          # with open(f'{self.data_folder}/pre_conversation_chat/{character_name}.json') as f:
          #     data = f.readlines()
          # pre_conversation_list = []
          # for item in data:
          #   pair = json.loads(item)
          #   pre_conversation_string += "Lord: " + pair['input'] + '\n'
          #   pre_conversation_string += f"{character_name}: " + pair['output'] + '\n'
          # pre_conversation_strin = pre_conversation_string[:-1] #remove the final enter
          
          # return pre_conversation_string


          selected_hero = []
          with open(f'{self.data_folder}/hero.txt') as f:
              selected_hero = f.readlines()
          selected_hero = [item.replace('\n', '') for item in selected_hero]

          # Load conversation from JSON file
          with open(f'{self.data_folder}/pre_conversation/{character_name}.json', 'r') as f:
              pre_conversation = json.load(f)['pre_conversation']

          # Shuffle the lines randomly
          random.shuffle(pre_conversation)

          pre_conversation_string = ""
          total_tokens = 0
          line_tokens = 0
          # Iterate through the lines until the token length exceeds or reaches 500
          if character_name in selected_hero:
            for line in pre_conversation:
                line_text = line['line']
                if "Lord" in line_text:
                      line_tokens = self.token_len(line_text)

                      # Check if adding the line will exceed the token limit
                      if total_tokens + line_tokens > 100:
                          break

                      # Add the line to the pre_conversation_string
                      pre_conversation_string += line_text + "\n"

                      # Update the total token count
                      total_tokens += line_tokens
          else:
            for line in pre_conversation:
                line_text = line['line']
                line_tokens = self.token_len(line_text)

                # Check if adding the line will exceed the token limit
                if total_tokens + line_tokens > 100:
                    break

                # Add the line to the pre_conversation_string
                pre_conversation_string += line_text + "\n"

                # Update the total token count
                total_tokens += line_tokens
          return pre_conversation_string

      def get_preconversation_for_chat(self, character_name):
          with open(f'{self.data_folder}/pre_conversation_chat/{character_name}.json') as f:
              data = f.readlines()
          pre_conversation_list = []

          for item in data:
            pair = json.loads(item)
            pre_conversation_list.append([pair['input'], pair['output']])
          return pre_conversation_list

      def get_hero_bio(self, hero_name):
          with open(f"{self.data_folder}/new_hero_list_en.json") as f:
                data = f.read()
          hero_bios = json.loads(data)
          result = ''
          result = ''
          for item in hero_bios[hero_name]:
            result += item + ': ' + str(hero_bios[hero_name][item]) + '\n'
          return result

      def get_hero_related_bios(self, character_name, query_hero_names=None):
          with open(f"{self.data_folder}/new_hero_list_en.json") as f:
                data = f.read()
          hero_bios = json.loads(data)
          related_hero_bios = ''
          if query_hero_names != None:
              for name in query_hero_names.split(","):
                name = name.replace(' ', '')
                if name in self.list_role() and name != character_name:
                    for item in hero_bios[name]:
                      related_hero_bios += item + ': ' + str(hero_bios[name][item]) + '\n'
                    related_hero_bios += '\n'
          return related_hero_bios

      def check_create(self, path):
          if not os.path.isdir(path):
            os.mkdir(path)

      def conversation_loader(self, transcribed_text, player_name, character_name):
          len_threshold = 6
          # Load the existing conversation from conversation.json
          self.check_create(f'{self.data_folder}/summary')
          conversation = []
          if os.path.isfile(f'{self.data_folder}/summary/{character_name}.json'):
              with open(f'{self.data_folder}/summary/{character_name}.json', 'r') as f:
                 conversation = json.load(f)['conversation']

          conversation_string = ''
          for line in conversation:
              conversation_string += line['sender'] + ": " + line['message'] + '\n'

          token_length = self.token_len(conversation_string)

          if token_length > 100:
              # create the template string
              current_prompt = summary_template
              current_prompt = current_prompt.replace('$conversation_string$', conversation_string)
              summary = self.text_task(current_prompt)
              #langchain mode
              #template = """{conversation_string}\n\nSummarize the above conversation in detail. The summary must be very descriptive."""
              #prompt = PromptTemplate(template=template, input_variables=[
              #                        'conversation_string'])

              # Create and run the llm chain
              #llm_chain = LLMChain(prompt=prompt, llm=self.llm_model)
              #summary = llm_chain.run(conversation_string=conversation_string)
              print('********************summary here****************************')
              print(summary)
              print('********************summary end****************************')
              self.me_instance.add_texts(texts=[summary], collection=character_name + '_summary')
              self.me_instance.add_texts(texts=[conversation_string], collection=character_name +'_conversation')
              conversation = []
              conversation_string = ''
          # Save the updated conversation back to conversation.json
          with open(f'{self.data_folder}/summary/{character_name}.json', 'w') as f:
              json.dump({'conversation': conversation}, f)
          if len(conversation) > len_threshold - 1:
              conversation_string_wthreshold = ''
              for line in conversation[-len_threshold:]:
                  conversation_string_wthreshold += line['sender'] + ": " + line['message'] + '\n'
              if self.model_name == 'chat_palm':
                  return conversation_string_wthreshold, conversation
              else:
                  return conversation_string_wthreshold, conversation_string

          else:
              #get conversation history if conversation in summary file is not enough
              self.check_create(f'{self.data_folder}/conversation')
              conversation = []
              if os.path.isfile(f'{self.data_folder}/conversation/{character_name}.json'):
                with open(f'{self.data_folder}/conversation/{character_name}.json', 'r') as f:
                    conversation = json.load(f)['conversation']
              if len(conversation) > len_threshold :
                  conversation = conversation[-len_threshold:]
              conversation_string = ''
              for line in conversation:
                  conversation_string += line['sender'] + ": " + line['message'] + '\n'
              if self.model_name == 'chat_palm':
                  return conversation_string, conversation
              else:
                  return conversation_string, conversation_string

      def conversation_loader_for_chat(self, transcribed_text, character_name):
            self.check_create(f'{self.data_folder}/conversation')
            conversation = []
            if os.path.isfile(f'{self.data_folder}/conversation/{character_name}.json'):
              with open(f'{self.data_folder}/conversation/{character_name}.json', 'r') as f:
                  conversation = json.load(f)['conversation']
            if len(conversation) > 7 :
                conversation = conversation[-6:]
            return conversation


      def get_public_data(self, conversation_string, transcribed_text, player_name, character_name, search_type="conversation"):
          if search_type == "conversation":
              conversation_string += player_name + ": " + transcribed_text + '\n'
              docs = self.me_instance.similarity_search(query=conversation_string, k=2, collection=f"public_dialog")
          elif search_type == "detailed_query":
              docs = self.me_instance.similarity_search(query=self.describe_query(conversation_string, transcribed_text), k=2, collection=f"public_dialog")

          public_info_string = "\n\n".join(doc.page_content for doc in docs)

          # to fix the lord's name's error
          #public_info_string = public_info_string.replace('Master:', 'Lord:')
          return public_info_string

      def get_character_data(self, conversation_string, character_name, player_name, transcribed_text, search_type="conversation"):
          if search_type == "conversation":
              conversation_string += player_name + ": " + transcribed_text + '\n'
              docs = self.me_instance.similarity_search(query=conversation_string, k=1, collection=f"{character_name}_summary")
          elif search_type == "detailed_query":
              docs = self.me_instance.similarity_search(query=self.describe_query(conversation_string, transcribed_text), k=1, collection=f"{character_name}_summary")
          character_info_string = "\n".join(doc.page_content for doc in docs)
          return character_info_string

      def extract_name(self, conversation_string, character_name, player_name, transcribed_text, search_type="conversation"):
          #langchain mode
          # extract_prompt = PromptTemplate(input_variables=["query"], template=extract_name_template)
          # extract_llm_chain = LLMChain(prompt=extract_prompt, verbose=True, llm=self.llm_model)
          # if search_type == "conversation":
          #     conversation_string += player_name + ": " + transcribed_text + '\n'
          #     extract_query = extract_llm_chain.predict(query=conversation_string)
          # elif search_type == "detailed_query":
          #     extract_query = extract_llm_chain.predict(query=self.describe_query(conversation_string, transcribed_text))
          current_prompt = extract_name_template
          if search_type == "conversation":
              conversation_string += player_name + ": " + transcribed_text + '\n'
              current_prompt = current_prompt.replace("$query$", conversation_string)
              extract_names = self.text_task(current_prompt)
          elif search_type == "detailed_query":
              detailed_query = self.describe_query(conversation_string, transcribed_text)
              extract_names = self.text_task(detailed_query)
          return extract_names

      def describe_query(self, conversation_string, transcribed_text):
          #langchain model
          # describe_prompt = PromptTemplate(input_variables=["conversation", "question"],template=describe_query_prompt)
          # describe_llm_chain = LLMChain(prompt=describe_prompt, verbose=True, llm=self.llm_model)
          # detailed_query = describe_llm_chain.predict(question=transcribed_text, conversation=(conversation_string))
          current_prompt = describe_query_prompt
          current_prompt = current_prompt.replace("$conversation$", conversation_string)
          current_prompt = current_prompt.replace("$question$", transcribed_text)
          detailed_query = self.text_task(current_prompt)
          return detailed_query

      def add_history(self, query, output, character_name, player_name):
          #add in conversation file
          conversation = []
          self.check_create(f'{self.data_folder}/conversation')
          if os.path.isfile(f'{self.data_folder}/conversation/{character_name}.json'):
            with open(f'{self.data_folder}/conversation/{character_name}.json', 'r') as f:
                conversation = json.load(f)['conversation']

          conversation.append({'sender': player_name, 'message': query})
          conversation.append({'sender': character_name, 'message': output})
          with open(f'{self.data_folder}/conversation/{character_name}.json', 'w') as f:
                json.dump({'conversation': conversation}, f)

          #add in summary file for candidate conversation
          self.check_create(f'{self.data_folder}/summary')
          if os.path.isfile(f'{self.data_folder}/summary/{character_name}.json'):
            with open(f'{self.data_folder}/summary/{character_name}.json', 'r') as f:
                conversation = json.load(f)['conversation']

          conversation.append({'sender': player_name, 'message': query})
          conversation.append({'sender': character_name, 'message': output})
          with open(f'{self.data_folder}/summary/{character_name}.json', 'w') as f:
                json.dump({'conversation': conversation}, f)

      def reset_history(self, character_name):
          #delete conversation file
          conversation = []
          if os.path.isdir(f'{self.data_folder}/conversation'):
            if os.path.isfile(f'{self.data_folder}/conversation/{character_name}.json'):
              with open(f'{self.data_folder}/conversation/{character_name}.json', 'w') as f:
                    json.dump({'conversation': conversation}, f)
          #delete summary file
          if os.path.isdir(f'{self.data_folder}/summary'):
            if os.path.isfile(f'{self.data_folder}/summary/{character_name}.json'):
              with open(f'{self.data_folder}/summary/{character_name}.json', 'w') as f:
                    json.dump({'conversation': conversation}, f)
          
          #delete me_instance collection
          try:
            self.me_instance.delete_collection(collection=f"{character_name}_summary")
            self.me_instance.delete_collection(collection=f"{character_name}_conversation")
          except:
            return False
          return True

      def text_task(self, query, temperature=0.2):
          model = TextGenerationModel.from_pretrained("text-bison@001")
          response = model.predict(
              query,
              temperature=temperature,
              max_output_tokens=1024,
              top_k=40,
              top_p=0.8,
          )
          return response.text

      def chat_task(self, character_name, context, examples, history, transcribed_text, temperature=0.2):
          chat_model = ChatModel.from_pretrained("chat-bison@001")
          parameters = {
            "temperature": temperature,
            "max_output_tokens": 1024,
            "top_p": 0.8,
            "top_k": 40,
          }
          #add example
          chat_examples = []
          for item in examples:
              chat_examples.append(InputOutputTextPair(input_text=item[0], output_text=item[1]))
          chat = chat_model.start_chat(context=context, examples=chat_examples)
          #add history
          for line in history:
            author = 'bot' if line['sender'] == character_name else 'user'
            chat.message_history.append(ChatMessage(content=line['message'], author=author))
          print(context)
          print(chat._examples)
          print(chat.message_history)
          response = chat.send_message(transcribed_text, **parameters)
          return response.text

      def chat_side_character(self, game_name, transcribed_text, player_name, character_name, emotion, search_type):
          #pre-defined information
          pre_conversation_string = ''
          pre_conversation_chat_list = []
          if self.model_name == "chat_palm":
            pre_conversation_chat_list = self.get_preconversation_for_chat(character_name)
          else:
            pre_conversation_string = self.get_preconversation_string(character_name)

          bio_string = self.get_hero_bio(character_name)
          world_string = self.get_world()
          #print(pre_conversation_string)
          #print(pre_conversation_chat_list)

          conversation_string = ''
          conversation_history = ''
          conversation_chat_list = []
          #print(conversation_string)
          if self.model_name == "chat_palm":
            #conversation_chat_list = self.conversation_loader_for_chat(transcribed_text, character_name)
            conversation_string, conversation_chat_list = self.conversation_loader(transcribed_text, player_name, character_name)
          else:
            conversation_string, conversation_history = self.conversation_loader(transcribed_text, player_name, character_name)
          #print(conversation_chat_list)

          public_data_string = self.get_public_data(conversation_string, transcribed_text, player_name, character_name, search_type)
          #print(public_data_string)

          character_data_string = self.get_character_data(conversation_string, character_name, player_name, transcribed_text, search_type)
          #print("**************character_data_string**************")
          #print(character_data_string)

          extract_heroes = self.extract_name(conversation_string, character_name, player_name, transcribed_text, search_type)
          #print(extract_heroes)
          related_hero_bios = self.get_hero_related_bios(character_name, extract_heroes)
          #print(related_hero_bios)
          character_reponse = 'The reponse is blocked.'
          if self.model_name == "chat_palm":
              #chat-bison
              current_prompt = chat_system_template
              current_prompt = current_prompt.replace("$game_name$", game_name)
              current_prompt = current_prompt.replace("$world_string$", world_string)
              current_prompt = current_prompt.replace("$character_name$", character_name)
              current_prompt = current_prompt.replace("$bio_string$", bio_string)
              current_prompt = current_prompt.replace("$related_hero_bios$", related_hero_bios)
              current_prompt = current_prompt.replace("$public_data_string$", public_data_string)
              current_prompt = current_prompt.replace("$character_data_string$", character_data_string)
              current_prompt = current_prompt.replace("$player_name$", player_name)
              current_prompt = current_prompt.replace("$emotion$", emotion)
              #print(current_prompt)

              character_response = self.chat_task(character_name, current_prompt, pre_conversation_chat_list, conversation_chat_list, transcribed_text)
              character_response = character_response.replace("\n", "")

          elif self.model_name== "text_palm":
              #text-bison
              current_prompt = text_template
              current_prompt = current_prompt.replace("$game_name$", game_name)
              current_prompt = current_prompt.replace("$world_string$", world_string)
              current_prompt = current_prompt.replace("$character_name$", character_name)
              current_prompt = current_prompt.replace("$bio_string$", bio_string)
              current_prompt = current_prompt.replace("$pre_conversation_string$", pre_conversation_string)
              current_prompt = current_prompt.replace("$related_hero_bios$", related_hero_bios)
              current_prompt = current_prompt.replace("$public_data_string$", public_data_string)
              current_prompt = current_prompt.replace("$character_data_string$", character_data_string)
              current_prompt = current_prompt.replace("$conversation_string$", conversation_history)
              current_prompt = current_prompt.replace("$player_name$", player_name)
              current_prompt = current_prompt.replace("$emotion$", emotion)
              current_prompt = current_prompt.replace("$text$", transcribed_text)

              print(current_prompt)
              character_response = self.text_task(current_prompt)
              character_response = character_response.replace("\n", "")

          self.add_history(transcribed_text, character_response, character_name, player_name)
          return character_response

      def send_message(self, role, message):
          emotion = 'neural'
          player_name = 'Lord'
          search_type = 'conversation'
          game_name = 'CyberPunk_2047'

          std_out = sys.stdout
          io_out = StringIO()
          sys.stdout = io_out
          response = self.chat_side_character(game_name, message, player_name, role, emotion, search_type)
          debug_log = io_out.getvalue()
          sys.stdout = std_out

          return response, debug_log

