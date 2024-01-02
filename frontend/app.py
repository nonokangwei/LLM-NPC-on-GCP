import gradio as gr
import requests
import json
import os

cb_model_choices = ["text_palm","chat_palm"]
cb_prompt_type_choices = ["native", "react", "conversational"]
kb_model_choices = ["palm", "enterprice_search"]
kb_vdb_collection_ops = None
backend_url = os.getenv('BACKEND_URL','https://nuvabackend-vnme4je7pq-uc.a.run.app')
port = int(os.getenv('PORT', 9000))

def vdb_collection_list():
    path = backend_url+'/collection'
    params = {'vectordbtype': 'matchengine'}
    headers = {'accept': 'application/json'}
    res = requests.get(path, params=params,headers=headers)
    result = res.json()['collections']
    print(result)
    return gr.Dropdown.update(choices=result),gr.Dropdown.update(choices=result)

def chatbot_model_onchange(model, prompt_type, collection):
    path = backend_url+'/chatbot/changemodel'
    headers = {'accept': 'application/json','Content-Type': 'application/json'}
    data = {"model_name": model,"prompt_type": prompt_type, "collection": collection}
    resp = requests.post(url=path,headers=headers, json=data)
    print(resp.text)
    
def hero_lists_get():
    path = backend_url+'/chatbot/list/role'
    headers = {'accept': 'application/json'}
    resp = requests.get(url=path,headers=headers)
    heros = json.loads(resp.text)
    result = heros['roles']
    return gr.update(choices=result)
    
    
def chatbot_respond(role, message, chat_history):
    path = backend_url+'/chatbot/sendmessage'
    headers = {'accept': 'application/json','Content-Type': 'application/json'}
    data = {"role":role ,"message": message}
    resp = requests.post(url=path,headers=headers, json=data)
    bot_message = json.loads(resp.text)['response_message']
    chat_history.append((message, bot_message))
    debug_message = json.loads(resp.text)['debug_message']
    print(debug_message)
    return "", chat_history, debug_message

def chatbot_reset(role):
    path = backend_url+'/chatbot/resetmessage'
    headers = {'accept': 'application/json','Content-Type': 'application/json'}
    data = {'role': role}
    resp = requests.post(url=path,headers=headers, json=data)
    print(resp)
    console = None
    return gr.update(value=console), gr.update(value=console)
    

def chatbot_debug_control(evt: gr.SelectData):
    if evt.selected == True:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


"""
Knowledge Bot implementation 
"""

def kb_vdb_collection_list():
    path = backend_url+'/collection'
    params = {'vectordbtype': 'matchengine'}
    headers = {'accept': 'application/json'}
    res = requests.get(path, params=params,headers=headers)
    result = res.json()['collections']
    return gr.Dropdown.update(choices=result)
    

def kb_model_change(model, prompt_type, collection):
    path = backend_url+'/kbbot/changemodel'
    headers = {'accept': 'application/json','Content-Type': 'application/json'}
    params = {'vectordbtype': 'matchengine'}
    data = {"model_name": model,"prompt_type": prompt_type, "collection": collection}
    resp = requests.post(url=path, params=params, headers=headers, json=data)
    print(resp.text)
    return resp.text


def kb_respond(message, chat_history):
    path = backend_url+'/kbbot/sendmessage'
    headers = {'accept': 'application/json','Content-Type': 'application/json'}
    data = {"message": message}
    resp = requests.post(url=path,headers=headers, json=data)
    bot_message = json.loads(resp.text)['response_message']
    chat_history.append((message, bot_message))
    debug_message = json.loads(resp.text)['debug_message']
    print(debug_message)
    return "", chat_history, debug_message



def vdb_coll_change():
    path = backend_url+'/collection'
    params = {'vectordbtype': 'matchengine'}
    headers = {'accept': 'application/json'}
    res = requests.get(path, params=params,headers=headers)
    result = res.json()['collections']
    return gr.Dropdown.update(choices=result)


    
def knowledge_bot_debug_control(evt: gr.SelectData):
    if evt.selected == True:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    

"""
Vector DB manapulation 
"""

def collections_update():
    new_options = utility.list_collections()
    print(new_options)
    return gr.Dropdown.update(choices=new_options)


def db_change(option):
    if option == "Match Engine":
        path = backend_url+'/collection'
        params = {'vectordbtype': 'matchengine'}
        headers = {'accept': 'application/json'}
        res = requests.get(path, params=params,headers=headers)
        result = res.json()['collections']
    elif option == "Milvus":
        result = ["Mil_collection_1", "Mil_collection_2"]
    return gr.Dropdown.update(choices=result)


def insert_methods_select(evt: gr.SelectData):
    if evt.value in ["streaming"]:
        return gr.update(visible=True),  gr.update(visible=False)
    elif evt.value == "batch":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        pass

def delete_methods_select(evt: gr.SelectData):
    if evt.value in ["streaming"]:
        return gr.update(visible=True),  gr.update(visible=False)
    elif evt.value == "batch":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        pass
  

def me_query(collection, doc, k):
    url = backend_url+'/search'
    params = {'vectordbtype': 'matchengine'}
    headers = {'accept': 'application/json','Content-Type': 'application/json'}
    data = {"doc": doc, "k":k, "collection":collection}
    resp = requests.post(url, params=params, json=data, headers=headers)
    return resp.text



def me_insert(option,doc,location,collection):
    if option == "streaming":
        path = backend_url+'/doc'
        params = {'vectordbtype': 'matchengine'}
        headers = {'accept': 'application/json','Content-Type': 'application/json'}
        data = {"doc": doc,"collection":collection}
        res = requests.post(url=path, params=params, headers=headers, json=data)
        return res.text
    elif option == "batch":
        path = backend_url+'/docsbatch'
        params = {'vectordbtype': 'matchengine'}
        headers = {'accept': 'application/json','Content-Type': 'application/json'}
        data = {"doc_location": location,"collection":collection}
        res = requests.post(url=path, params=params, headers=headers, json=data)
        return res.text

def me_delete(option, doc, location, collection):
    if option == "streaming":
        path = backend_url+'/doc'
        headers = {'accept': 'application/json','Content-Type': 'application/json'}
        params = {'vectordbtype': 'matchengine'}
        data = {"doc": doc,"collection":collection}
        res = requests.delete(url=path, params=params, headers=headers, json=data)
        return res.text
    elif option == "batch":
        path = backend_url+'/docsbatch'
        params = {'vectordbtype': 'matchengine'}
        headers = {'accept': 'application/json','Content-Type': 'application/json'}
        data = {"doc_location": location,"collection":collection}
        res = requests.delete(url=path, params=params, headers=headers, json=data)
        return res.text
        
        


with gr.Blocks() as demo:
    with gr.Tab("chatbot") as cb_tabs:
        with gr.Row():
            chatbot_model = gr.Dropdown(cb_model_choices,label="model_name",scale=6)
            cb_prompt_type = gr.Dropdown(cb_prompt_type_choices,label="prompt_type",scale=6)
            cb_vdb_collections = gr.Dropdown(label="Vector DB collections",interactive=True,scale=6)
            cb_model_change_bt = gr.Button(value="change model",min_width=1,scale=2)
        heros = gr.Radio(choices=[""],label="hero lists")
        heros_get = gr.Button(value="refresh", label="click to refresh the hero list")
        chatbot = gr.Chatbot(label="message history")
        chatbot.style(height=500)
        chat_msg = gr.Textbox(label="message")
        chat_clear = gr.Button("Clear")
        chatbot_debug_check = gr.Checkbox(label="Debug", info="display debug console")
        chabot_debug_console = gr.Textbox(interactive=True, visible=False,label="Debug Console")
        
        

        heros_get.click(hero_lists_get, None, heros)
        cb_model_change_bt.click(chatbot_model_onchange, [chatbot_model,cb_prompt_type,cb_vdb_collections], None)
        chat_msg.submit(chatbot_respond, [heros, chat_msg, chatbot], [chat_msg, chatbot, chabot_debug_console])
        chat_clear.click(chatbot_reset, heros, [chatbot,chabot_debug_console])
        chatbot_debug_check.select(chatbot_debug_control, None, chabot_debug_console)
        
    with gr.Tab("knowledge bot") as kb_tabs:
        with gr.Row():
            knowledge_bot_model = gr.Dropdown(kb_model_choices, label="model_name", max_choices=5,scale=6)
            
            kb_prompt_type = gr.Dropdown(visible=False, label="prompt_type",)
            kb_vdb_collections = gr.Dropdown(label="Vector DB collections",interactive=True,scale=6)
            kb_model_change_bt = gr.Button(value="change model",scale=2, min_width=1)
        knowledge_bot = gr.Chatbot(label="message history")
        knowledge_bot.style(height=500)
        knowledge_bot_msg = gr.Textbox(label="message")
        knowledge_bot_clear = gr.Button("Clear")
        knowledge_debug_check = gr.Checkbox(label="Debug", info="display debug console")
        knowledge_debug_console = gr.Textbox(interactive=True, visible=False,label="Debug Console")
        knowledge_debug_check.select(knowledge_bot_debug_control, None, knowledge_debug_console)
        kb_model_change_bt.click(kb_model_change, [knowledge_bot_model,kb_prompt_type, kb_vdb_collections], None)
        knowledge_bot_msg.submit(kb_respond, [knowledge_bot_msg, knowledge_bot], [knowledge_bot_msg, knowledge_bot, knowledge_debug_console])
        knowledge_bot_clear.click(lambda: None, None, knowledge_bot, queue=False)
    with gr.Tab("vec-db query and upsert") as vdb_tabs:
        with gr.Row():
            dbs = gr.Dropdown(["Match Engine", "Milvus"], label="Vector DBs", interactive=True, scale=6)
            collections = gr.Dropdown(None, label="Vector DB collections", interactive=True, scale=6)
            collections_bt = gr.Button("refresh",scale=1,min_width=1)

        with gr.Row():
            vector_query_inp = gr.Textbox(placeholder="input your text to search", label="Vertor search", scale=6)
            topk = gr.Slider(1, 10, value=1, step=1, label="Topk", info="Choose between 1 and 10", scale=6)
            search_submit = gr.Button(value="search", scale=1,min_width=1)

        with gr.Row():
            with gr.Column():
                insert_methods = gr.Radio(["streaming", "batch"], label="insert options")
                s_insert_inp = gr.Textbox(label="streaming insert input",visible=False)
                b_insert_inp = gr.Textbox(label="batch insert input",visible=False)
                insert_submit = gr.Button(value="insert")
            with gr.Column():
                delete_methods = gr.Radio(["streaming", "batch"], label="delete options")
                s_delete_inp = gr.Textbox(label="streaming delete input",visible=False)
                b_delete_inp = gr.Textbox(label="batch delete input",visible=False)
                delete_submit = gr.Button(value="delete")
                
        result = gr.Textbox(label="result output", max_lines=20)
        
        
        dbs.select(db_change, dbs, collections)
        collections_bt.click(collections_update, None, None)
        
        insert_methods.select(insert_methods_select, None, [s_insert_inp,b_insert_inp])
        delete_methods.select(delete_methods_select, None, [s_delete_inp,b_delete_inp])
        search_submit.click(me_query,[collections,vector_query_inp,topk],result)
        insert_submit.click(me_insert, [insert_methods,s_insert_inp,b_insert_inp,collections],result)
        delete_submit.click(me_delete, [delete_methods,s_delete_inp, b_delete_inp,collections], result)
        demo.load(vdb_collection_list,None,[cb_vdb_collections,kb_vdb_collections])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=port)
