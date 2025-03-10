import requests
import streamlit as st
import json, ast
from openai import OpenAI
from dotenv import load_dotenv  
import os
import re  
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langgraph.graph import START, StateGraph,MessagesState,END
from typing_extensions import List, TypedDict
#from langchain_core.messages import HumanMessage
#from langchain_core.messages import SystemMessage, trim_messages
#from langchain_core.messages.base import BaseMessage
# Load and chunk contents of the blog
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_community.document_loaders import DirectoryLoader
from langgraph.checkpoint.memory import MemorySaver
import tempfile
import json

# 加载.env文件  
load_dotenv("./1.env")  



embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)


vector_store = InMemoryVectorStore(embeddings)  
with st.sidebar:
    uploaded_files = st.file_uploader("Upload files", type=("txt", "md","pdf"), accept_multiple_files=True)

st.title("Agentic RAG with QwQ 32B")
documents=[]
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name  
        save_path = os.path.join(os.getcwd(), file_name)  # Save to current directory  
        with open(save_path, "wb") as f:  
            f.write(uploaded_file.getvalue())  
        # 加载文档
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(save_path)
        else:
            loader = TextLoader(save_path)
        documents += loader.load()

    
    
    
    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    
    
    # Index chunks
    _ = vector_store.add_documents(documents=split_docs)


model=os.getenv("model")
key=os.getenv("key")
url=os.getenv("url")
#"http://localhost:11434/v1"
#client = OpenAI(api_key=os.environ['api_key'], base_url=os.getenv("url")) 


client = OpenAI(api_key=key, base_url=url) 
messages=[]

sysmesg = {"role":"system","content":'''You always query information from the function of fetch_related_content. And alway find answer from its result. Please use the language of the question to give answer!
(Note if you cannot find any answer please said you don't know)
'''}
fun_query_kb_desc = {
    "type": "function",
    'function':{
        'name': 'query_knowledge',
        'description': '查询知识库,输入查询内容，返回相关的若干个结果。',
        'parameters': {
            'type': 'object',
            'properties': {
                'query_content': {
                    'type': 'string',
                    'description': '查询内容'
                },
            },
            'required': ['query_content']
        }
    }
}
def query_knowledge(query_content):
    retrieved_docs = vector_store.similarity_search(query_content)
    
    docs_content="Search result:\n\n".join(f"""[Related file name]: {doc.metadata["source"]}
[file chunk content begin]
{doc.page_content}
[file chunk content end]"""  for doc in retrieved_docs)
    return docs_content
    
def getTools():
    return [fun_query_kb_desc]

def generate_text(prompt,think_handle=None,content_handle=None):
    global messages
    
    messages = []
    
    for msg in st.session_state.messages[-10:]:
        print(msg)
        if msg["role"]=="user":
            messages.append({ "role": "user","content": msg["content"]})
        elif msg is not None and msg["content"] is not None:
            messages.append({ "role": "assistant", "content":msg["content"]})
            
    tools=getTools()
    cont = run_conversation(messages,tools,think_handle,content_handle)
    print(cont)
    return cont["content"]
    
def Get_Chat_Deployment():
    deploymentModel = os.environ["Azure_OPENAI_Chat_API_Deployment"]    
    return deploymentModel

def getLLMResponse(messages,tools,stream=True):
    i=20
    messages_ai = messages[-i:]
    while 'role' in messages_ai[0] and messages_ai[0]["role"] == 'tool':
        i+=1
        messages_ai = messages[-i:]
    
    response = client.chat.completions.create(
        model="Qwen/QwQ-32B",
        messages=[sysmesg]+messages_ai,
        temperature=0.6,
        max_tokens=2000,
        tools=getTools(),
        tool_choice="auto", 
        stream=stream
    )
    return response#.choices[0].message


def get_content_think_result(text):
    """
    截取字符串中在 "</think>" 之后的内容。

    :param text: 输入的字符串
    :return: 返回 "</think>" 之后的内容，如果不存在则返回空字符串
    """
    # 使用 split() 方法分割字符串
    parts = text.split("</think>")

    # 如果分割后有多于一个部分，说明找到了 </think>
    if len(parts) > 1:
        # 取分割后的第二部分，即 </think> 之后的内容
        return parts[0],parts[1]
    else:
        # 如果没有找到 </think>，返回空字符串
        return parts[0],""
        
def run_conversation(messages,tools,think_handle,content_handle):
    # Step 1: send the conversation and available functions to the model
    response_message = getLLMResponse(messages,tools)
    content=''
    reasoning_content=''
    function_list=[]
    index=0
    for chunk in response_message:
        if chunk:
            chunk_message =  chunk.choices[0].delta
            #print(chunk_message)
            if chunk_message.content:
                content+=chunk_message.content
                print(chunk_message.content,end="")
                if content_handle:
                    content_handle(content)
                    
            if chunk_message.reasoning_content and think_handle:
                print(chunk_message.reasoning_content,end="")
                reasoning_content+=chunk_message.reasoning_content
                think_handle(reasoning_content)
                
            if chunk_message.tool_calls:
                
                for tool_call in chunk_message.tool_calls:
                    
                    if len(function_list)<tool_call.index+1:
                        function_list.append({'name':'','args':'','id':tool_call.id})
                    if tool_call and tool_call.function.name:
                        function_list[tool_call.index]['name']+=tool_call.function.name
                    if tool_call and tool_call.function.arguments:
                        function_list[tool_call.index]['args']+=tool_call.function.arguments
                        

    print(function_list)
    
    if len(function_list)>0:
        findex=0
        tool_calls=[]
        temp_messages=[]
        for func in function_list:
            function_name = func["name"]
            print(function_name)
            function_args = func["args"]
            toolid=func["id"]
            if function_name !='':
                print(f'⏳Call {function_name}...')
                function_to_call = globals()[function_name]
                function_args = json.loads(function_args)
                print(f'⏳Call params: {function_args}')
                
                function_response = function_to_call(**function_args)
                print(f'⏳Call internal function done! ')
                print("执行结果：")
                print(function_response)
                tool_calls.append({"id":toolid,"function":{"arguments":func["args"], "name":"query_knowledge"}, "type":"function","index":findex})
                
                temp_messages.append(
                    {
                    "tool_call_id": toolid,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                    }
                )
                #print(messages)
                findex+=1
                
        messages.append({
                    "role":"assistant",
                    "content":content,
                    "tool_calls":tool_calls,
                })
        for m in temp_messages:
            messages.append(m)
        print("-------------------------")
        print(messages)  
        if think_handle:
            think_handle(content)
        
        return run_conversation(messages,tools,think_handle,content_handle)
    elif content!='' and messages[-1]["role"]=="tool":
        content+="\nAnaylzing and formating..."
        if content_handle:
            content_handle(content)
        #print(content)
        r=getLLMResponse(messages,tools,stream=False)
        c=r.choices[0].message.content
        think,content = get_content_think_result(c)
        messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content":think
                }
            )
        if content_handle:
            content_handle(content)
        if think_handle:
            think_handle(think)
        print(messages)
        return messages[-1]
    elif content!='' and messages[-1]["role"]!="tool":
        messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
        print(messages)
        return messages[-1]
        
def extract_and_remove_think(content):  
    # 将 <think> 标签及其内容替换为空  
    cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)  
    # 去除多余的空格  
    cleaned_content = cleaned_content.strip()  
      
    return cleaned_content  
    

        

    

if "messages" not in st.session_state:
    st.session_state.messages = []
    

    
for message1 in st.session_state.messages:
    with st.chat_message(message1["role"]):
        c=message1["content"]
        think_content = re.findall(r'<think>(.*?)</think>', c, flags=re.DOTALL)  
        main_content = re.sub(r'<think>.*?</think>', '', c, flags=re.DOTALL) 
        if message1["role"]=="assistant":
            with st.expander("Think", expanded=False):
                if think_content and len(think_content)>0:
                    st.markdown(think_content[0])
                if "reasoning_content" in message1 and len(message1["reasoning_content"])>0:
                    st.markdown(message1["reasoning_content"])
        st.markdown(main_content)
        
        

def writeReply(cont,exp,msg):
    main_content = re.sub(r'<think>.*?</think>', '', msg, flags=re.DOTALL) 
    if main_content.startswith("<think>"):
        exp.write(main_content[7:])
    else:
        cont.write(main_content)
def writeThinkReply(exp,msg):
    main_content = re.sub(r'<think>.*?</think>', '', msg, flags=re.DOTALL) 
    exp.write(main_content[7:])


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.expander("Think", expanded=False):
            exp=st.empty()
            exp.text("⏳...")
        p=st.empty()
        p.text("⏳...")
        res = generate_text(prompt,lambda x:writeThinkReply(exp,x),lambda x:writeReply(p,exp,x))
        print(res)
        st.session_state.messages.append({"role": "assistant", "content": res})