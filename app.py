import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import YoutubeLoader

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import  pipeline
import torch
import transformers
from torch import cuda
st.set_page_config(page_title='YoutubeChatbot')
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
model_name = "bn22/Mistral-7B-Instruct-v0.1-sharded"
st.header(":hand: Welcome To Youtube Chatbot : ")
st.info("This chatbot Uses all-MiniLM-L6-v2 as an embedding model and Mistral-7B LLM.")
st.write("""
1 /  You can change temperature, top-ranked, and top_p values from the slider.\n
2 /  Entre your  Youtube Video ID.\n
3 /  Start chatting . \n
""")
with st.sidebar : 
    st.image("icon.png")
    temperature = st.sidebar.slider("Select your temperature value : " ,min_value=0.1 ,
                                 max_value=1.0 ,
                                   value=0.5)
    top_p = st.sidebar.slider("Select your top_p value : " ,min_value=0.1 ,
                           max_value=1.0 , 
                           value=0.5)
    k_n = st.number_input("Enter the number of top-ranked retriever Results:" ,
                             min_value=1 , max_value=4 , value=1)
    
    you_vid_id = st.text_input("Entre your  Youtube Video ID. ex : 0phRae42lqY" , value="0phRae42lqY")

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
with st.spinner("Downloading the Mistral 7B  and all-MiniLM-L6-v2....") : 
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2
    )

local_llm = HuggingFacePipeline(pipeline=pipe)


with st.spinner("Loading Video content ..") : 
    loader = YoutubeLoader()
    documents = loader.load(you_vid_id)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
                documents,
                embedding=embed_model,
                persist_directory="DB"
            )

qa_chain = RetrievalQA.from_chain_type(
    chain_type="stuff" , 
    llm=local_llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": k_n}),
    return_source_documents=False,
    verbose=False
)





def run_qa(qs) : 
    return qa_chain.run(qs)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi :hand: Im Your open-source Youtube Chatbot, we can chat about Your YouTube video  content"}]
# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_qa(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)