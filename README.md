# Youtube-Chatbot

## Description:
**Youtube-Chatbot** is a cutting-edge chatbot designed to engage users in discussions about their desired video content on YouTube. It harnesses the power of **Mistral-7B**, a highly efficient 7.3B parameter large language model (LLM) developed by Mistral AI, and the Retrieval-Augmented Generation (RAG) methodology. The chatbot's vector store database relies on Chroma DB, providing a seamless experience for users to chat about their target video content.

## Key Features:

+ **Mistral-7B Language Model**: Employs the efficient and versatile Mistral-7B LLM for natural language understanding and generation.
+ **RAG Methodology**: Utilizes the Retrieval-Augmented Generation (RAG) approach to combine retrieval from a vast corpus of text with generative text generation, ensuring informative and engaging conversations.
+ **Chroma DB Vector Store**: Utilizes Chroma DB as a vector store database for efficient storage and retrieval of vector embeddings, optimizing performance.
+ **Youtube-Transcript-API**: Integrates the **Youtube-Transcript-API** to extract and analyze video transcripts, enabling the chatbot to engage in discussions about specific video content.
+ **Langchain Framework**: Utilizes the **Langchain framework's RetrievalQA** capabilities to combine various components seamlessly and provide meaningful responses.

## Use Cases:

+ Content Recommendations
+ Video Discussion and Analysis
+ Querying Video Transcripts
+ Entertainment and Information


## Benefits:

+ Personalized Video Content Discussions
+ Accurate and Informative Responses
+ Efficient and User-Friendly Interface
+ Accessible for Customization
+ Seamless Integration with YouTube


## Troubleshooting : 

+ If you see the following error:
```
Traceback (most recent call last):
File "", line 1, in
File "/transformers/models/auto/auto_factory.py", line 482, in from_pretrained
config, kwargs = AutoConfig.from_pretrained(
File "/transformers/models/auto/configuration_auto.py", line 1022, in from_pretrained
config_class = CONFIG_MAPPING[config_dict["model_type"]]
File "/transformers/models/auto/configuration_auto.py", line 723, in getitem
raise KeyError(key)
KeyError: 'mistral'
```

**Installing transformers from source should solve the issue:**

```
pip install git+https://github.com/huggingface/transformers
```

## Usage : 

```
pip install -r requirements.txt
streamlit run app.py
```

![Screenshot at 2023-10-07 14-59-49](https://github.com/Kirouane-Ayoub/Youtube-Chatbot-APP/assets/99510125/2889cdd4-2481-4279-8555-ff7213d03d89)


+ **Developed By Kirouane Ayoub**
