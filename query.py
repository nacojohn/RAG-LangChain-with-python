import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import gradio as gr


load_dotenv(override=True)
llm = ChatAnthropic(model=os.getenv("LLM_MODEL", "claude-2"), max_tokens=500)

embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
db_name = os.getenv("CHROMA_DB_NAME", "chroma_db")

# embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
embeddings = OpenAIEmbeddings(model=embedding_model)
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)


prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Cite sources as [source, page]. \n\n
Context:\n{context}

"""
prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    ("human", "{question}")
])

def test_retrieval(query):
    """
    Test retrieval of documents based on a query.

    Args:
        query (str): The query string to search for relevant documents.
        embedding_model (str): The embedding model to use for generating embeddings.
        db_name (str): The name of the Chroma database to use.
    Returns:
        list: A list of retrieved Document objects.
    """

    results = vectorstore.similarity_search_with_score(query, k=4)
    for doc, score in results:
        print(f"[{score:.3f}] {doc.metadata.get('source')} p.{doc.metadata.get('page')}")
        print(doc.page_content[:200], "\n")

    return results

def ask(question, k=4):
    """
    Ask a question to the AI assistant using the provided context.

    Args:
        question (str): The question to ask.
        k (int): The number of context pieces to retrieve.
    Returns:
        str: The answer from the AI assistant.
    """
    
    # Retrieve relevant context from the vector store
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([f"[{doc.metadata['source']}, page {doc.metadata['page']}]: {doc.page_content}" for doc in docs])

    # Format the prompt with the retrieved context and the question
    formatted_prompt = prompt.format_messages(context=context, question=question)

    # Get the answer from the LLM
    answer = llm.invoke(formatted_prompt)

    return answer.text

if __name__ == "__main__":
    gr.Interface(fn=ask,
             inputs=[gr.Textbox(label="Enter your question")],
             outputs=[gr.Textbox(label="Answer")],
             title="AI Assistant",
             description="Ask questions based on the ingested documents.").launch()
    # while True:
    #     user_input = input("Enter your question (or 'exit' to quit): ")
    #     if user_input.lower() == 'exit':
    #         break
    #     answer = ask(user_input)
    #     print(f"Answer: {answer}\n")