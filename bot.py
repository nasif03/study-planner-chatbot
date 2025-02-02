import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question below, use the given context if necessary:

{context}

---

Answer the question, rely on the context given above if needed and do not refer to the context in your response. The user providing the question is not privy to the context: {question}
"""

def load_model():
    # Load model once at the start.
    model = OllamaLLM(model="llama3.2")
    return model

def query_rag(query_text: str, model: OllamaLLM):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("\nData retrieved\n")
    
    # Get response from the loaded model.
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}\n"

    # Print data retrieval status and formatted response
    print(formatted_response)
    
    return response_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, nargs="?", help="The query text.")
    args = parser.parse_args()
    
    # Load model once.
    model = load_model()

    if args.query_text:
        # If query_text is passed as an argument, use it for the first query.
        query_rag(args.query_text, model)
    else:
        # If no query_text is passed, enter interactive mode.
        while True:
            query_text = input("\nEnter your query (or type 'exit' to quit): ")
            if query_text.lower() == 'exit':
                break
            query_rag(query_text, model)

if __name__ == "__main__":
    main()
