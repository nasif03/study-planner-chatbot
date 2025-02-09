import argparse
import re
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
    model = OllamaLLM(model="llama3.2")
    return model

def query_rag(query_text: str, model: OllamaLLM, show_thoughts: bool, show_sources: bool, show_prompt: bool):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    if show_prompt:
        print("\n=== PROMPT ===")
        print(prompt)

    response_text = model.invoke(prompt)

    # Parse the response into thoughtProcess and response
    thought_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if thought_match:
        thoughtProcess = thought_match.group(1).strip()
        response = response_text[thought_match.end():].strip()
        response = re.sub(r'^Response:\s*', '', response)
    else:
        thoughtProcess = ""
        response = response_text.strip()

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    if show_thoughts:
        print("\n\n=== THOUGHT PROCESS ===")
        print(thoughtProcess)
    
    print("\n\n=== FINAL RESPONSE ===")
    print(response)
    
    if show_sources:
        print("\n\nSources:", sources)
    
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, nargs="?", help="The query text.")
    args = parser.parse_args()
    
    model = load_model()
    
    # Ask user independently if they want to print thought process and sources
    show_thoughts = input("Do you want to print the thought process? (yes/no): ").strip().lower() == "yes"
    show_sources = input("Do you want to print the sources? (yes/no): ").strip().lower() == "yes"
    show_prompt = input("Do you want to print the prompt? (yes/no): ").strip().lower() == "yes"
    if args.query_text:
        query_rag(args.query_text, model, show_thoughts, show_sources)
    else:
        while True:
            query_text = input("\nEnter your query (or type 'exit' to quit): ")
            if query_text.lower() == 'exit':
                break
            query_rag(query_text, model, show_thoughts, show_sources, show_prompt)

if __name__ == "__main__":
    main()
