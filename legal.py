import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# ✅ Set your Together API key
os.environ['TOGETHER_API_KEY'] = 'a0ea6b08429661d592dab6beca8f9f95437f7efdbe794e254f4710d04b75e89d'

# ✅ Prompt Template for Chatbot
prompt_template = """[INST]
This is a chat template and as a Law School Assistant Chatbot, your primary objective is to help 8th–12th grade and law students understand legal and constitutional topics in simple, clear terms.

Your role:
- Use the provided knowledge base for accurate answers.
- Explain rights, duties, IPC sections, and procedures with relatable examples.
- Stick to the context; avoid adding your own questions or assumptions.
- Be brief, clear, and informative.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
[/INST]"""

# ✅ Load HuggingFace Embeddings (text model)
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={
        "trust_remote_code": True,
        "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"
    }
)

# ✅ Load FAISS Vector DB (Must exist: folder 'ipc_vector_db')
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ✅ Prompt Template Setup
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

# ✅ LLM using TogetherAI
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=os.environ['TOGETHER_API_KEY']
)

# ✅ Conversation Memory Buffer
memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    return_messages=True
)

# ✅ Conversational Retrieval QA Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# ✅ Reset memory
def reset_conversation():
    memory.clear()

# ✅ Main chatbot function
def chatbot(query):
    if query.lower().strip() == "reset":
        reset_conversation()
        return "Conversation reset."
    try:
        result = qa.invoke({"question": query})
        return result["answer"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ Optional CLI usage
if __name__ == "__main__":
    print("📚 Law School Chatbot Ready. Type 'reset' to clear memory. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == "exit":
            print("👋 Goodbye!")
            break
        reply = chatbot(user_input)
        print("Bot:", reply)