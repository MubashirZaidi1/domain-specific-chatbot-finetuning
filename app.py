from flask import Flask, request, render_template, jsonify
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

app = Flask(__name__)

# Declare LangChain components as global variables
embeddingfunc = None
db3 = None
llm = None
chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    global embeddingfunc, db3, llm, chain

    # Get the selected model from the POST request
    model = request.json['model']

    # Initialize LangChain components with the selected model
    embeddingfunc = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    db3 = Chroma(persist_directory="./chroma_db", collection_name="local-rag", embedding_function=embeddingfunc)
    llm = ChatOllama(model=model)

    # Configure LangChain chain based on the selected model
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=f"""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {{question}}""",
    )

    retriever = MultiQueryRetriever.from_llm(db3.as_retriever(), llm, prompt=QUERY_PROMPT)

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return jsonify({'message': f'Model selected: {model}'})

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        user_question = request.json['user_question']
        if chain:
            output = chain.invoke(user_question)
            return jsonify({'output': output})
        else:
            return jsonify({'error': 'Model not initialized'})
@app.route('/chat')
def chat():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
