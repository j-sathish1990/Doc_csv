import os
os.environ["OPENAI_API_KEY"] = "sk..7d"


from langchain.embeddings  import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain import OpenAI, VectorDBQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import pandas as pd

from flask import Flask, request, jsonify
app = Flask(__name__)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
embeddings = OpenAIEmbeddings()
llm = OpenAI()

@app.route('/query', methods = ['POST'])

def query_1():
  data = request.get_json(force = True)

  if 'df_1' not in data or 'query' not in data:
    return jsonify({'error': 'No dataframe or query'}), 400

  #get dataframe

  df_1 = pd.DataFrame(data['df_1'])
  query = data['query']

  df_1.to_csv('df_1.txt', index = False, header = False)

  with open("df_1.txt", encoding = 'utf-8') as f:
    df_1_text = f.read()

  texts = text_splitter.split_text(df_1_text)
  vectorstore = FAISS.from_texts(texts, embeddings)
  qa = VectorDBQA.from_chain_type(llm = OpenAI(), chain_type='stuff', vectorstore= vectorstore)
  #response
  response = qa.run(query)

  return jsonify({'response':response})

if __name__ == '__main__':
    app.run(port= 5000, debug=True)


