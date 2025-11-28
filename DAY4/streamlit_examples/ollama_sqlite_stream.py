import sqlite3
import streamlit as st
import ollama

# 1. Set up SQLite database and store sample documents
def setup_database():
    conn = sqlite3.connect('myfile.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY,
                        content TEXT
                      )''')
    sample_documents = [
        ("The first letter is alpha.",),
        ("The second letter is beta.",)
    ]
    cursor.executemany("INSERT INTO documents (content) VALUES (?)", sample_documents)
    conn.commit()
    conn.close()

# 2. Function to retrieve documents based on query
def retrieve_documents(query):
    conn = sqlite3.connect('myfile.db')
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM documents WHERE content LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    conn.close()
    return [result[0] for result in results]

# 3. Function to generate a response using Ollama
def generate_response(query, documents):
    context = "\n".join(documents)
    prompt = f"Query: {query}\n\nContext:\n{context}\n\nResponse:"
    response = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": prompt}])
    return response['message']

# 4. Complete flow: Query input, document retrieval, and response generation
def handle_query(query):
    retrieved_docs = retrieve_documents(query)
    if retrieved_docs:
        response = generate_response(query, retrieved_docs)
        return response
    else:
        return "No relevant documents found."

# Streamlit UI
def main():
    st.title("Document Query and Response Generator")
    st.write("This app retrieves documents from a local SQLite database based on your query, and generates a response using Ollama.")

    # Initialize the database
    setup_database()

    # Get user input
    query = st.text_input("Enter your query:")

    if query:
        st.write(f"Searching for documents related to: **{query}**")

        # Handle the query and generate the response
        response = handle_query(query)

        # Display the response
        st.write(f"**Response:** {response}")

if __name__ == "__main__":
    main()
