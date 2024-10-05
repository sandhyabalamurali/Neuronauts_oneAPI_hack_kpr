<<<<<<< HEAD
from flask import Flask, request, jsonify, send_from_directory
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from embedding import get_embedding_function
from groq import Groq
import os
import easyocr
import json
from googleapiclient.discovery import build

# Set up your YouTube API key here
YOUTUBE_API_KEY = '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'

os.environ['GROQ_API_KEY'] = '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'

app = Flask(__name__)

CHROMA_PATH = "chroma2"
UPLOAD_FOLDER = "uploads2"  # Folder to save uploaded images
HISTORY_FILE = "conversation_history.json"  # File to store conversation history

# Ensure the folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

PROMPT_TEMPLATE = """
Based on the full conversation history and report text, answer the question briefly:

Full Conversation History:
{conversation_history}

Report Context:
{context_text}
{context_text2}

---

Answer this question briefly only based on the above report context and full conversation history. If not found, don't tell anything. Don't use your prior knowledge. Question: {question}
"""

# Load conversation history from a file if it exists
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        conversation_history = json.load(f)
else:
    conversation_history = {}

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def save_conversation_history():
    """Save the conversation history to a file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(conversation_history, f)

def perform_ocr(image_path):
    results = reader.readtext(image_path)
    detected_text = []
    for (bbox, text, _) in results:
        detected_text.append({
            "text": text,
            "bounding_box": bbox
        })
    return detected_text

def ocr_local_image_full(image_path):
    detected_text = perform_ocr(image_path)
    report_text = " ".join([item['text'] for item in detected_text])
    return report_text

def search_youtube_videos(query):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    request = youtube.search().list(
        part='snippet',
        q=query,
        type='video',
        maxResults=5  # Adjust the number of results as needed
    )
    response = request.execute()

    videos = []
    for item in response.get('items', []):
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({
            'title': title,
            'url': url
        })
    
    return videos

def query_retro_rag(user_id: str, query_text: str) -> str:
    # Define keywords that trigger YouTube video recommendations
    keywords = ["diet", "exercise", "tips", "care", "nutrition", "health", "advice", "pregnancy", "motherhood"]
    
    # Check if any keyword is in the user's query
    if any(keyword.lower() in query_text.lower() for keyword in keywords):
        # If a keyword is found, search for YouTube videos
        recommended_videos = search_youtube_videos(query_text)
        video_response = "\n".join([f"{video['title']}: {video['url']}" for video in recommended_videos])
        return video_response  # Return the video links as a string
    
    # If no keywords are matched, proceed with normal response generation
    # Step 1: First retrieval pass based on the initial query
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results_first_pass = db.similarity_search_with_score(query_text, k=5)

    # Combine the context from the first retrieval
    context_text_1 = "\n\n---\n\n".join([doc.page_content for doc, _ in results_first_pass])

    # Get the user's entire conversation history
    history = conversation_history.get(user_id, [])
    history.append({"role": "user", "content": query_text})

    # Construct the full conversation history string
    full_conversation = "\n".join([f"{item['role']}: {item['content']}" for item in history])

    # Step 2: Generate initial response based on the first retrieval context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    initial_prompt = prompt_template.format(
        conversation_history=full_conversation,
        question=query_text,
        context_text=context_text_1,
        context_text2=""  # Initially empty for the second pass
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    initial_chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI-powered pregnancy care chatbot. Ask the user how many months they are into their pregnancy, inquire about any inconveniences they are facing, and provide diet plan suggestions and relevant advice. If any other questions are asked, do not answer and don't recommend specific clinics."}] + history,
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        top_p=0.90,
    )

    initial_response_text = initial_chat_completion.choices[0].message.content

    # Step 3: Use the initial response to perform a second retrieval (refining the query)
    refined_query = query_text + " " + initial_response_text  # Combine query and response
    results_second_pass = db.similarity_search_with_score(refined_query, k=5)

    # Combine the context from the second retrieval
    context_text_2 = "\n\n---\n\n".join([doc.page_content for doc, _ in results_second_pass])

    # Step 4: Generate the final response based on both retrieval contexts
    final_prompt = prompt_template.format(
        conversation_history=full_conversation,
        question=query_text,
        context_text=context_text_1,  # Context from the first pass
        context_text2=context_text_2   # Additional context from the second pass
    )

    final_chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI-powered pregnancy care chatbot. Ask the user how many months they are into their pregnancy, inquire about any inconveniences they are facing, and provide diet plan suggestions and relevant advice. If any other questions are asked, do not answer and don't recommend specific clinics."}] + history,
        model="llama-3.1-70b-versatile",
        temperature=0.4,
        top_p=0.90,
    )

    final_response_text = final_chat_completion.choices[0].message.content

    # Step 5: Update conversation history and save it
    history.append({"role": "system", "content": final_response_text})
    conversation_history[user_id] = history
    save_conversation_history()  # Save history after each interaction

    # Return the final formatted response
    return final_response_text

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/chatbot')
def chatbot():
    return send_from_directory('templates', 'chatbot.html')

@app.route('/query', methods=['POST'])
def query_rag_route():  
    data = request.json
    user_id = data.get("user_id", "default_user")
    query_text = data.get("query", "")
    response_text = query_retro_rag(user_id, query_text)
    return jsonify({'response': response_text})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)
    
    # Perform OCR and get the extracted text
    report_text = ocr_local_image_full(image_path)
    os.remove(image_path)  # Clean up the uploaded image file

    # Add instruction to analyze the report
    report_text += "\n \n Analyze this report very briefly and give detailed insights."

    # Perform the chatbot query using the extracted text
    user_id = 'default_user'
    response_text = query_retro_rag(user_id, report_text)
    
    # Return the extracted report text and chatbot response
    return jsonify({
        'report_text': report_text,
        'response': response_text
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    data = request.json
    user_id = data.get("user_id", "default_user")
    if user_id in conversation_history:
        del conversation_history[user_id]
        save_conversation_history()  # Save history after clearing
        message = 'Conversation history cleared.'
    else:
        message = 'No history found to clear.'
    return jsonify({'message': message}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
=======
from flask import Flask, request, redirect, render_template, flash

from signup import signup_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Call the signup_user function (make sure this function is defined/imported)
        signup_user(username, email, password)

        # Redirect to a success page
        return redirect('/signup_success.html')
    
    # Render the signup page
    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> acbfd6d97a3d50af35de90b1ea920d808bcf5877
