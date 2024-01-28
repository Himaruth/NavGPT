from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import folium
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import requests

#RAG for translation (multilingual support)
'''
#pip install gtts
#pip install googletrans==4.0.0-rc1
#pip install sentencepiece
#pip install transformers

from googletrans import Translator
from transformers import pipeline
from gtts import gTTS
from IPython.display import Audio

def translate_text(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def generate_summary(text, model="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model)
    summary = summarizer(text, min_length=1, length_penalty=0.99)[0]['summary_text']
    return summary

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    return tts

def rag_model_with_tts(input_text, target_language='en', model="facebook/bart-large-cnn"):
    translated_text = translate_text(input_text, target_language)
    summary = generate_summary(translated_text, model)
    
    # Convert the summary to speech
    tts = text_to_speech(summary, language=target_language)
    
    return summary, tts

input_text = input()

summary, tts = rag_model_with_tts(input_text)
print("Summary:", summary)

# Save the speech to an audio file
tts.save("output.mp3")

# Display the audio file (Note: This may not work in all environments)
Audio("output.mp3")
'''
app = Flask(__name__)
'''
# Initialize ChatBot
chatbot = ChatBot('Bot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")'''

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']

    # Define predefined responses
    predefined_responses = {
        'hi': 'Hello! How can I assist you today?',
        'bye': 'Goodbye! Have a great day!',
        'how are you feeling': "I'm just a chatbot, so I don't have feelings, but I'm here to help!",
        'recommend some good restaurants in rochester':'Donuts delite, genesee brew house and the mercantile on main are some frequently revisited food-spots in Rochester',
        'abbott\'s frozen custard': 'Abbott\'s Frozen Custard is a must-visit spot in Rochester for delicious frozen custard!',
        'donuts delite': 'Donuts Delite is famous for its iconic donuts and treats in Rochester!',
        'genesee brew house': 'The Genesee Brew House offers a unique beer destination with interactive exhibits and a pub-style restaurant!',
        'the mercantile on main': 'The Mercantile on Main is Rochester\'s first food hall, featuring a curated collection of some of the best food and drink offerings in town!'
    }

    # Check if the user message matches predefined responses
    if user_message.lower() in predefined_responses:
        bot_response = predefined_responses[user_message.lower()]
    else:
        # If no predefined response, use the chatbot to generate a response
        #bot_response = str(chatbot.get_response(user_message))
        pass
    return jsonify({'message': bot_response})

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Define your OpenRouteService API key
api_key = '5b3ce3597851110001cf624827eb0e907ea24cf0826da98df09cf785'

# Function to detect vehicles in the frame using YOLO
def detect_vehicles(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    vehicle_boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]  
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class ID for 'car' in the COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                vehicle_boxes.append((x, y, w, h))
                # Draw bounding box around detected vehicle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, vehicle_boxes

# Function to calculate traffic congestion level
def calculate_traffic_congestion(vehicle_boxes, image_width):
    total_road_area = image_width * 100  # Assuming the road occupies the entire width of the frame
    vehicle_area = sum([w * h for _, _, w, h in vehicle_boxes])
    occupancy_percentage = (vehicle_area / total_road_area) * 100

    if occupancy_percentage < 50:
        congestion_level = "Free/Light Traffic"
    elif 50 <= occupancy_percentage < 70:
        congestion_level = "Moderate Traffic"
    else:
        congestion_level = "Heavy Traffic"

    return congestion_level

# Function to convert address to coordinates using OpenRouteService Geocoding API
def address_to_coordinates(address):
    endpoint = 'https://api.openrouteservice.org/geocode/search'
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': f'Bearer {api_key}'
    }
    params = {
        'api_key': api_key,
        'text': address
    }
    response = requests.get(endpoint, params=params, headers=headers)
    data = response.json()
    if 'features' in data and data['features']:
        coordinates = data['features'][0]['geometry']['coordinates']
        return coordinates
    else:
        return None
    
'''@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle POST request data here if needed
        pass
    return render_template('index.html')
'''
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        source_address = request.form['source_address']
        destination_address = request.form['destination_address']

        # Convert addresses to coordinates
        source_coordinates = address_to_coordinates(source_address)
        destination_coordinates = address_to_coordinates(destination_address)

        source_coordinate=(43.152955,-77.604475)
        destination_coordinate=(43.156548,-77.608865)

        if source_coordinate and destination_coordinate:
            # Define the API endpoint for route calculation
            endpoint = 'https://api.openrouteservice.org/v2/directions/driving-car'

            # Define headers
            headers = {
                'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
                'Authorization': f'Bearer {api_key}'
            }

            # Define query parameters
            params = {
                'start': f'{source_coordinate[1]},{source_coordinate[0]}',
                'end': f'{destination_coordinate[1]},{destination_coordinate[0]}'
            }

            # Make a GET request to the API
            response = requests.get(endpoint, params=params, headers=headers)
            print("Response status code:", response.status_code)  # Add this line for debugging

            # Parse the response JSON data
            data = response.json()
            print("Response data:", data)  # Add this line for debugging

            # Extract the coordinates of the route
            route_coordinates = []
            if 'features' in data:
                route_coordinates = data['features'][0]['geometry']['coordinates']
                print("Route coordinates:", route_coordinates)  # Add this line for debugging
            else:
                print("No 'features' key found in response:", data)  # Add this line for debugging

            # Create a Folium map centered at the origin
            m = folium.Map(location=source_coordinate, zoom_start=15)

            # Add markers for the origin and destination points
            folium.Marker(location=source_coordinate, popup='Source').add_to(m)
            folium.Marker(location=destination_coordinate, popup='Destination').add_to(m)

            # Add a polyline for the route if coordinates are available
            if route_coordinates:
                folium.PolyLine(locations=[(coord[1], coord[0]) for coord in route_coordinates], color='blue', weight=5).add_to(m)

            # Video surveillance
            video_url = 'https://s53.nysdot.skyvdn.com/rtplive/R4_227/playlist.m3u8'
            cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

            ret, frame = cap.read()
            if ret:
                # Detect vehicles in the frame
                frame_with_vehicles, vehicle_boxes = detect_vehicles(frame)

                # Calculate traffic congestion level
                congestion_level = calculate_traffic_congestion(vehicle_boxes, frame.shape[1])

                # Add marker with popup showing congestion level at a separate coordinate
                congestion_coordinate = (43.154059, -77.605101)  # Separate coordinate for congestion level
                folium.Marker(location=[congestion_coordinate[0], congestion_coordinate[1]], popup=f"Traffic Congestion Level: {congestion_level}").add_to(m)

            return render_template('map.html')
        else:
            return "Error: Unable to convert addresses to coordinates."
    return render_template('search.html')

@app.route('/congestion_level', methods=['GET'])
def congestion_level():
    video_url = 'https://s53.nysdot.skyvdn.com/rtplive/R4_227/playlist.m3u8'
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    ret, frame = cap.read()
    if ret:
        frame_with_vehicles, vehicle_boxes = detect_vehicles(frame)


        congestion_level = calculate_traffic_congestion(vehicle_boxes, frame.shape[1])

    
        return jsonify({'level': congestion_level})

# Video feed generator function
def generate_video_feed():
    video_url = 'https://s53.nysdot.skyvdn.com/rtplive/R4_227/playlist.m3u8'
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles and overlay on the frame
        frame_with_vehicles, _ = detect_vehicles(frame)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_with_vehicles)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release video stream
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
