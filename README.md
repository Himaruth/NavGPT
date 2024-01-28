# NavGPT
Features
Chatbot Interaction:

Users can interact with a chatbot to receive responses to their messages.
Predefined responses are available for common queries.
Real-time Video Feed:

Display of a real-time video feed from a specified source.
Integration with OpenCV for video processing and vehicle detection.
Traffic Congestion Detection:

Utilizes YOLOv3 model for vehicle detection in the video feed.
Calculates traffic congestion level based on vehicle density.
Route Mapping:

Integration with OpenRouteService API to generate routes between specified source and destination addresses.
Display of route maps using Folium library with markers for source, destination, and traffic congestion levels.
Multilingual Support:

Utilizes RAG (Retrieve and Generate) model for translation and text summarization.
Supports translation of user messages into multiple languages.
Dynamic Web Interface:

Responsive web interface for easy interaction with the application.
Allows users to input source and destination addresses for route mapping.
Provides a chat interface for interacting with the chatbot.
Scalability and Flexibility:

Built using Flask, a lightweight and extensible web framework in Python.
Easily extensible with additional functionalities or integrations as needed.
