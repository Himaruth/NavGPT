// Initialize Leaflet map
var map = L.map('map').setView([43.153108, -77.602934], 13);

// Add Tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// Function to update map with route
function updateMap() {
    var sourceAddress = document.getElementById('sourceInput').value;
    var destinationAddress = document.getElementById('destinationInput').value;

    // Call routing API to get route coordinates
    // Replace this with your routing API call
    var routeCoordinates = getRouteCoordinates(sourceAddress, destinationAddress);

    if (routeCoordinates) {
        // Clear existing route layer if any
        if (map.hasLayer(routeLayer)) {
            map.removeLayer(routeLayer);
        }

        // Add new route layer to map
        var routeLayer = L.polyline(routeCoordinates, {color: 'blue'}).addTo(map);

        // Fit map to bounds of route
        map.fitBounds(routeLayer.getBounds());
    } else {
        alert("No route found.");
    }
}

// Function to send message in chatbot
function sendMessage() {
    var userInput = document.getElementById('userInput').value;
    var chatDiv = document.getElementById('chat');
    var userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.innerHTML = userInput;
    chatDiv.appendChild(userMessage);
}
