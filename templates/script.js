function sendMessage() {
    var userInput = document.getElementById("userInput").value;
    document.getElementById("userInput").value = ""; // Clear input field
    if (userInput !== "") {
        appendMessage("user", userInput); // Display user message
        fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: userInput
            })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage("bot", data.message); // Display chatbot response
        })
        .catch(error => console.error("Error:", error));
    }
}

function appendMessage(sender, message) {
    var chatDiv = document.getElementById("chat");
    var messageDiv = document.createElement("div");
    messageDiv.className = sender;
    messageDiv.innerHTML = message;
    chatDiv.appendChild(messageDiv);
}
