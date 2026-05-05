async function sendMessage() {

    const input = document.getElementById("question");

    const question = input.value;

    input.value = "";

    const chatBox = document.getElementById("chat-box");

    // User message
    chatBox.innerHTML += `
        <div class="user">
            <b>You:</b> ${question}
        </div>
    `;

    // Bot message div
    const botDiv = document.createElement("div");

    botDiv.className = "bot";

    botDiv.innerHTML = "<b>Bot:</b> ";

    chatBox.appendChild(botDiv);

    // Fetch stream
    const response = await fetch(
        "http://127.0.0.1:8000/chat",
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                history: chatBox.innerText.slice(-200)
            })
        }
    );

    // Stream reader
    const reader = response.body.getReader();

    const decoder = new TextDecoder();

    while (true) {

        const { done, value } =
            await reader.read();

        if (done) break;

        const chunk =
            decoder.decode(value);

        botDiv.innerHTML += chunk;

        chatBox.scrollTop =
            chatBox.scrollHeight;
    }
}

document
    .getElementById("question")
    .addEventListener(
        "keypress",
        function(event) {

            if (event.key === "Enter") {
                sendMessage();
            }
        }
    );