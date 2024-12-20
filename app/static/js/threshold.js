function enableInput() {
   document.getElementById("message-input").disabled = false;
}

const correctMessage = "For The Glory Of Mankind";
const terminal = document.getElementById("terminal");
const inputField = document.getElementById("message-input");

function addTerminalLine(text, isError = false) {
   const line = document.createElement("div");
   line.className = "terminal-line";
   const prompt = document.createElement("span");
   prompt.className = "prompt";
   prompt.textContent = "codebind@code:~$ ";
   const message = document.createElement("span");
   message.textContent = text;

   if (isError) {
      message.style.color = "red";
   }

   line.appendChild(prompt);
   line.appendChild(message);
   terminal.insertBefore(line, document.getElementById("current-prompt"));

   terminal.scrollTop = terminal.scrollHeight;
}

function handleKeyPress(event) {
   if (event.key === "Enter") {
      const input = inputField.value.trim();

      if (!input) {
         addTerminalLine("", true);
         return;
      }

      fetch("/game_two", {
         method: "POST",
         headers: {
            "Content-Type": "application/json",
         },
         body: JSON.stringify({ message: input }),
      }).then(response => response.json())
         .then(data => {

            console.log("Response from server:", data);
            if (data.message == "Invalid message") {
               addTerminalLine("Incorrect! Try again.", true);
            } else {
               addTerminalLine("Correct! You've solved the puzzle.");
               addTerminalLine("Redirecting to the combat...");
            }
         })
         .catch(error => {

            addTerminalLine("An error occurred. Please try again.", true);

            console.error("Error:", error);
         });

      inputField.value = ""; 
   }
}

const sse = new EventSource('/sse_mini_game_two');

sse.onmessage = function (event) {
   console.log("Status:", event.data);
   if (event.data === "redirect") {
      sse.close();
      const loadingOverlay = document.getElementById('loading-overlay');
      loadingOverlay.style.display = 'flex';

      setTimeout(() => {
         window.location.href = "/combat";
      }, 2000);
   }
};

sse.onerror = function () {
   console.error("SSE connection failed");
   sse.close();
}
