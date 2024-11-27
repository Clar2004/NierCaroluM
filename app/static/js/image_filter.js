const socket = io("http://127.0.0.1:5000", {
   transports: ["websocket"],
   reconnectionDelay: 2000,
});

console.log("Socket.IO connected:", socket);

socket.on("disconnect", () => {
   console.log("Disconnected from server");
});

// Define goBack function globally
function goBack() {
   const url = "/release_camera";
   navigator.sendBeacon(url); // Ensure the camera release request is sent
   console.log("Camera release request sent using Beacon API.");

   setTimeout(() => {
      console.log("Page is unloading...");
   }, 200);
   window.history.back();
}

document.addEventListener("DOMContentLoaded", () => {
   const submitButton = document.getElementById("submit-button");
   const messageInput = document.getElementById("message-input");
   const errorMessage = document.getElementById("error-message");

   function checkMessage() {
      const input = messageInput.value.trim();
      const correctMessage = "NAR25-1 Semangat Jangan Merasa Aman";

      if (input === correctMessage) {
         window.location.href = "{{ url_for('menu') }}";
      } else {
         errorMessage.textContent = "Incorrect! Try again.";
         errorMessage.style.color = "red";
         errorMessage.style.fontSize = "1.2em";
         errorMessage.style.display = "block";
      }
   }

   submitButton.addEventListener("click", checkMessage);

   messageInput.addEventListener("input", () => {
      errorMessage.style.display = "none";
   });
});

window.addEventListener("beforeunload", (event) => {
   const url = "/release_camera";
   navigator.sendBeacon(url); // Ensure the camera release request is sent
   console.log("Camera release request sent using Beacon API.");

   setTimeout(() => {
      console.log("Page is unloading...");
   }, 200);
});

window.addEventListener("load", () => {
   fetch("/initialize_camera", {
      method: "POST",
   }).then((response) => {
      if (response.ok) {
         console.log("Camera initialized.");
      }
   });
});

//CMD Prompt
function enableInput() {
   document.getElementById("message-input").disabled = false;
}

const correctMessage = "NAR25-1 Semangat Jangan Merasa Aman";
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
      message.style.color = "red"; // Red text for incorrect inputs
   }

   line.appendChild(prompt);
   line.appendChild(message);
   terminal.insertBefore(line, document.getElementById("current-prompt"));

   // Scroll terminal to the bottom
   terminal.scrollTop = terminal.scrollHeight;
}

function handleKeyPress(event) {
   if (event.key === "Enter") {
      const input = inputField.value.trim();

      if (!input) {
         addTerminalLine("", true); // Add empty prompt line for empty input
      } else if (input === correctMessage) {
         addTerminalLine("Correct! You've solved the puzzle.");
         addTerminalLine("Redirecting to the menu...");
         setTimeout(() => {
            window.location.href = "/"; // Replace with your correct URL
         }, 2000);
      } else {
         addTerminalLine("Incorrect! Try again.", true);
      }

      inputField.value = ""; // Clear the input field
   }
}