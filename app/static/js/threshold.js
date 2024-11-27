const socket = io("http://127.0.0.1:5000", {
   transports: ["websocket"],
   reconnectionDelay: 2000,
});

console.log("Socket.IO connected:", socket);

socket.on("disconnect", () => {
   console.log("Disconnected from server");
});

function handleKeyPress(event) {
   if (event.key === "Enter") {
      checkThresholdMessage();
   }
}

function checkThresholdMessage() {
   const userMessage = document.getElementById("decoded-message").value.trim();
   const correctMessage = "For The Glory Of Mankind";

   const terminal = document.getElementById("terminal");
   const currentPrompt = document.getElementById("current-prompt");

   // Create a new terminal line to show the user's input
   const newLine = document.createElement("div");
   newLine.classList.add("terminal-line");
   newLine.innerHTML = `<span class="prompt">codebind@code:~$</span> ${userMessage}`;
   terminal.insertBefore(newLine, currentPrompt);

   if (userMessage === correctMessage) {
      // If the answer is correct, show success and move to next step
      const successLine = document.createElement("div");
      successLine.classList.add("terminal-line");
      successLine.innerHTML = `<span class="prompt">codebind@code:~$</span> Correct! You've solved the puzzle.`;
      terminal.insertBefore(successLine, currentPrompt);
      window.location.href = "/"; // Replace with your success page or action
   } else if (userMessage === "") {
      // If the answer is incorrect, show an error message in the terminal
      const errorLine = document.createElement("div");
      errorLine.classList.add("terminal-line");
      errorLine.innerHTML = `<span class="prompt">codebind@code:~$</span>`;
      terminal.insertBefore(errorLine, currentPrompt);
   } else{
      // If the answer is incorrect, show an error message in the terminal
      const errorLine = document.createElement("div");
      errorLine.classList.add("terminal-line");
      errorLine.innerHTML = `<span class="prompt">codebind@code:~$</span> Incorrect! Try again.`;
      terminal.insertBefore(errorLine, currentPrompt);
   }
   terminal.scrollTop = terminal.scrollHeight;

   // Clear the input field for the next command
   document.getElementById("decoded-message").value = "";
}

// Ensure the loading overlay is visible during page unload
window.addEventListener("beforeunload", (event) => {
   const url = "/release_camera";
   navigator.sendBeacon(url); // Ensure the camera release request is sent
   console.log("Camera release request sent using Beacon API.");

   setTimeout(() => {
      console.log("Page is unloading...");
   }, 200);
});

// Ensure the loading overlay hides after page load
window.addEventListener("load", () => {
   fetch("/initialize_camera", {
      method: "POST",
   }).then((response) => {
      if (response.ok) {
         console.log("Camera initialized.");
      }
   });
});
