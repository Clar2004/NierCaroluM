const images = [
   "static/assets/image_matching/circle.jpg",
   "static/assets/image_matching/rhombus.jpg",
   "static/assets/image_matching/square.jpg"
];

const baseImage = "static/assets/image_matching/white_bg.jpg";

let isGameStart = false;
let countdownValue = 0;  // Declare countdownValue to be accessible
let accuracy_match = 0;
const accuracyText = document.getElementById('accuracy-text');
const countdownMessage = document.getElementById('countdown-message');
let countdownInterval;  // Declare interval for countdown
let isTriggered = false;
let isTriggered2 = false;
let isReceived = false;
let isCheatActivated = false;
let isGameEnd = false;

// Start countdown with a given number of seconds
// function startCountdown(seconds, stateNum) {
//    countdownValue = seconds;

//    // Clear any previous countdown interval
//    clearInterval(countdownInterval);

//    countdownInterval = setInterval(() => {
//       if (stateNum === 1) {
//          countdownMessage.textContent = `Game Starting in ${countdownValue}`;  // Use string interpolation
//          accuracyText.textContent = ``;
//       } else if (stateNum === 3) {
//          countdownMessage.textContent = "";
//          accuracyText.textContent = `Accuracy: Calculating...%`;
//       } else if (stateNum === 2) {
//          countdownMessage.textContent = `Time Remaining : ${countdownValue}`;  // Use string interpolation
//          accuracyText.textContent = ``;
//       }

//       countdownValue--;

//       // if (countdownValue == 2 && stateNum == 3) {
//       //    selectRandomImage();
//       // }

//       if (countdownValue < 0) {
//          clearInterval(countdownInterval);

//          if (stateNum == 3) {
//             countdownMessage.textContent = "Thumbs up to start...";
//             isTriggered = false;
//             if (isReceived == true) {
//                accuracyText.textContent = `Accuracy: ${accuracy_match.toFixed(2)}%`;
//                isReceived = false;
//                isTriggered2 = false;
//                isGameEnd = true;
//             }
//          }
//       }
//    }, 1000);
// }

function selectImage(num) {
   document.getElementById('random-image').src = images[num];
}

window.onload = function () {
   document.getElementById('random-image').src = baseImage;
   countdownMessage.textContent = "Thumbs up to start the game...";
   accuracyText.textContent = ``;
};

// Initialize variables for cheat code detection
const cheatCode = "duaempatsatu";
let currentInput = "";

document.addEventListener('keydown', function (event) {
   // Check if the pressed key is backspace
   if (event.key === 'Backspace') {
      // Remove the last character from currentInput if it's not empty
      currentInput = currentInput.slice(0, -1);
   } else {
      // Add the pressed key to the current input (convert to lowercase to match the cheat code)
      currentInput += event.key.toLowerCase();
   }

   // Check if the current input matches the cheat code
   if (currentInput === cheatCode) {
      triggerCheatCode();  // Call the cheat code action
      currentInput = "";  // Reset the input after triggering
   }

   // If the input exceeds the length of the cheat code, reset
   if (currentInput.length > cheatCode.length) {
      currentInput = currentInput.slice(1);  // Remove the first character if it's too long
   }
});

// Define the action to trigger when the cheat code is entered
function triggerCheatCode() {
   showCheatText();
   console.log("Cheat code activated! Game behavior changed.");
   isCheatActivated = true;
}

function showCheatText() {
   const cheatTextElement = document.getElementById('cheat-text');

   // Ensure data.message is available and then show the cheat text
   if (cheatTextElement) {
      cheatTextElement.textContent = "Cheat code activated!";
      cheatTextElement.style.display = 'block';

      // Hide cheat text after 3 seconds
      setTimeout(() => {
         cheatTextElement.style.display = 'none';
      }, 3000);
   }
}

// Setup SSE connection
const eventSource = new EventSource('/sse_mini_game_four');

eventSource.onmessage = function (event) {
   console.log(event.data)
   
   if (event.data === 'redirect'){
      const loadingOverlay = document.getElementById('loading-overlay');
      loadingOverlay.style.display = 'flex';

      setTimeout(() => {
         window.location.href = "/combat";
      }, 4000);
   }
};

// SSE connection for accuracy updates
const eventSource2 = new EventSource('/sse_mini_game_four_accuracy');

eventSource2.onmessage = function (event) {
   const data = JSON.parse(event.data);

   if (data.event === 'accuracy') {
      accuracy_match = data.accuracy;

      if (isCheatActivated) {
         console.log("Cheat activated, accuracy set to 999.99");
         accuracy_match = 999.99;  // For cheat mode
      }

      // Update accuracy text after the game
      countdownMessage.textContent = "Thumbs up to start...";
      accuracyText.textContent = `Accuracy: ${accuracy_match.toFixed(2)}%`;

      // Perform action based on accuracy
      if (accuracy_match > 70) {

         fetch("/game_four", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "Success" }),
         });
      }

      console.log("Accuracy:", accuracy_match);
      isReceived = true;
   }

   else if (data.event === "game_start"){
      index_image = data.image_index
      selectImage(index_image);
   }

   else if (data.event === "countdown_start"){
      data_time = data.time
      countdownMessage.textContent = `Game Starting in ${data_time}s`;
      accuracyText.textContent = ``;
   }

   else if (data.event === "drawing_start") {
      data_time = data.time
      countdownMessage.textContent = `Time Remaining: ${data_time}s`;
      accuracyText.textContent = ``;
   }

   else if (data.event === "countdown_end"){
      countdownMessage.textContent = "";
      accuracyText.textContent = `Accuracy: Calculating...%`;
   }
};

eventSource.onerror = function (error) {
   console.error("EventSource failed:", error);
};
