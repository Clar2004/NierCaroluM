const images = [
   "static/assets/image_matching/house.jpg",
   "static/assets/image_matching/star.jpg",
   "static/assets/image_matching/sword.png"
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
function startCountdown(seconds, stateNum) {
   countdownValue = seconds;

   // Clear any previous countdown interval
   clearInterval(countdownInterval);

   countdownInterval = setInterval(() => {
      if (stateNum === 1) {
         countdownMessage.textContent = `Game Starting in ${countdownValue}`;  // Use string interpolation
         accuracyText.textContent = ``;
      } else if (stateNum === 3) {
         countdownMessage.textContent = "";
         accuracyText.textContent = `Accuracy: Calculating...%`;
      } else if (stateNum === 2) {
         countdownMessage.textContent = `Time Remaining : ${countdownValue}`;  // Use string interpolation
         accuracyText.textContent = ``;
      }

      countdownValue--;

      if (countdownValue == 2 && stateNum == 3) {
         selectRandomImage();
      }

      if (countdownValue < 0) {
         clearInterval(countdownInterval);

         if (stateNum == 3) {
            countdownMessage.textContent = "Thumbs up to start...";
            isTriggered = false;
            if (isReceived == true) {
               accuracyText.textContent = `Accuracy: ${accuracy_match.toFixed(2)}%`;
               isReceived = false;
               isTriggered2 = false;
               isGameEnd = true;
            }
         }
      }
   }, 1000);
}

function selectRandomImage() {
   const randomIndex = Math.floor(Math.random() * images.length);
   document.getElementById('random-image').src = images[randomIndex];

   // Send the index of the selected image to the server
   fetch('/set_image_index', {
      method: 'POST',
      headers: {
         'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imageIndex: randomIndex })
   })
      .then(response => response.json())
      .then(data => console.log("Image index sent to server:", data))
      .catch(error => console.error("Error sending image index:", error));
}

window.onload = function () {
   document.getElementById('random-image').src = baseImage;
   selectRandomImage();
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

   // Handle different events
   if (event.data === 'countdown_start') {
      if (isTriggered === false) {
         startCountdown(3, 1);
         isTriggered = true;
         isGameEnd = false;
      }
   } else if (event.data === 'countdown_end') {
      startCountdown(3, 3);
   } else if (event.data === 'drawing_start') {
      startCountdown(5, 2);
   } 
};

const eventSource2 = new EventSource('/sse_mini_game_four_accuracy');

eventSource2.onmessage = function (event) {
   const data = JSON.parse(event.data);

   if (data.event === 'accuracy') {
      const accuracy = data.accuracy;
      accuracy_match = accuracy;

      if (isCheatActivated){
         console.log("Cheat activated, accuracy set to 999.99");
         accuracy_match = 999.99;
      }

      if (accuracy_match > 70) {
         const loadingOverlay = document.getElementById('loading-overlay');
         loadingOverlay.style.display = 'flex';

         input = "Sucess";

         fetch("/game_four", {
            method: "POST",
            headers: {
               "Content-Type": "application/json",
            },
            body: JSON.stringify({ message: input }),
         })

         setTimeout(() => {
            window.location.href = "/combat";
         }, 4000);
         eventSource.close();
         eventSource2.close();
      }

      console.log("Accuracy:", accuracy);
      isReceived = true;
   }
}

eventSource.onerror = function (error) {
   console.error("EventSource failed:", error);
};
