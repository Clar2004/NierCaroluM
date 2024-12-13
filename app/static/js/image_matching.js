const images = [
   "static/assets/image_matching/circle.jpg",
   "static/assets/image_matching/square.jpg"
];

const baseImage = "static/assets/image_matching/white_bg.jpg";

let isGameStart = false;
let countdownValue = 0; 
const accuracyText = document.getElementById('accuracy-text');
const countdownMessage = document.getElementById('countdown-message');
let countdownInterval;  
let isTriggered = false;
let isTriggered2 = false;
let isReceived = false;
let isCheatActivated = false;
let isGameEnd = false;
let index_image = -1;
let accuracy_match = -1;

function selectImage(num) {
   document.getElementById('random-image').src = images[num];
}

window.onload = function () {
   document.getElementById('random-image').src = baseImage;
   countdownMessage.textContent = "Thumbs up to start the game...";
   accuracyText.textContent = ``;
};

const cheatCode = "duaempatsatu";
let currentInput = "";

document.addEventListener('keydown', function (event) {
   if (event.key === 'Backspace') {
      currentInput = ""
   } else {
      currentInput += event.key.toLowerCase();
   }

   if (currentInput === cheatCode) {
      triggerCheatCode();
      currentInput = ""; 
   }
});

function triggerCheatCode() {
   showCheatText();
   console.log("Cheat code activated! Game behavior changed.");
   isCheatActivated = true;
}

function showCheatText() {
   const cheatTextElement = document.getElementById('cheat-text');

   if (cheatTextElement) {
      cheatTextElement.textContent = "Cheat code activated!";
      cheatTextElement.style.display = 'block';

      setTimeout(() => {
         cheatTextElement.style.display = 'none';
      }, 3000);
   }
}

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

const eventSource2 = new EventSource('/sse_mini_game_four_accuracy');

eventSource2.onmessage = function (event) {
   const data = JSON.parse(event.data);

   // if (data.event === "game_start"){
   //    index_image = data.image_index
   //    console.log(index_image)
   //    document.getElementById('random-image').src = images[index_image];
   //    // selectImage(index_image);
   // }

   const randomImageElement = document.getElementById('random-image');

   if (data.event === "game_start") {
      console.log("game start")
      index_image = data.image_index;
      console.log("Image index received:", index_image);
      if (randomImageElement) {
         const newSrc = images[index_image];
         if (newSrc) {
            randomImageElement.src = `${newSrc}?t=${new Date().getTime()}`;
            console.log("Image updated to:", newSrc);
         } else {
            console.error("Invalid index:", index_image);
         }
      } else {
         console.error("Element with ID 'random-image' not found!");
      }
   }

   else if (data.event === "countdown_start"){
      console.log("countdown start")
      if (index_image == -1) {
         index_image = 0;
         const newSrc = images[0];
         randomImageElement.src = `${newSrc}?t=${new Date().getTime()}`;
         console.log("Test");
      }
      data_time = data.time
      countdownMessage.textContent = `Game Starting in ${data_time}s`;
      accuracyText.textContent = ``;
   }

   else if (data.event === "drawing_start") {
      console.log("drawing start")
      data_time = data.time
      countdownMessage.textContent = `Time Remaining: ${data_time}s`;
      accuracyText.textContent = ``;
   }

   else if (data.event === "countdown_end"){
      console.log("countdown end")
      countdownMessage.textContent = "";
      accuracyText.textContent = `Accuracy: Calculating...%`;
   }

   else if (data.event === 'accuracy') {
      console.log("accuracy received")
      accuracy_match = data.accuracy;

      if (isCheatActivated) {
         console.log("Cheat activated, accuracy set to 999.99");
         accuracy_match = 999.99;

         countdownMessage.textContent = "Thumbs up to start the game...";
         accuracyText.textContent = `Accuracy: ${accuracy_match}%`;
      }

      if (accuracy_match > 70) {

         fetch("/game_four", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "Success" }),
         });

         eventSource2.close();
      }

      console.log("Accuracy:", accuracy_match);
      isReceived = true;
   }

   else if (data.event === 'wait') {
      console.log("Waiting for user input...", isCheatActivated);
      if (isCheatActivated) {
         countdownMessage.textContent = "Thumbs up to start the game...";
         accuracyText.textContent = `Accuracy: 999.99%`;
      }

      else {
         countdownMessage.textContent = "Thumbs up to start the game...";
         if (accuracy_match === -1){
            accuracyText.textContent = ``;
         }else{
            accuracyText.textContent = `Accuracy: ${accuracy_match.toFixed(2)}%`;
         }
      }

   }
};

eventSource.onerror = function (error) {
   console.error("EventSource failed:", error);
};
