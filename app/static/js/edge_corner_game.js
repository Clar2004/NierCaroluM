function updateHealth(currentHealth) {

   const healthContainer = document.getElementById('health-container');
   if (healthContainer) {
      const icons = healthContainer.getElementsByClassName('health-icon');

      // Hide the health container if health is 0
      if (currentHealth === 0) {
         healthContainer.style.display = 'none';  // Hide the entire health container
      } else {
         healthContainer.style.display = 'flex';  // Show the health container if health > 0

         // Loop through the icons and toggle their visibility based on current health
         for (let i = 0; i < icons.length; i++) {
            if (i < currentHealth) {
               icons[i].style.display = 'flex';  // Show the icon if health allows
            } else {
               icons[i].style.display = 'none';  // Hide the icon if health doesn't allow
            }
         }
      }
      console.log("Health updated to", currentHealth);
   }
}

setTimeout(() => {
   const videoContainer = document.createElement('div');
   videoContainer.className = 'game-container';
   videoContainer.innerHTML = '<img id="video" src="/maze_game_feed" alt="Maze Game Feed">';
   document.body.appendChild(videoContainer);
}, 100);

function showCheatText() {
      const cheatTextElement = document.getElementById('cheat-text');

      // Ensure data.message is available and then show the cheat text
      if (cheatTextElement ) {
         cheatTextElement.textContent = "Cheat code activated!";
         cheatTextElement.style.display = 'block';

         // Hide cheat text after 3 seconds
         setTimeout(() => {
            cheatTextElement.style.display = 'none';
         }, 3000);
      }
}

const sse = new EventSource('/sse_mini_game_three');

sse.onmessage = function (event) {
   console.log("Status:", event.data);

   if (event.data === "redirect") {
      // sse.close();
      const loadingOverlay = document.getElementById('loading-overlay');
      loadingOverlay.style.display = 'flex';

      setTimeout(() => {
         window.location.href = "/combat";
      }, 2000);
   }

   if (event.data === "cheat") {
      showCheatText();
   }

   if (event.data === "zero") {
      updateHealth(0);
   } else if (event.data === "one") {
      updateHealth(1);
   } else if (event.data === "two") {
      updateHealth(2);
   } else if (event.data === "three") {
      updateHealth(3);
   }
};

sse.onerror = function () {
   console.error("SSE connection failed");
   sse.close(); // Close SSE connection on error
};