const socket = io("http://127.0.0.1:5000", {
   transports: ["websocket"],
   reconnectionDelay: 2000,
});

console.log("Socket.IO connected:", socket);

socket.on('connect', () => {
   console.log("Socket.IO connected2:", socket.connected);
});

socket.on('test', function (data) {
   console.log("Test event received:", data);
});

socket.on('connect_error', (err) => {
   console.error("Connection failed: ", err);
});

socket.on('disconnect', () => {
   console.log("Disconnected from server");
});

setTimeout(() => {
   const videoContainer = document.createElement('div');
   videoContainer.className = 'game-container';
   videoContainer.innerHTML = '<img id="video" src="/maze_game_feed" alt="Maze Game Feed">';
   document.body.appendChild(videoContainer);
}, 2000);

window.addEventListener("beforeunload", (event) => {
   const url = "/release_camera";
   navigator.sendBeacon(url); // Ensure the camera release request is sent
   console.log("Camera release request sent using Beacon API.");

   setTimeout(() => {
      console.log("Page is unloading...");
   }, 200);
});

function goBack() {
   
   const url = "/release_camera";
   navigator.sendBeacon(url); // Ensure the camera release request is sent
   console.log("Camera release request sent using Beacon API.");

   setTimeout(() => {
      console.log("Page is unloading...");
   }, 200);

   window.history.back();
}

// Handle CPU reached event
socket.on('cpu_reached', function (data) {
   console.log("CPU reached event received:", data);
   window.location.href = "/";
});

// Handle health update
socket.on('health_update', function (data) {
   console.log("Health update event received:", data);

   const healthContainer = document.getElementById('health-container');
   if (healthContainer) {
      // Get the current health value
      const currentHealth = data.health;
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

      console.log("Health updated to ", currentHealth);
   } else {
      console.error("Failed to update health, health container not found.");
   }
});