
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
   videoContainer.innerHTML = '<img class="video-feed" src="/combat_feed" alt="Maze Game Feed">';
   document.body.appendChild(videoContainer);
}, 2000);

socket.on('redirect_to_menu', function (data) {
   console.log("Boss Defeated:", data);
   window.location.href = "/";
});

window.addEventListener("beforeunload", (event) => {
   const url = "/release_camera";
   navigator.sendBeacon(url);
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