const sse = new EventSource('/sse_game_status');

sse.onmessage = function (event) {
   console.log("Status:", event.data);
   if (event.data === "game_one") {
      sse.close();
      showVideoOverlayAndNavigate("/image_filter");
   }
   else if (event.data === "game_two") {
      sse.close();
      // window.location.href = "/threshold";
      showVideoOverlayAndNavigate("/threshold");
   }else if(event.data === "game_three"){
      sse.close();
      // window.location.href = "/edge_corner";
      showVideoOverlayAndNavigate("/edge_corner");
   }else if(event.data === "game_four"){
      sse.close();
      // window.location.href = "/image_match";
      showVideoOverlayAndNavigate("/image_match");
   }else if(event.data === "dead"){
      sse.close();
      // window.location.href = "/";
      showVideoOverlayAndNavigate2("/");
   }
};

sse.onerror = function () {
   console.error("SSE connection failed");
   sse.close(); // Close SSE connection on error
};

window.onload = function () {
   const overlay = document.getElementById("video-overlay");
   overlay.style.display = "none";
}

function showVideoOverlayAndNavigate(url) {
   const overlay = document.getElementById("video-overlay");
   const video = document.getElementById("transition-video");
   const audio = document.getElementById("transition-sound");

   // Show the overlay and start the video
   overlay.style.display = "flex";
   video.play();
   audio.play();

   video.style.opacity = 0.5;
   audio.volume = 0.3;

   // Hide the overlay after 3 seconds (video should play for 3 seconds)
   setTimeout(function () {
      audio.pause(); // Stop the audio
      audio.currentTime = 0;
      window.location.href = url;
   }, 2000); // Wait for 3 seconds before redirecting
}

function showVideoOverlayAndNavigate2(url) {
   const overlay = document.getElementById("video-overlay2");
   const video = document.getElementById("transition-video2");
   const audio = document.getElementById("transition-sound2");

   overlay.style.display = "flex";
   video.currentTime = 11; 
   video.play();
   audio.play();

   audio.volume = 0.3;

   // Hide the overlay after 5 seconds (video should play for 5 seconds)
   setTimeout(function () {
      audio.pause(); 
      audio.currentTime = 0;
      video.pause(); 
      video.currentTime = 10; 
      window.location.href = url;
   }, 4000); 
}
