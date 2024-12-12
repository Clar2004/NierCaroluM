const sse = new EventSource('/sse_mini_game_three');

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
   sse.close();
};