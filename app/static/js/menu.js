const sse = new EventSource('/sse_menu');

sse.onmessage = function (event) {
   console.log("Status:", event.data);
   if (event.data === "redirect") {

      const loadingOverlay = document.getElementById('loading-overlay');
      loadingOverlay.style.display = 'flex';

      setTimeout(() => {
         window.location.href = "/combat?reset=true";
         sse.close();
      }, 2000);
   }
};

sse.onerror = function () {
   console.log("SSE connection failed");
   sse.close();
};

const sse2 = new EventSource('/sse_game_status');

sse2.onmessage = function (event) {
   console.log("Status:", event.data);
   if (event.data === "dead") {
      sse2.close();
      location.reload();
   }
};
