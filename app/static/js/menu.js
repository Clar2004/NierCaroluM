const sse = new EventSource('/sse_menu');

sse.onmessage = function (event) {
   console.log("Status:", event.data);
   if (event.data === "redirect") {
      sse.close();

      const loadingOverlay = document.getElementById('loading-overlay');
      loadingOverlay.style.display = 'flex';

      setTimeout(() => {
         window.location.href = "/combat?reset=true";
      }, 2000);
   }
};

sse.onerror = function () {
   console.error("SSE connection failed");
   sse.close();
};
