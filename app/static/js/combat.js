const sse = new EventSource('/sse_game_status');

sse.onmessage = function (event) {
   console.log("Status:", event.data);
   if (event.data === "game_one") {
      // sse.close();
      window.location.href = "/image_filter";
   }
   else if (event.data === "game_two") {
      // sse.close();
      window.location.href = "/threshold";
   }else if(event.data === "game_three"){
      // sse.close();
      window.location.href = "/edge_corner";
   }else if(event.data === "dead"){
      window.location.href = "/";
   }

   //bisa tambah lagi
};

sse.onerror = function () {
   console.error("SSE connection failed");
   sse.close(); // Close SSE connection on error
};

function forcePageRefresh() {
   window.location.reload(true);  // The true parameter forces a hard reload
}