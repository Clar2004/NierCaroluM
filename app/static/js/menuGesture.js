// const socket = io("http://127.0.0.1:5000", {
//    transports: ["websocket"],
//    reconnectionDelay: 2000,
// });

// console.log("Socket.IO connected:", socket);

// const customCursor = document.getElementById('custom-cursor');

// socket.on("disconnect", () => {
//    console.log("Disconnected from server");

//    socket.off("move_cursor"); // Unsubscribe from socket events
//    socket.off("click"); // Unsubscribe from socket events
// });

// // Function to set the active button by adding/removing the active class
// function setActiveButton(button) {
//    // Remove the active class from playDemoButton
//    playDemoButton.classList.remove('active');

//    // Add active class to the selected button (i.e., playDemoButton when hovered)
//    if (button) {
//       button.classList.add('active');
//    }
// }

// // Function to detect hover using MediaPipe cursor position
// function checkHoverWithMediaPipe(cursorX, cursorY) {
//    const playDemoRect = playDemoButton.getBoundingClientRect();

//    console.log(`Checking hover for cursor at (${cursorX}, ${cursorY})`);

//    if (
//       cursorX >= playDemoRect.left &&
//       cursorX <= playDemoRect.right &&
//       cursorY >= playDemoRect.top &&
//       cursorY <= playDemoRect.bottom
//    ) {
//       setActiveButton(playDemoButton);
//    } else {
//       setActiveButton(null); // Remove active status if not hovered
//    }
// }

// // MediaPipe move_cursor event listener
// socket.on('move_cursor', (data) => {
//    if (!data.x || !data.y) {
//       console.error('Missing x or y in move_cursor data');
//       return;
//    }

//    const cursorX = data.x; // Declare cursorX
//    const cursorY = data.y; // Declare cursorY

//    console.log('Cursor position:', cursorX, cursorY);

//    // Update custom cursor position
//    customCursor.style.left = `${cursorX}px`;
//    customCursor.style.top = `${cursorY}px`;
//    customCursor.style.display = 'block';

//    // Check for hover using MediaPipe cursor
//    checkHoverWithMediaPipe(cursorX, cursorY);
// });

// // Listen for the 'click' event to simulate a click at the specified position
// socket.on('click', (data) => {
//    console.log('Click event received:', data);

//    // Optional visual feedback for the click
//    customCursor.classList.add('click-effect');
//    setTimeout(() => customCursor.classList.remove('click-effect'), 200);

//    // Simulate a click at the specified position
//    const event = new MouseEvent('click', {
//       bubbles: true,
//       cancelable: true,
//       view: window,
//       clientX: data.x,
//       clientY: data.y,
//    });

//    // Dispatch the click event on the element at the cursor's position
//    const targetElement = document.elementFromPoint(data.x, data.y);
//    console.log('Target element:', targetElement); // Log the target element
//    if (targetElement) {
//       targetElement.dispatchEvent(event);
//    } else {
//       console.error('No target element found at the cursor position.');
//    }
// });

// // Track custom cursor's position and update its location
// document.addEventListener('mousemove', (event) => {
//    customCursor.style.left = `${event.pageX}px`;
//    customCursor.style.top = `${event.pageY}px`;
// });

// Handle button click with loading overlay
// function handleButtonClick(event, url) {
   // const loadingOverlay = document.getElementById('loading-overlay');
   // loadingOverlay.style.display = 'flex'; // Show the loader

   // window.location.href = url;
   // setTimeout(() => {
      // loadingOverlay.style.display = 'none'; // Hide the loader
   // }, 500);
   // fetch("/release_camera", {
   //    method: "POST",
   // })
   //    .then((response) => {
   //       if (response.ok) {
   //          console.log("Camera released.");
   //       }
   //    })
   //    .catch((error) => {
   //       console.error("Failed to release camera:", error);
   //    })
   //    .finally(() => {
   //       // Navigate to the next page after releasing the camera
   //        // Delay ensures the camera release completes
   //    });
// }

// Ensure the loading overlay is visible during page unload
// window.addEventListener("beforeunload", (event) => {
//    const loadingOverlay = document.getElementById('loading-overlay');
//    loadingOverlay.style.display = 'flex'; // Show the loading overlay

//    const url = "/release_camera";
//    navigator.sendBeacon(url); // Ensure the camera release request is sent
//    console.log("Camera release request sent using Beacon API.");

//    setTimeout(() => {
//       console.log("Page is unloading...");
//    }, 200);
// });

// Ensure the loading overlay hides after page load
// window.addEventListener("load", () => {
//    const loadingOverlay = document.getElementById('loading-overlay');
//    fetch("/initialize_camera", {
//       method: "POST",
//    }).then((response) => {
//       if (response.ok) {
//          console.log("Camera initialized.");
//       }
//    }).finally(() => {
//       loadingOverlay.style.display = 'none'; // Hide the loading overlay after initialization
//    });
// });
