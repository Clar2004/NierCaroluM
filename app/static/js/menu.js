const audio = document.getElementById('background-audio');
audio.volume = 0.5;

document.addEventListener("DOMContentLoaded", () => {
   const audio = document.getElementById("background-audio");

   // Try to play the audio with sound directly
   audio.play().then(() => {
      // Audio playback started without restrictions
      console.log("Audio is playing with sound!");
   }).catch(error => {
      // Autoplay was blocked, show the enable sound button
      console.log("Autoplay with sound was blocked, prompting user interaction.");
   });
});

// Button Hover Logic
// Select both buttons and the menu container
const playDemoButton = document.getElementById('play-demo');
const menuContainer = document.querySelector('.menu');

// Set "Play Demo" as the default active button
playDemoButton.classList.add('active');

// Function to set active class on a button
function setActiveButton(button) {
   // Remove active class from both buttons
   playDemoButton.classList.remove('active');
   // Add active class to the selected button
   button.classList.add('active');
}

// Add event listeners to each button for hover effect
playDemoButton.addEventListener('mouseenter', () => setActiveButton(playDemoButton));

//redirect logic
function handleButtonClick(event, url) {
   event.preventDefault();

   // Delay before showing the loading overlay
   setTimeout(() => {
      // Show the loading overlay
      const loadingOverlay = document.getElementById('loading-overlay');
      loadingOverlay.style.display = 'flex';

      // Redirect to the specified URL after showing the overlay
      setTimeout(() => {
         loadingOverlay.style.display = 'none'; // Optionally hide overlay again
         window.location.href = url;
      }, 2000); // Delay for 2 seconds before redirecting (for the spinner to be visible)
   }, 500); // Initial delay before showing overlay (0.5 seconds)
}