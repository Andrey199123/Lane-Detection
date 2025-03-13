document.addEventListener("DOMContentLoaded", function () {
  const logContent = document.getElementById("log-content");
  const buttons = document.querySelectorAll(".arrow-btn");
  const pauseResumeButton = document.getElementById("pause-resume-btn");

  // Function to send commands to the server
  function sendCommand(command) {
    fetch("http://127.0.0.1:10000/move", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        in_command: [command],
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(`Command sent: ${data.out_command}`);
        updateLog(`Command sent: ${data.out_command}`);
      })
      .catch((error) => {
        console.error("Error:", error);
        updateLog(`Error: ${error}`);
      });
  }

  // Function to update log and scroll to bottom, with filtering
  function updateLog(newLogEntry) {
    // Filter out unwanted log entries
    const ignorePatterns = [
      "GET /logs/api_output.log",
      "GET /logs/login_record.log",
    ];

    // Check if the log entry contains any of the ignore patterns
    const shouldIgnore = ignorePatterns.some((pattern) =>
      newLogEntry.includes(pattern)
    );
    if (shouldIgnore) return; // Skip adding the log if it matches ignore patterns

    // Append new log entry and scroll to the bottom
    logContent.textContent += `\n${newLogEntry}`;
    logContent.scrollTop = logContent.scrollHeight;
  }

  // Add event listeners for hold-to-move functionality
  buttons.forEach((button) => {
    const command = button.getAttribute("data-command");

    button.addEventListener("mousedown", () => {
      sendCommand(command);
    });

    button.addEventListener("mouseup", () => {
      sendCommand("stop");
    });

    button.addEventListener("mouseleave", () => {
      sendCommand("stop");
    });
  });

  pauseResumeButton.addEventListener("click", () => {
  fetch("http://127.0.0.1:10000/toggle_play", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then(response => response.json())
    .then(data => {
      if (data.status === "Playing") {
        pauseResumeButton.textContent = "Pause";
        pauseResumeButton.setAttribute("data-status", "playing");
        updateLog("Videos resumed");
      } else if (data.status === "Paused") {
        pauseResumeButton.textContent = "Resume";
        pauseResumeButton.setAttribute("data-status", "paused");
        updateLog("Videos paused");
      }
    })
    .catch(error => {
      console.error("Error:", error);
      updateLog(`Error: ${error}`);
    });
});

  // Example function to simulate log updates (for testing purposes)
  function simulateLogUpdates() {
    const sampleLogs = [
      "INFO: Connection established",
      "INFO: Command received: forward",
      "INFO: Command executed",
      "ERROR: Connection timeout",
      "INFO: Reconnecting...",
    ];

    let index = 0;
    setInterval(() => {
      if (index < sampleLogs.length) {
        updateLog(sampleLogs[index]);
        index++;
      } else {
        index = 0; // Loop through the sample logs
      }
    }, 2000); // Update log every 2 seconds
  }

  // Call simulateLogUpdates to demonstrate live scrolling (remove in production)
  simulateLogUpdates();
});
