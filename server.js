const express = require("express");
const bodyParser = require("body-parser");
const { exec } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(express.static("public"));
app.use("/images", express.static("images"));

const MAX_QUEUE_SIZE = 10; // Prevent unlimited queue growth
const queue = [];
let isProcessing = false;

function processQueue() {
    if (isProcessing || queue.length === 0) return;
    isProcessing = true;

    const { prompt, res } = queue.shift();
    const command = `python generate.py "${prompt.replace(/"/g, '\\"')}"`;

    console.log(`Processing prompt: "${prompt}"`);

    const child = exec(command, {
        timeout: 300000, // 5 minutes timeout for CPU generation timeout: 600000, // 10 minutes

        maxBuffer: 1024 * 1024 * 10 // 10MB buffer
    }, (err, stdout, stderr) => {
        isProcessing = false;


        if (err) {
            console.error("Generation error:", {
                killed: err.killed,
                code: err.code,
                signal: err.signal,
                stderr: stderr.toString().substring(0, 500) // Limit error length
            });
            let errorMsg = "Image generation failed";
            if (err.killed) {
                errorMsg = "Generation timed out (5 minutes) - try a simpler prompt";
            } else if (stderr.includes("CUDA")) {
                errorMsg = "GPU not available - using CPU mode";
            }

            return res.status(500).json({
                error: errorMsg,
                details: "System is processing in CPU-only mode. This may take several minutes for the first request."
                    //details: stderr.toString().substring(0, 200) // Limit detail length
            });
        }
        // ... rest of your existing code
    });

    // Handle child process exit
    child.on('exit', (code) => {
        if (code !== 0 && !isProcessing) {
            console.error(`Python script exited with code ${code}`);
        }
    });
}

app.post("/generate-image", (req, res) => {
    const prompt = req.body.prompt;
    if (!prompt) return res.status(400).json({ error: "No prompt provided" });

    if (queue.length >= MAX_QUEUE_SIZE) {
        return res.status(429).json({
            error: "Server busy",
            message: "Too many pending requests. Please try again later."
        });
    }

    queue.push({ prompt, res });
    processQueue();
});

app.listen(PORT, () => {
    console.log(`🚀 Server running at http://localhost:${PORT}`);
    // Ensure images directory exists
    if (!fs.existsSync(path.join(__dirname, 'images'))) {
        fs.mkdirSync(path.join(__dirname, 'images'));
        console.log('Created images directory');
    }
});