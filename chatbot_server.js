const express = require("express");
const axios = require("axios");
require("dotenv").config();
const { Configuration, OpenAIApi } = require("openai");
const natural = require("natural");

const app = express();
app.use(express.json());

const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const conversations = {}; // Stores user sessions

function detectBias(text) {
    const biasedWords = ["always", "never", "everyone", "no one", "clearly", "obviously"];
    return biasedWords.some(word => text.toLowerCase().includes(word));
}

app.post("/chat", async (req, res) => {
    const { userId, message } = req.body;
    if (!userId || !message) {
        return res.status(400).json({ error: "Missing userId or message" });
    }

    if (!conversations[userId]) {
        conversations[userId] = [];
    }
    conversations[userId].push({ role: "user", content: message });

    try {
        const response = await openai.createChatCompletion({
            model: "gpt-4",
            messages: conversations[userId],
        });
        let botReply = response.data.choices[0].message.content;
        
        if (detectBias(botReply)) {
            botReply += "\n(Note: This response may contain biased language. Please consider multiple perspectives.)";
        }
        
        conversations[userId].push({ role: "assistant", content: botReply });
        res.json({ reply: botReply });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// Example: Integrating a weather API
app.get("/weather", async (req, res) => {
    const { city } = req.query;
    if (!city) {
        return res.status(400).json({ error: "City parameter is required" });
    }
    try {
        const weatherResponse = await axios.get(
            `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${process.env.WEATHER_API_KEY}&units=metric`
        );
        res.json(weatherResponse.data);
    } catch (error) {
        res.status(500).json({ error: "Could not fetch weather data" });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
