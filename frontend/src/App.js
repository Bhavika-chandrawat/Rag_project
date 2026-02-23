import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const uploadFile = async () => {
  if (!file) {
    alert("Please select a file first");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    await axios.post("http://127.0.0.1:8000/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    alert("Document uploaded successfully");
  } catch (error) {
    console.error("Upload error:", error);
    alert("Upload failed. Check backend or CORS.");
  }
};


  const askQuestion = async () => {
    const response = await axios.post(
      `http://127.0.0.1:8000/ask?query=${question}`
    );

    setMessages([
      ...messages,
      { role: "user", text: question },
      { role: "bot", text: response.data.answer },
    ]);

    setQuestion("");
  };

  return (
    <div className="container">
      <h1>Technology Document RAG</h1>

      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={uploadFile}>Upload</button>

      <div className="chat">
        {messages.map((msg, index) => (
          <div key={index} className={msg.role}>
            <strong>{msg.role === "user" ? "You" : "AI"}:</strong> {msg.text}
          </div>
        ))}
      </div>

      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question..."
      />
      <button onClick={askQuestion}>Ask</button>
    </div>
  );
}

export default App;
