// App.jsx
import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import axios from "axios";
import { BrowserRouter } from 'react-router-dom';

const dictionary = {
  "./pdfs\\1.-GUIDELINES-FOR-APPOINTMENT-OF-EXAMINERS-FOR-Ph.D.-THESIS-EXAMINATION.pdf": "https://drive.google.com/drive/folders/1_uMRE3LMgZNQfUNAOfzYRErZsNf5bfmT",
  "./pdfs\\BITS-Pilani-International-Travel-Award_Guidelines-1.pdf":"2",
  "./pdfs\\10.-Check-list-for-proposal-submission.pdf":"https://drive.google.com/drive/folders/1_uMRE3LMgZNQfUNAOfzYRErZsNf5bfmT",
  "./pdfs\\CheckList_PhD-Thesis-submission.pdf": "4",
   "./pdfs\\Documents_required.pdf":"5",
   "./pdfs\\DRC_Guidelines-2015-updated.pdf":"6",
   "./pdfs\\GCIR SOP_Hyd_11oct.pdf":"7",
   "./pdfs\\Guidelines_for-PhD-Proposal.pdf":"8",
   "./pdfs\\Guidelines-for-Recruiting-JRF-.pdf":"9",
   "./pdfs\\Leave-policy-for-the-institute-supported-PhD-students-1.pdf":"10",
   "./pdfs\\Leave-policy-for-the-institute-supported-PhD-students.pdf":"11",
   "./pdfs\\List-of-Sub-areas.pdf":"12",
   "./pdfs\\PhD Guideline Brochure_2019.pdf":"13",

};



function App() {

  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hi, I'm your BITS Pilani document assistant. What would you like to know?",
      sender: "bot",
      timestamp: new Date(),
      source: [""],
    },
  ]);
  const [message1, setMessage1] = useState([
    {
      role: "system",
      content:
        "Hi, I'm your BITS Pilani document assistant. What would you like to know?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus on input field on component mount
  useEffect(() => {
    inputRef.current.focus();
  }, []);

  const handleSend = async (e) => {
    e.preventDefault();

    if (input.trim() === "") return;

    // Add user message
    const userMessage = {
      id: messages.length + 1,
      text: input,
      sender: "user",
      timestamp: new Date(),
    };

    const userMessage1 = {
      role: "user",
      content: input,
    };
    const updatedMessage1 = [...message1, userMessage1]; // use this directly

    setMessages((prev) => [...prev, userMessage]);
    setMessage1(updatedMessage1);
    setInput("");
    setIsLoading(true);

    try {
      // Await the response from the POST request
      const response = await axios.post("http://localhost:5000/respond", {
        message: updatedMessage1,
      });

      const responseMessage = response.data;
      console.log(responseMessage);

      setMessage1((prev) => [
        ...prev,
        { role: "system", content: responseMessage },
      ]);
      const lines = responseMessage.match(/^.*\bSOURCE\b.*$/gm);
      // Remove "SOURCE:" from lines
      const cleanedLines = [
        ...new Set(lines.map((line) => line.replace(/SOURCE:/, "").trim())),
      ];

      console.log(cleanedLines);

      const cleanedMessage = responseMessage.replace(
        /^.*\bSOURCE\b.*\n?/gm,
        ""
      );

      // Add bot response
      const botResponse = {
        id: messages.length + 2,
        text: cleanedMessage, // Assuming the response has the text as data
        sender: "bot",
        timestamp: new Date(),
        sources: cleanedLines,
      };

      setMessages((prev) => [...prev, botResponse]);
    } catch (error) {
      console.error("Error sending request:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Format timestamp
  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <BrowserRouter>

    <div className="app-container">
      <div className="chat-container">
        <header className="chat-header">
          <div className="brand-container">
            <div className="logo">BITS</div>
            <h1>Document Assistant</h1>
          </div>
        </header>

        <div className="messages-container">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${
                message.sender === "user" ? "user-message" : "bot-message"
              }`}
            >
              <div className="message-bubble">
                <p>{message.text}</p>

                <div className="sources">
                  {message.sources?.map((src, i) => (
                    <div key={i} className="source"> 
                    <button
                      key={i}
                      className="source-button"
                      onClick={() => window.open("https://drive.google.com/drive/folders/1_uMRE3LMgZNQfUNAOfzYRErZsNf5bfmT?usp=sharing")}
                    >
                      {src}
                    </button>
                    </div>
                  ))}
                </div>
                <span className="timestamp">
                  {formatTime(message.timestamp)}
                </span>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message bot-message">
              <div className="message-bubble loading">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="input-area" onSubmit={handleSend}>
          <input
            type="text"
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything about BITS Pilani..."
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={input.trim() === "" || isLoading}
            className={input.trim() === "" ? "disabled" : ""}
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M22 2L11 13"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M22 2L15 22L11 13L2 9L22 2Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </form>
      </div>
    </div>
    </BrowserRouter>

  );
}

export default App;
