/**
 * Copyright 2024 Google LLC
 * Licensed under the Apache License, Version 2.0
 */

import React, { useRef, useState, useCallback } from "react";
import "./App.scss";
import { LiveAPIProvider } from "./contexts/LiveAPIContext";
import SidePanel from "./components/side-panel/SidePanel";
import { Altair } from "./components/altair/Altair";
import ControlTray from "./components/control-tray/ControlTray";
import cn from "classnames";
import CaregiverSuggestions from "./components/CaregiverSuggestions";
import FaceEmotionDetector from "./components/FaceEmotionDetector";

type Emotion = "happy" | "sad" | "angry" | "neutral";

export interface DetectedSuggestions {
  emotion: Emotion;
  suggestionForPatient: string;
  suggestionForCaregiver: string;
}

const API_KEY = process.env.REACT_APP_GEMINI_API_KEY as string;
if (typeof API_KEY !== "string") {
  throw new Error("Set REACT_APP_GEMINI_API_KEY in .env");
}

const host = "generativelanguage.googleapis.com";
const uri = `wss://${host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent`;

const EMOTION_PROMPT = `You are an expert in autism spectrum disorders and special needs care. The patient you are evaluating is an autistic individual with sensory sensitivities and communication challenges. Analyze ONLY the facial expressions and body language provided in the input video.
Based on your analysis, respond in plain text using exactly the following format (each on a new line) and do not include any curly brackets, quotation marks, or extra punctuation:

Emotion: <one of happy, sad, angry, neutral>
Suggestion for Patient: <a short message tailored for an autistic patient>
Suggestion for Caregiver: <a short instruction for the caregiver>

For example, if the patient appears sad, your response should be:

Emotion: sad
Suggestion for Patient: It looks like you're feeling low. Try a calm, quiet activity or take a sensory break.
Suggestion for Caregiver: Patient appears sad. Consider offering a sensory-friendly snack and a quiet space for a break.

Do not include any additional text.`;

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const [detectedSuggestions, setDetectedSuggestions] = useState<DetectedSuggestions | null>(null);

  const handleAPIResponse = useCallback((response: any) => {
    console.log("API response:", response);
    try {
      let rawOutput = "";
      // Look for the modelTurn response that contains our text.
      if (response.modelTurn && Array.isArray(response.modelTurn.parts)) {
        rawOutput = response.modelTurn.parts
          .map((part: any) => part.text || "")
          .join("")
          .trim();
      } else if (typeof response.output === "string") {
        rawOutput = response.output.trim();
      }
      console.log("Raw output:", rawOutput);

      // Parse the plain text response.
      // Expected format (each on a new line):
      // Emotion: <emotion>
      // Suggestion for Patient: <patient suggestion>
      // Suggestion for Caregiver: <caregiver suggestion>
      const lines = rawOutput.split("\n").map(line => line.trim()).filter(line => line);
      let emotion: Emotion = "neutral";
      let suggestionForPatient = "";
      let suggestionForCaregiver = "";
      for (const line of lines) {
        if (line.toLowerCase().startsWith("emotion:")) {
          const value = line.substring("emotion:".length).trim().toLowerCase();
          if (["happy", "sad", "angry", "neutral"].includes(value)) {
            emotion = value as Emotion;
          }
        } else if (line.toLowerCase().startsWith("suggestion for patient:")) {
          suggestionForPatient = line.substring("suggestion for patient:".length).trim();
        } else if (line.toLowerCase().startsWith("suggestion for caregiver:")) {
          suggestionForCaregiver = line.substring("suggestion for caregiver:".length).trim();
        }
      }
      setDetectedSuggestions({
        emotion,
        suggestionForPatient,
        suggestionForCaregiver,
      });
    } catch (error) {
      console.error("Error parsing API response:", error);
      setDetectedSuggestions(null);
    }
  }, []);

  // Callback to receive emotion from the custom face detector.
  const handleCustomEmotionDetected = useCallback((emotion: Emotion) => {
    console.log("Custom detected emotion:", emotion);
    setDetectedSuggestions(prev => ({
      emotion,
      suggestionForPatient: prev?.suggestionForPatient || "",
      suggestionForCaregiver: prev?.suggestionForCaregiver || "",
    }));
  }, []);

  return (
    <div className="App">
      <LiveAPIProvider
        url={uri}
        apiKey={API_KEY}
        systemInstruction={{ parts: [{ text: EMOTION_PROMPT }] }}
        onResponse={handleAPIResponse}
      >
        <div className="streaming-console">
          <SidePanel />
          <main>
            <div className="main-app-area">
              <Altair />
              <video
                className={cn("stream", {
                  hidden: !videoRef.current || !videoStream,
                })}
                ref={videoRef}
                autoPlay
                playsInline
              />
              {/* Render the custom face emotion detector */}
              <FaceEmotionDetector videoRef={videoRef} onEmotionDetected={handleCustomEmotionDetected} />
              {detectedSuggestions && (
                <div className="caregiver-feedback">
                  <CaregiverSuggestions suggestions={detectedSuggestions} />
                </div>
              )}
            </div>
            <ControlTray
              videoRef={videoRef}
              supportsVideo={true}
              onVideoStreamChange={setVideoStream}
            />
          </main>
        </div>
      </LiveAPIProvider>
    </div>
  );
}

export default App;
