import React, { useEffect, useState } from "react";
import * as faceapi from "face-api.js";

interface FaceEmotionDetectorProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  onEmotionDetected: (emotion: "happy" | "sad" | "angry" | "neutral") => void;
}

const FaceEmotionDetector: React.FC<FaceEmotionDetectorProps> = ({ videoRef, onEmotionDetected }) => {
  const [modelsLoaded, setModelsLoaded] = useState(false);

  useEffect(() => {
    async function loadModels() {
      try {
        const MODEL_URL = process.env.PUBLIC_URL + "/models";
        // Load the required models
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        setModelsLoaded(true);
        console.log("face-api.js models loaded successfully");
      } catch (error) {
        console.error("Error loading face-api.js models:", error);
      }
    }
    loadModels();
  }, []);

  useEffect(() => {
    // We only start detecting if models are loaded
    if (!modelsLoaded) return;

    let intervalId: NodeJS.Timeout;

    async function detectEmotion() {
      if (!videoRef.current) return;

      // Attempt detection
      const detection = await faceapi
        .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceExpressions();

      if (detection && detection.expressions) {
        // Find the expression with the highest probability
        const expressions = detection.expressions;
        const [detectedEmotion] = Object.entries(expressions).sort(([, a], [, b]) => b - a)[0];

        // Map the detected emotion to our custom set
        const emotionMapping: { [key: string]: "happy" | "sad" | "angry" | "neutral" } = {
          happy: "happy",
          sad: "sad",
          angry: "angry",
          fearful: "neutral",
          disgusted: "neutral",
          surprised: "neutral",
          neutral: "neutral",
        };
        const mappedEmotion = emotionMapping[detectedEmotion] || "neutral";
        onEmotionDetected(mappedEmotion);
      }
    }

    // Run detection every second
    intervalId = setInterval(detectEmotion, 1000);

    return () => clearInterval(intervalId);
  }, [modelsLoaded, videoRef, onEmotionDetected]);

  return null;
};

export default FaceEmotionDetector;
