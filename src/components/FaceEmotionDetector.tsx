import React, { useEffect } from "react";
import * as faceapi from "face-api.js";

interface FaceEmotionDetectorProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  onEmotionDetected: (emotion: "happy" | "sad" | "angry" | "neutral") => void;
}

const FaceEmotionDetector: React.FC<FaceEmotionDetectorProps> = ({ videoRef, onEmotionDetected }) => {
  // Load models when the component mounts.
  useEffect(() => {
    async function loadModels() {
      // Make sure these files exist in: /public/models
      const MODEL_URL = process.env.PUBLIC_URL + "/models";
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
    }
    loadModels();
  }, []);

  // Periodically detect face and emotion.
  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    async function detectEmotion() {
      if (videoRef.current) {
        const detection = await faceapi
          .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceExpressions();

        if (detection && detection.expressions) {
          // Find the expression with the highest probability
          const expressions = detection.expressions;
          const [detectedEmotion] = Object.entries(expressions).sort(([, a], [, b]) => b - a)[0];

          // Map the detected emotion to our custom "happy | sad | angry | neutral" set
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
    }

    // Run detection every second (adjust as needed)
    intervalId = setInterval(detectEmotion, 1000);

    return () => clearInterval(intervalId);
  }, [videoRef, onEmotionDetected]);

  return null;
};

export default FaceEmotionDetector;
