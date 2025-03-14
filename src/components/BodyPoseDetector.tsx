import React, { useEffect, useRef } from "react";
import { Pose } from "@mediapipe/pose";
import * as cam from "@mediapipe/camera_utils";

interface BodyPoseDetectorProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  onPoseDetected: (poseLabel: string) => void;
}

const BodyPoseDetector: React.FC<BodyPoseDetectorProps> = ({ videoRef, onPoseDetected }) => {
  const cameraRef = useRef<cam.Camera | null>(null);

  useEffect(() => {
    if (!videoRef.current) return;

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    // Label the pose - replace with soemthing more meaningful
    pose.onResults((results) => {
      if (!results.poseLandmarks) {
        onPoseDetected("No Pose Detected");
        return;
      }
      const label = interpretPose(results.poseLandmarks);
      onPoseDetected(label);
    });

    // Use MediaPipe's Camera helper to stream frames from <video> to Pose
    cameraRef.current = new cam.Camera(videoRef.current, {
      onFrame: async () => {
        if (!videoRef.current) return;
        await pose.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });
    cameraRef.current.start();

    return () => {
      cameraRef.current?.stop();
    };
  }, [videoRef, onPoseDetected]);

  function interpretPose(landmarks: any): string {
    // This is just a placeholder, do something more complex
    return "neutral-pose";
  }

  return null;
};

export default BodyPoseDetector;
