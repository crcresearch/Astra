// // src/components/MyEmotionCaptureApp.tsx
// import React, { useRef, useEffect } from "react";
// import { useWebcamCapture } from "../hooks/use-webcam";
// import { useFaceApiEmotionDetection } from "../hooks/useFaceApiEmotionDetection";

// export default function MyEmotionCaptureApp() {
//   console.log("[MyEmotionCaptureApp] Rendering component...");

//   const videoRef = useRef<HTMLVideoElement>(null);

//   // 1. Use the new webcam capture logic
//   const { stream, isStreaming, start, stop } = useWebcamCapture();

//   // 2. Use the face-api detection hook
//   const { isModelReady, detectionResult } = useFaceApiEmotionDetection(videoRef);

//   // 3. Attach the stream to the <video> element
//   useEffect(() => {
//     console.log("[MyEmotionCaptureApp] useEffect for attaching stream:", stream);

//     if (videoRef.current && stream) {
//       videoRef.current.srcObject = stream;
//       console.log("[MyEmotionCaptureApp] Stream assigned to videoRef.");
//     }
//   }, [stream]);

//   return (
//     <div style={{ padding: 20 }}>
//       <h1>Webcam + Emotion Detection</h1>

//       <div style={{ marginBottom: 10 }}>
//         <button onClick={start} disabled={isStreaming}>
//           Start Webcam
//         </button>
//         <button onClick={stop} disabled={!isStreaming}>
//           Stop Webcam
//         </button>
//       </div>

//       <div>
//         <video
//           ref={videoRef}
//           style={{ width: 640, height: 360, background: "#000" }}
//           autoPlay
//           playsInline
//           muted
//         />
//       </div>

//       <div style={{ marginTop: 20 }}>
//         <p>
//           <strong>Model status:</strong>{" "}
//           {isModelReady ? "Ready ✅" : "Loading..."}
//         </p>
//         {detectionResult ? (
//           <div>
//             <p>
//               <strong>Best Expression:</strong>{" "}
//               {detectionResult.bestExpression} (
//               {Math.round(detectionResult.bestExpressionScore * 100)}%)
//             </p>
//             <pre>{JSON.stringify(detectionResult.expressions, null, 2)}</pre>
//           </div>
//         ) : (
//           <p>No face detected or webcam not started yet.</p>
//         )}
//       </div>
//     </div>
//   );
// }


// // // src/components/MyEmotionCaptureApp.tsx
// // import React, { useRef, useEffect } from "react";
// // import { useFaceApiEmotionDetection } from "../hooks/useFaceApiEmotionDetection";
// // import { useLiveAPIContext } from "../contexts/LiveAPIContext";

// // const MyEmotionCaptureApp: React.FC = () => {
// //   const videoRef = useRef<HTMLVideoElement>(null);
// //   const { client } = useLiveAPIContext();

// //   // Callback function to handle detected emotions
// //   const handleEmotionDetected = (emotion: string) => {
// //     console.log(`[MyEmotionCaptureApp] Detected Emotion: ${emotion}`);

// //     // Implement cooldown to prevent spamming
// //     if (!handleEmotionDetected.cooldown) {
// //       const response = `Oh, you are feeling ${emotion}.`;
// //       client.send([{ text: response }]);
// //       console.log(`[MyEmotionCaptureApp] Sent emotion response: "${response}"`);

// //       // Activate cooldown
// //       handleEmotionDetected.cooldown = true;
// //       setTimeout(() => {
// //         handleEmotionDetected.cooldown = false;
// //       }, 5000); // 5-second cooldown
// //     }
// //   };
// //   // Initialize cooldown property
// //   handleEmotionDetected.cooldown = false;

// //   // 2. Use the face-api detection hook with the callback
// //   const { isModelReady, detectionResult } = useFaceApiEmotionDetection(videoRef, handleEmotionDetected);

// //   // 3. Attach the stream to the <video> element
// //   useEffect(() => {
// //     const startWebcam = async () => {
// //       try {
// //         const stream = await navigator.mediaDevices.getUserMedia({ video: true });
// //         if (videoRef.current) {
// //           videoRef.current.srcObject = stream;
// //           videoRef.current.play();
// //         }
// //       } catch (err) {
// //         console.error("Error accessing webcam:", err);
// //       }
// //     };

// //     startWebcam();

// //     return () => {
// //       if (videoRef.current && videoRef.current.srcObject) {
// //         (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
// //       }
// //     };
// //   }, []);

// //   return (
// //     <div>
// //       <h1>Emotion Capture App</h1>
// //       <video ref={videoRef} style={{ width: "640px", height: "480px" }} muted playsInline />
// //       <div>
// //         {isModelReady ? (
// //           detectionResult ? (
// //             <div>
// //               <p>Detected Emotion: {detectionResult.bestExpression}</p>
// //             </div>
// //           ) : (
// //             <p>No face detected.</p>
// //           )
// //         ) : (
// //           <p>Loading models...</p>
// //         )}
// //       </div>
// //     </div>
// //   );
// // };

// // export default MyEmotionCaptureApp;
