/**
 * Copyright 2024 Google LLC
 * Licensed under the Apache License, Version 2.0
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  MultimodalLiveAPIClientConnection,
  MultimodalLiveClient,
} from "../lib/multimodal-live-client";
import { LiveConfig } from "../multimodal-live-types";
import { AudioStreamer } from "../lib/audio-streamer";
import { audioContext } from "../lib/utils";
import VolMeterWorket from "../lib/worklets/vol-meter";

export type UseLiveAPIResults = {
  client: MultimodalLiveClient;
  setConfig: (config: LiveConfig) => void;
  config: LiveConfig;
  connected: boolean;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  volume: number;
};

export interface UseLiveAPIOptions extends MultimodalLiveAPIClientConnection {
  systemInstruction?: {
    parts: Array<{ text: string }>;
  };
  onResponse?: (response: any) => void;
}

/**
 * useLiveAPI is a custom hook that sets up the Multimodal Live API client along with audio streaming.
 */
export function useLiveAPI({
  url,
  apiKey,
  systemInstruction,
  onResponse,
}: UseLiveAPIOptions): UseLiveAPIResults {
  const client = useMemo(
    () => new MultimodalLiveClient({ url, apiKey }),
    [url, apiKey]
  );
  const audioStreamerRef = useRef<AudioStreamer | null>(null);

  const [config, setConfig] = useState<LiveConfig>({
    model: "models/gemini-2.0-flash-exp",
  });
  const [connected, setConnected] = useState(false);
  const [volume, setVolume] = useState(0);

  // Listen for "content" events from the client (which serve as our response).
  useEffect(() => {
    if (onResponse) {
      client.on("content", onResponse);
    }
    return () => {
      if (onResponse) {
        client.off("content", onResponse);
      }
    };
  }, [client, onResponse]);

  // Setup the audio streamer.
  useEffect(() => {
    if (!audioStreamerRef.current) {
      audioContext({ id: "audio-out" }).then((audioCtx: AudioContext) => {
        audioStreamerRef.current = new AudioStreamer(audioCtx);
        audioStreamerRef.current
          .addWorklet<any>("vumeter-out", VolMeterWorket, (ev: any) => {
            setVolume(ev.data.volume);
          })
          .catch((err) => {
            console.error("Failed to add worklet:", err);
          });
      });
    }
  }, []);

  // Register event listeners for connection-related events.
  useEffect(() => {
    const onClose = () => {
      setConnected(false);
    };

    const stopAudioStreamer = () => audioStreamerRef.current?.stop();

    const onAudio = (data: ArrayBuffer) =>
      audioStreamerRef.current?.addPCM16(new Uint8Array(data));

    client.on("close", onClose)
      .on("interrupted", stopAudioStreamer)
      .on("audio", onAudio);

    return () => {
      client.off("close", onClose);
      client.off("interrupted", stopAudioStreamer);
      client.off("audio", onAudio);
    };
  }, [client]);

  const connect = useCallback(async () => {
    if (!config) throw new Error("Config not set");
    const fullConfig: LiveConfig = systemInstruction
      ? {
          ...config,
          systemInstruction: {
            parts: [
              ...systemInstruction.parts,
              { text: "RESPOND ONLY WITH LOWERCASE EMOTION" },
            ],
          },
        }
      : {
          ...config,
          systemInstruction: {
            parts: [{ text: "RESPOND ONLY WITH LOWERCASE EMOTION" }],
          },
        };

    client.disconnect();
    await client.connect(fullConfig);
    setConnected(true);
  }, [client, config, systemInstruction]);

  const disconnect = useCallback(async () => {
    client.disconnect();
    setConnected(false);
  }, [client]);

  return {
    client,
    config,
    setConfig,
    connected,
    connect,
    disconnect,
    volume,
  };
}
