/**
 * Copyright 2024 Google LLC
 * Licensed under the Apache License, Version 2.0
 */

import { createContext, FC, ReactNode, useContext } from "react";
import { useLiveAPI, UseLiveAPIResults } from "../hooks/use-live-api";
import { LiveConfig } from "../multimodal-live-types";

interface SystemInstruction {
  parts: Array<{ text: string }>;
}

export type LiveAPIProviderProps = {
  children: ReactNode;
  url?: string;
  apiKey: string;
  systemInstruction?: SystemInstruction;
  onResponse?: (response: any) => void;
};

const LiveAPIContext = createContext<UseLiveAPIResults | undefined>(undefined);

export const LiveAPIProvider: FC<LiveAPIProviderProps> = ({
  url,
  apiKey,
  children,
  systemInstruction,
  onResponse
}) => {
  const liveAPI = useLiveAPI({
    url,
    apiKey,
    systemInstruction,
    onResponse
  });

  return (
    <LiveAPIContext.Provider value={liveAPI}>
      {children}
    </LiveAPIContext.Provider>
  );
};

export const useLiveAPIContext = () => {
  const context = useContext(LiveAPIContext);
  if (!context) {
    throw new Error("useLiveAPIContext must be used within a LiveAPIProvider");
  }
  return context;
};