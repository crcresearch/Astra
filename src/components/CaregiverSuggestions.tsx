import React from "react";

type Emotion = "happy" | "sad" | "angry" | "neutral";

export interface DetectedSuggestions {
  emotion: Emotion;
  suggestionForPatient: string;
  suggestionForCaregiver: string;
}

interface CaregiverSuggestionsProps {
  suggestions: DetectedSuggestions;
}

const CaregiverSuggestions: React.FC<CaregiverSuggestionsProps> = ({ suggestions }) => {
  return (
    <div className="caregiver-suggestions">
      <h3>Detected Emotion: {suggestions.emotion.toUpperCase()}</h3>
      <div className="suggestion-item">
        <strong>Suggestion for You:</strong>
        <p>{suggestions.suggestionForPatient}</p>
      </div>
      <div className="suggestion-item">
        <strong>Suggestion for Caregiver:</strong>
        <p>{suggestions.suggestionForCaregiver}</p>
      </div>
    </div>
  );
};

export default CaregiverSuggestions;
