'use client';

import React, { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { BellIcon } from 'lucide-react';

export default function EmotionPage() {
  const [patients, setPatients] = useState([]);

  useEffect(() => {
    let lastEmotions = {};

    const fetchData = async () => {
      try {
        const res = await fetch('https://cpystjo4c9.execute-api.us-east-1.amazonaws.com/production/emotion');
        const data = await res.json();
        setPatients(data);

        // Trigger logic: alert when someone becomes distressed
        data.forEach((p) => {
          if (p.emotion === 'distressed' && lastEmotions[p.clientid] !== 'distressed') {
            alert(`🚨 User ${p.clientid} is now distressed!`);
          }
          lastEmotions[p.clientid] = p.emotion;
        });
      } catch (error) {
        console.error('Failed to fetch emotions:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const emotionColor = (emotion) => {
    switch (emotion) {
      case 'happy': return 'bg-green-100 text-green-800';
      case 'sad': return 'bg-blue-100 text-blue-800';
      case 'distressed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="p-6 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
      {patients.map((patient) => (
        <Card key={patient.clientid} className="rounded-2xl shadow-md">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold">User {patient.clientid}</h2>
              <BellIcon className="w-5 h-5" />
            </div>
            <div className={`mt-2 inline-block px-3 py-1 rounded-full text-sm font-medium ${emotionColor(patient.emotion)}`}>
              {patient.emotion.toUpperCase()}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
