"use client";

import React, { createContext, useContext, useEffect, useState } from 'react';
import { TelemetryClient, EngagementScore, Intervention } from '@/lib/telemetry';
import { v4 as uuidv4 } from 'uuid';

interface TelemetryContextType {
    client: TelemetryClient | null;
    sessionId: string;
    engagement: EngagementScore | null;
    intervention: Intervention | null;
    connected: boolean;
}

const TelemetryContext = createContext<TelemetryContextType>({
    client: null,
    sessionId: '',
    engagement: null,
    intervention: null,
    connected: false,
});

export const useTelemetryContext = () => useContext(TelemetryContext);

interface TelemetryProviderProps {
    children: React.ReactNode;
    userId?: string;
}

export const TelemetryProvider: React.FC<TelemetryProviderProps> = ({ children, userId }) => {
    const [client, setClient] = useState<TelemetryClient | null>(null);
    const [engagement, setEngagement] = useState<EngagementScore | null>(null);
    const [intervention, setIntervention] = useState<Intervention | null>(null);
    const [connected, setConnected] = useState(false);

    const [sessionId] = useState<string>(() => {
        if (typeof window !== 'undefined') {
            const stored = sessionStorage.getItem('nl_session_id');
            if (stored) return stored;
            const newId = uuidv4();
            sessionStorage.setItem('nl_session_id', newId);
            return newId;
        }
        return uuidv4();
    });

    useEffect(() => {
        if (typeof window === 'undefined') return;

        const effectiveUserId = userId || 'anonymous';
        const telemetryUrl = process.env.NEXT_PUBLIC_TELEMETRY_URL || 'ws://localhost:8002/ws';

        const newClient = new TelemetryClient({
            telemetryUrl,
            learnerId: effectiveUserId,
            sessionId,
            throttleMs: 100
        });

        newClient.onEngagement(setEngagement);
        newClient.onIntervention(setIntervention);
        newClient.onConnection(setConnected);

        newClient.connect();
        setClient(newClient);

        // Report metrics on unmount (end of session)
        return () => {
            newClient.reportSessionMetrics({
                total_dwell_ms: 0, // In real imp, track this
                valid_dwell_ms: 0,
                engagement_score: 0.5
            });
            newClient.disconnect();
        };
    }, [userId, sessionId]);

    return (
        <TelemetryContext.Provider value={{ client, sessionId, engagement, intervention, connected }}>
            {children}
        </TelemetryContext.Provider>
    );
};
