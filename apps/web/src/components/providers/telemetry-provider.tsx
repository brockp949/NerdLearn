import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { TelemetryClient, EventType } from '@/lib/telemetry-client';
import { v4 as uuidv4 } from 'uuid';
import throttle from 'lodash.throttle';

interface TelemetryContextType {
    client: TelemetryClient | null;
    sessionId: string;
}

const TelemetryContext = createContext<TelemetryContextType>({
    client: null,
    sessionId: '',
});

export const useTelemetry = () => useContext(TelemetryContext);

interface TelemetryProviderProps {
    children: React.ReactNode;
    userId?: string; // Optional: if not provided, can be anonymous or fetched from auth context
}

export const TelemetryProvider: React.FC<TelemetryProviderProps> = ({ children, userId }) => {
    const [client, setClient] = useState<TelemetryClient | null>(null);
    const [sessionId] = useState<string>(() => {
        // Persist session ID in sessionStorage to maintain across reloads, or new per tab
        if (typeof window !== 'undefined') {
            const stored = sessionStorage.getItem('nl_session_id');
            if (stored) return stored;
            const newId = uuidv4();
            sessionStorage.setItem('nl_session_id', newId);
            return newId;
        }
        return uuidv4();
    });

    const idleTimerRef = useRef<NodeJS.Timeout | null>(null);
    const IDLE_THRESHOLD_MS = 10000; // 10 seconds for testing, 60s for prod

    // We use a ref for the client to access it in event listeners without dependency cycles
    const clientRef = useRef<TelemetryClient | null>(null);

    useEffect(() => {
        // Only connect if we are in browser
        if (typeof window === 'undefined') return;

        // Use a default user ID if none provided (e.g. "anonymous") or wait for auth
        const effectiveUserId = userId || 'anonymous';
        const telemetryUrl = process.env.NEXT_PUBLIC_TELEMETRY_URL || 'ws://localhost:8002';

        const newClient = new TelemetryClient(telemetryUrl, effectiveUserId, sessionId);
        newClient.connect();

        setClient(newClient);
        clientRef.current = newClient;

        return () => {
            newClient.disconnect();
        };
    }, [userId, sessionId]);

    useEffect(() => {
        if (!client) return;

        const handleMouseMove = throttle((e: MouseEvent) => {
            resetIdleTimer();
            clientRef.current?.sendEvent(EventType.MOUSE_MOVE, { x: e.clientX, y: e.clientY });
        }, 100); // 100ms throttle = 10 events/sec max

        const handleClick = (e: MouseEvent) => {
            resetIdleTimer();
            clientRef.current?.sendEvent(EventType.MOUSE_CLICK, { x: e.clientX, y: e.clientY });
        };

        const handleScroll = throttle((e: Event) => {
            resetIdleTimer();
            clientRef.current?.sendEvent(EventType.SCROLL, {
                scrollY: window.scrollY,
                percentage: window.scrollY / (document.body.scrollHeight - window.innerHeight)
            });
        }, 500);

        const handleKeyDown = (e: KeyboardEvent) => {
            resetIdleTimer();
            // Don't send every key, maybe just activity indicator
            // or specific keys if in focused input (privacy concern)
            // For now, just reset idle
        };

        const resetIdleTimer = () => {
            if (idleTimerRef.current) {
                clearTimeout(idleTimerRef.current);
                // If we were idle, send IDLE_END? Tracking idle state complexity...
                // Simple version: just cancel the start timer
            }
            idleTimerRef.current = setTimeout(() => {
                clientRef.current?.sendEvent(EventType.IDLE_START, {});
            }, IDLE_THRESHOLD_MS);
        };

        // Attach listeners
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('click', handleClick);
        window.addEventListener('scroll', handleScroll);
        window.addEventListener('keydown', handleKeyDown);

        // Initial page view
        client.sendEvent(EventType.PAGE_VIEW, { path: window.location.pathname });

        // Initial idle timer
        resetIdleTimer();

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('click', handleClick);
            window.removeEventListener('scroll', handleScroll);
            window.removeEventListener('keydown', handleKeyDown);
            if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
        };
    }, [client]);

    return (
        <TelemetryContext.Provider value={{ client, sessionId }}>
            {children}
        </TelemetryContext.Provider>
    );
};
