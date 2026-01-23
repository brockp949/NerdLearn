import { v4 as uuidv4 } from 'uuid';

export enum EventType {
    MOUSE_MOVE = "mouse_move",
    MOUSE_CLICK = "mouse_click",
    KEY_PRESS = "key_press",
    SCROLL = "scroll",
    FOCUS = "focus",
    BLUR = "blur",
    PAGE_VIEW = "page_view",
    CONTENT_INTERACTION = "content_interaction",
    IDLE_START = "idle_start",
    IDLE_END = "idle_end",
}

export interface TelemetryEvent {
    user_id: string;
    session_id: string;
    event_type: EventType;
    timestamp: number;
    data: Record<string, any>;
    resource_id?: string;
}

export class TelemetryClient {
    private ws: WebSocket | null = null;
    private url: string;
    private userId: string;
    private sessionId: string;
    private isConnected: boolean = false;
    private buffer: TelemetryEvent[] = [];
    private batchSize: number = 20;
    private flushInterval: number = 2000; // 2 seconds
    private flushTimer: NodeJS.Timeout | null = null;
    private reconnectAttempts: number = 0;
    private maxReconnectAttempts: number = 5;

    constructor(url: string, userId: string, sessionId: string) {
        this.url = url;
        this.userId = userId;
        this.sessionId = sessionId;
    }

    public connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        try {
            // Correctly construct the WS URL: ws://host:port/ws/{userId}/{sessionId}
            // Ensure we don't have double slashes if url ends with /
            const baseUrl = this.url.replace(/\/$/, '');
            const wsUrl = `${baseUrl}/ws/${this.userId}/${this.sessionId}`;

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('✅ Telemetry WebSocket Connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.flushBuffer();
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === 'intervention') {
                        console.log('🤖 Intervention received:', message);
                        // Dispatch a custom event for the UI to handle
                        window.dispatchEvent(new CustomEvent('telemetry-intervention', { detail: message }));
                    }
                } catch (e) {
                    console.error('Error parsing telemetry message:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('Telemetry WebSocket Closed');
                this.isConnected = false;
                this.handleReconnect();
            };

            this.ws.onerror = (error) => {
                console.warn('Telemetry WebSocket Error:', error);
                // onError will usually trigger onClose
            };

            // Start periodic flush
            this.startFlushTimer();

        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
        }
    }

    public disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
            this.flushTimer = null;
        }
    }

    public sendEvent(type: EventType, data: Record<string, any> = {}, resourceId?: string) {
        const event: TelemetryEvent = {
            user_id: this.userId,
            session_id: this.sessionId,
            event_type: type,
            timestamp: Date.now() / 1000, // Unix timestamp in seconds (float)
            data,
            resource_id: resourceId
        };

        this.buffer.push(event);

        // If urgent event (like Click or Page View), flush immediately
        if (type !== EventType.MOUSE_MOVE && type !== EventType.SCROLL) {
            this.flushBuffer();
        } else if (this.buffer.length >= this.batchSize) {
            this.flushBuffer();
        }
    }

    private flushBuffer() {
        if (!this.isConnected || this.buffer.length === 0 || !this.ws) {
            return;
        }

        const eventsToSend = [...this.buffer];
        this.buffer = [];

        // If single event, send as object. If multiple, we might need a batch endpoint,
        // but for now the WS endpoint accepts single events. 
        // The Python backend handles single JSON objects.
        // If we want to send batch, we should format as such.
        // However, the backend `websocket_endpoint` loop expects one JSON object at a time.
        // So we iterate.

        eventsToSend.forEach(event => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify(event));
            } else {
                // Re-buffer if connection lost mid-send
                this.buffer.push(event);
            }
        });
    }

    private startFlushTimer() {
        if (this.flushTimer) clearInterval(this.flushTimer);
        this.flushTimer = setInterval(() => {
            this.flushBuffer();
        }, this.flushInterval);
    }

    private handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            const timeout = Math.min(1000 * (2 ** this.reconnectAttempts), 30000);
            this.reconnectAttempts++;
            console.log(`Reconnecting telemetry in ${timeout}ms...`);
            setTimeout(() => this.connect(), timeout);
        }
    }
}
