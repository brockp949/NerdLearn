/**
 * Telemetry Client - Real-time behavioral tracking
 *
 * Connects to Telemetry service via WebSocket to track:
 * - Mouse movements (velocity, entropy, saccades)
 * - Dwell time (time spent on content)
 * - Engagement scoring (real-time cognitive load assessment)
 *
 * Implements Evidence-Centered Design (ECD) for stealth assessment
 */

export interface MouseEvent {
    x: number
    y: number
    timestamp: number
}

export interface EngagementScore {
    score: number // 0-1
    cognitive_load: 'low' | 'medium' | 'high'
    attention_level: 'low' | 'medium' | 'high'
    timestamp: string
}

export interface ScrollEvent {
    depth_percentage: number
    max_depth_percentage: number
    timestamp: number
}

export interface Intervention {
    message: string
    action: string
    type: 'prompt' | 'notification'
}


export interface TelemetryConfig {
    telemetryUrl?: string
    sessionId: string
    learnerId: string
    throttleMs?: number
}

export class TelemetryClient {
    private ws: WebSocket | null = null
    private sessionId: string
    private learnerId: string
    private telemetryUrl: string
    private throttleMs: number
    private mouseBuffer: MouseEvent[] = []
    private lastSent: number = 0
    private connected: boolean = false
    private reconnectAttempts: number = 0
    private maxReconnectAttempts: number = 5
    private reconnectDelay: number = 2000
    private connectionErrorLogged: boolean = false
    private readonly MAX_BUFFER_SIZE = 100 // Prevent memory issues

    // Idle Detection
    private lastActivity: number = Date.now()
    private isIdle: boolean = false
    private idleThreshold: number = 30000 // 30s
    private idleCheckInterval: NodeJS.Timeout | null = null

    // Scroll Tracking
    private maxScrollDepth: number = 0

    // Callbacks
    private onEngagementUpdate?: (score: EngagementScore) => void
    private onInterventionTrigger?: (intervention: Intervention) => void
    private onConnectionChange?: (connected: boolean) => void

    constructor(config: TelemetryConfig) {
        this.sessionId = config.sessionId
        this.learnerId = config.learnerId
        this.telemetryUrl = config.telemetryUrl || 'ws://localhost:8002/ws'
        this.throttleMs = config.throttleMs || 50 // Send every 50ms max

        // Start idle check
        this.startIdleCheck()

        // Add event listeners for global activity
        if (typeof window !== 'undefined') {
            window.addEventListener('scroll', this.handleScroll.bind(this), { passive: true })
            window.addEventListener('click', this.resetIdle.bind(this))
            window.addEventListener('keydown', this.resetIdle.bind(this))
        }
    }

    /**
     * Connect to Telemetry service WebSocket
     */
    connect() {
        try {
            this.ws = new WebSocket(this.telemetryUrl)

            this.ws.onopen = () => {
                console.log('✅ Telemetry connected')
                this.connected = true
                this.reconnectAttempts = 0
                this.connectionErrorLogged = false

                // Send initial handshake
                this.send({
                    type: 'init',
                    session_id: this.sessionId,
                    learner_id: this.learnerId,
                    timestamp: Date.now()
                })

                if (this.onConnectionChange) {
                    this.onConnectionChange(true)
                }
            }

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    this.handleMessage(data)
                } catch (error) {
                    console.error('Failed to parse telemetry message:', error)
                }
            }

            this.ws.onerror = () => {
                if (!this.connectionErrorLogged) {
                    console.warn('Telemetry service unavailable (this is normal if backend is not running)')
                    this.connectionErrorLogged = true
                }
            }

            this.ws.onclose = () => {
                this.connected = false

                if (this.onConnectionChange) {
                    this.onConnectionChange(false)
                }

                // Attempt reconnection silently
                this.attemptReconnect()
            }

        } catch (error) {
            console.error('Failed to connect to telemetry:', error)
            this.connected = false
        }
    }

    /**
     * Attempt to reconnect with exponential backoff
     */
    private attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            // Only log once when max attempts reached
            if (this.reconnectAttempts === this.maxReconnectAttempts) {
                console.info('Telemetry: max reconnection attempts reached, running in offline mode')
            }
            return
        }

        this.reconnectAttempts++
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)

        setTimeout(() => {
            this.connect()
        }, delay)
    }

    /**
     * Handle incoming messages from server
     */
    private handleMessage(data: any) {
        switch (data.type) {
            case 'engagement_score':
                if (this.onEngagementUpdate) {
                    this.onEngagementUpdate({
                        score: data.score,
                        cognitive_load: data.cognitive_load,
                        attention_level: data.attention_level,
                        timestamp: data.timestamp
                    })
                }
                break

            case 'intervention':
                if (this.onInterventionTrigger) {
                    this.onInterventionTrigger({
                        message: data.message,
                        action: data.action,
                        type: data.intervention_type || 'prompt'
                    })
                }
                break

            case 'ack':
                // Acknowledgment received
                break

            case 'error':
                console.error('Telemetry error:', data.message)
                break

            default:
                console.warn('Unknown telemetry message type:', data.type)
        }
    }

    /**
     * Send data to telemetry service
     */
    private send(data: any) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return
        }

        try {
            this.ws.send(JSON.stringify(data))
        } catch (error) {
            console.error('Failed to send telemetry data:', error)
        }
    }

    /**
     * Track mouse movement
     * Throttled to avoid overwhelming the server
     */
    trackMouseMove(event: { clientX: number; clientY: number }) {
        if (!this.connected) return

        const mouseEvent: MouseEvent = {
            x: event.clientX,
            y: event.clientY,
            timestamp: Date.now()
        }

        this.mouseBuffer.push(mouseEvent)

        // Flush if buffer exceeds size limit (prevents memory issues)
        if (this.mouseBuffer.length >= this.MAX_BUFFER_SIZE) {
            this.flushMouseBuffer()
            this.lastSent = Date.now()
            this.resetIdle()
            return
        }

        // Throttle sending
        const now = Date.now()
        if (now - this.lastSent >= this.throttleMs) {
            this.flushMouseBuffer()
            this.lastSent = now
        }

        this.resetIdle()
    }

    /**
     * Flush accumulated mouse events
     */
    private flushMouseBuffer() {
        if (this.mouseBuffer.length === 0) return

        this.send({
            type: 'mouse_events',
            session_id: this.sessionId,
            events: this.mouseBuffer,
            count: this.mouseBuffer.length
        })

        this.mouseBuffer = []
    }

    /**
     * Track dwell time on a card
     */
    trackDwellTime(cardId: string, dwellTimeMs: number) {
        if (!this.connected) return

        this.send({
            type: 'dwell_time',
            session_id: this.sessionId,
            card_id: cardId,
            dwell_time_ms: dwellTimeMs,
            timestamp: Date.now()
        })
    }

    /**
     * Track card interaction (show answer, rating, etc.)
     */
    trackInteraction(cardId: string, interactionType: string, data?: any) {
        if (!this.connected) return

        this.send({
            type: 'interaction',
            session_id: this.sessionId,
            card_id: cardId,
            interaction_type: interactionType,
            data: data || {},
            timestamp: Date.now()
        })
    }

    /**
     * Track hesitation (pauses in interaction)
     */
    trackHesitation(cardId: string, hesitationCount: number) {
        if (!this.connected) return

        this.send({
            type: 'hesitation',
            session_id: this.sessionId,
            card_id: cardId,
            hesitation_count: hesitationCount,
            timestamp: Date.now()
        })
    }

    /**
     * Scroll Tracking
     */
    private handleScroll() {
        if (!this.connected) return

        const scrollTop = window.scrollY
        const docHeight = document.documentElement.scrollHeight - window.innerHeight
        const scrollPercent = Math.min(100, Math.max(0, (scrollTop / docHeight) * 100))

        if (scrollPercent > this.maxScrollDepth) {
            this.maxScrollDepth = scrollPercent

            this.send({
                type: 'scroll',
                session_id: this.sessionId,
                depth_percentage: scrollPercent,
                max_depth_percentage: this.maxScrollDepth,
                timestamp: Date.now()
            })
        }

        this.resetIdle()
    }

    /**
     * Idle Detection Logic
     */
    private startIdleCheck() {
        if (this.idleCheckInterval) clearInterval(this.idleCheckInterval)

        this.idleCheckInterval = setInterval(() => {
            const now = Date.now()
            if (!this.isIdle && (now - this.lastActivity > this.idleThreshold)) {
                this.isIdle = true
                console.log('💤 User is idle')

                this.send({
                    type: 'idle_start',
                    session_id: this.sessionId,
                    duration_ms: now - this.lastActivity,
                    timestamp: now
                })
            }
        }, 1000)
    }

    private resetIdle() {
        this.lastActivity = Date.now()
        if (this.isIdle) {
            this.isIdle = false
            console.log('⚡ User is active')

            this.send({
                type: 'idle_end',
                session_id: this.sessionId,
                timestamp: Date.now()
            })
        }
    }

    /**
     * Register callback for interventions
     */
    onIntervention(callback: (intervention: Intervention) => void) {
        this.onInterventionTrigger = callback
    }

    /**
     * Register callback for engagement score updates
     */
    onEngagement(callback: (score: EngagementScore) => void) {
        this.onEngagementUpdate = callback
    }

    /**
     * Register callback for connection status changes
     */
    onConnection(callback: (connected: boolean) => void) {
        this.onConnectionChange = callback
    }

    /**
     * Disconnect from telemetry service
     */
    disconnect() {
        if (this.ws) {
            // Flush any remaining mouse events
            this.flushMouseBuffer()

            // Send disconnect message
            this.send({
                type: 'disconnect',
                session_id: this.sessionId,
                timestamp: Date.now()
            })

            this.ws.close()
            this.ws = null
            this.connected = false
        }

        if (this.idleCheckInterval) {
            clearInterval(this.idleCheckInterval)
        }

        if (typeof window !== 'undefined') {
            window.removeEventListener('scroll', this.handleScroll)
            window.removeEventListener('click', this.resetIdle)
            window.removeEventListener('keydown', this.resetIdle)
        }
    }

    /**
     * Report session metrics to Analytics API
     */
    async reportSessionMetrics(metrics: { total_dwell_ms: number, valid_dwell_ms: number, engagement_score: number }) {
        try {
            // Use fetch to post to API
            await fetch('/api/analytics/metrics/session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: 1, // TODO: Get actual user ID from auth context or config
                    session_id: this.sessionId,
                    total_dwell_ms: metrics.total_dwell_ms,
                    valid_dwell_ms: metrics.valid_dwell_ms,
                    engagement_score: metrics.engagement_score
                })
            })
            console.log('📊 Session metrics reported')
        } catch (error) {
            console.error('Failed to report session metrics:', error)
        }
    }

    /**
     * Check if connected
     */
    isConnected(): boolean {
        return this.connected
    }
}

/**
 * Hook for using telemetry in React components
 */
export function useTelemetry(sessionId: string, learnerId: string) {
    const [client, setClient] = React.useState<TelemetryClient | null>(null)
    const [engagement, setEngagement] = React.useState<EngagementScore | null>(null)
    const [intervention, setIntervention] = React.useState<Intervention | null>(null)
    const [connected, setConnected] = React.useState(false)

    React.useEffect(() => {
        if (!sessionId || !learnerId) return

        const telemetryClient = new TelemetryClient({
            sessionId,
            learnerId,
            telemetryUrl: process.env.NEXT_PUBLIC_TELEMETRY_URL || 'ws://localhost:8002/ws',
            throttleMs: 50
        })

        telemetryClient.onEngagement(setEngagement)
        telemetryClient.onIntervention(setIntervention)
        telemetryClient.onConnection(setConnected)
        telemetryClient.connect()

        setClient(telemetryClient)

        return () => {
            telemetryClient.disconnect()
        }
    }, [sessionId, learnerId])

    return {
        client,
        engagement,
        intervention,
        connected,
        trackMouseMove: (e: { clientX: number; clientY: number }) => client?.trackMouseMove(e),
        trackDwellTime: (cardId: string, dwellMs: number) => client?.trackDwellTime(cardId, dwellMs),
        trackInteraction: (cardId: string, type: string, data?: any) =>
            client?.trackInteraction(cardId, type, data),
        trackHesitation: (cardId: string, count: number) => client?.trackHesitation(cardId, count)
    }
}

// Add React import at top
import * as React from 'react'
