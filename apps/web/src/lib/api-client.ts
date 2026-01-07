/**
 * API Client for NerdLearn Microservices
 * Provides typed interfaces to all backend services
 */

import axios, { AxiosInstance } from 'axios'

const SCHEDULER_API = process.env.NEXT_PUBLIC_SCHEDULER_API_URL || 'http://localhost:8001'
const TELEMETRY_API = process.env.NEXT_PUBLIC_TELEMETRY_API_URL || 'http://localhost:8002'
const INFERENCE_API = process.env.NEXT_PUBLIC_INFERENCE_API_URL || 'http://localhost:8003'

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface ReviewRequest {
  card_id: string
  rating: 'again' | 'hard' | 'good' | 'easy'
  review_time?: string
  learner_id: string
}

export interface CardState {
  card_id: string
  stability: number
  difficulty: number
  scheduled_days: number
  due_date: string
  state: string
  review_count: number
}

export interface TelemetryEvent {
  user_id: string
  session_id: string
  event_type: string
  timestamp: number
  data: Record<string, any>
  resource_id?: string
  concept_id?: string
}

export interface EngagementScore {
  user_id: string
  session_id: string
  overall_score: number
  attention_score: number
  struggle_indicator: number
  confidence_score: number
}

export interface PredictionRequest {
  learner_id: string
  interaction_history: Array<{
    concept_id: number
    is_correct: boolean
  }>
  target_concept_id: number
}

export interface PredictionResponse {
  learner_id: string
  target_concept_id: number
  predicted_probability: number
  confidence: number
}

export interface ZPDAssessmentResponse {
  learner_id: string
  concept_id: string
  zone: 'frustration' | 'optimal' | 'comfort'
  success_rate: number
  trend: string
  recommended_actions: string[]
  active_scaffolding: string[]
  confidence: number
}

// ============================================================================
// API CLIENTS
// ============================================================================

class SchedulerClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: SCHEDULER_API,
      timeout: 10000,
    })
  }

  async processReview(request: ReviewRequest): Promise<CardState> {
    const response = await this.client.post<CardState>('/review', request)
    return response.data
  }

  async previewIntervals(learnerId: string, cardId: string) {
    const response = await this.client.get(`/preview/${learnerId}/${cardId}`)
    return response.data
  }

  async getDueCards(learnerId: string, limit: number = 20) {
    const response = await this.client.get(`/due/${learnerId}`, {
      params: { limit }
    })
    return response.data
  }
}

class TelemetryClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: TELEMETRY_API,
      timeout: 5000,
    })
  }

  async ingestEvent(event: TelemetryEvent): Promise<void> {
    await this.client.post('/event', event)
  }

  async ingestBatch(events: TelemetryEvent[]): Promise<void> {
    await this.client.post('/batch', events)
  }

  async getEngagementScore(userId: string, sessionId: string): Promise<EngagementScore> {
    const response = await this.client.get<EngagementScore>(
      `/analysis/engagement/${userId}/${sessionId}`
    )
    return response.data
  }

  async getMouseAnalysis(userId: string, sessionId: string) {
    const response = await this.client.get(
      `/analysis/mouse/${userId}/${sessionId}`
    )
    return response.data
  }

  createWebSocket(userId: string, sessionId: string): WebSocket {
    const wsUrl = TELEMETRY_API.replace('http', 'ws')
    return new WebSocket(`${wsUrl}/ws/${userId}/${sessionId}`)
  }
}

class InferenceClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: INFERENCE_API,
      timeout: 15000,
    })
  }

  async predictPerformance(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await this.client.post<PredictionResponse>('/predict', request)
    return response.data
  }

  async getKnowledgeState(learnerId: string, interactionHistory: any[]) {
    const response = await this.client.post('/knowledge-state', {
      learner_id: learnerId,
      interaction_history: interactionHistory
    })
    return response.data
  }

  async assessZPD(
    learnerId: string,
    conceptId: string,
    success: boolean
  ): Promise<ZPDAssessmentResponse> {
    const response = await this.client.post<ZPDAssessmentResponse>('/zpd/assess', {
      learner_id: learnerId,
      concept_id: conceptId,
      success
    })
    return response.data
  }

  async getRecommendations(
    learnerId: string,
    interactionHistory: any[],
    availableConcepts: string[],
    conceptDifficulties: Record<string, number>
  ) {
    const response = await this.client.post('/recommend', {
      learner_id: learnerId,
      interaction_history: interactionHistory,
      available_concepts: availableConcepts,
      concept_difficulties: conceptDifficulties
    })
    return response.data
  }
}

// Export singleton instances
export const schedulerClient = new SchedulerClient()
export const telemetryClient = new TelemetryClient()
export const inferenceClient = new InferenceClient()
