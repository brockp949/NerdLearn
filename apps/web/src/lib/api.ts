import axios from 'axios';

// Since we are using Next.js proxy, baseURL is relative to the server
// The backend routers are mounted at /api directly (e.g. /api/chat)
const api = axios.create({
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

export const ragApi = {
    chat: async (query: string) => {
        // Real backend expects user_id and course_id. Mocking for now.
        const response = await api.post('/chat/', {
            query,
            user_id: 1, // Mock User
            course_id: 1 // Mock Course
        });
        return response.data;
    },
};

export const graphApi = {
    getGraph: async (courseId: number = 1) => {
        const response = await api.get('/graph/', {
            params: { course_id: courseId }
        });
        return response.data;
    },
    getCourseGraph: async (courseId: number) => {
        const response = await api.get(`/graph/courses/${courseId}`);
        return response.data;
    },
    getLearningPath: async (courseId: number, targetConcepts: string[], masteredConcepts?: string[]) => {
        const response = await api.post(`/graph/courses/${courseId}/learning-path`, {
            target_concepts: targetConcepts,
            mastered_concepts: masteredConcepts || [],
        });
        return response.data;
    },
};

export const adaptiveApi = {
    estimateCognitiveLoad: async (metrics: any[], currentDifficulty: number = 5.0) => {
        const response = await api.post('/adaptive/cognitive-load/estimate', {
            recent_metrics: metrics,
            content_difficulty: currentDifficulty,
        });
        return response.data;
    },
};

// =============================================================================
// COGNITIVE API
// =============================================================================

export const cognitiveApi = {
    // Frustration Detection
    detectFrustration: async (data: {
        user_id: string;
        events: any[];
        context?: Record<string, unknown>;
    }) => {
        const response = await api.post('/cognitive/frustration/detect', data);
        return response.data;
    },

    updateBaseline: async (userId: string, events: any[]) => {
        const response = await api.post('/cognitive/frustration/update-baseline', {
            user_id: userId,
            events,
        });
        return response.data;
    },

    // Metacognition
    getMetacognitionPrompt: async (data: {
        user_id: string;
        concept_name: string;
        timing: string;
        context?: Record<string, unknown>;
        force?: boolean;
    }) => {
        const response = await api.post('/cognitive/metacognition/prompt', data);
        return response.data;
    },

    getConfidenceScale: async (conceptName: string, scaleType: string = 'numeric') => {
        const response = await api.get('/cognitive/metacognition/confidence-scale', {
            params: { concept_name: conceptName, scale_type: scaleType },
        });
        return response.data;
    },

    recordConfidence: async (data: {
        user_id: string;
        concept_id: string;
        content_id: string;
        confidence: number;
        context?: string;
    }) => {
        const response = await api.post('/cognitive/metacognition/record-confidence', data);
        return response.data;
    },

    analyzeExplanation: async (data: {
        explanation_text: string;
        concept_name: string;
        expected_concepts?: string[];
        common_misconceptions?: string[];
    }) => {
        const response = await api.post('/cognitive/metacognition/analyze-explanation', data);
        return response.data;
    },

    // Calibration
    calculateCalibration: async (data: {
        user_id: string;
        concept_id?: string;
        time_window_hours?: number;
    }) => {
        const response = await api.post('/cognitive/calibration/calculate', data);
        return response.data;
    },

    getCalibrationFeedback: async (data: {
        user_id: string;
        concept_id?: string;
        time_window_hours?: number;
    }) => {
        const response = await api.post('/cognitive/calibration/feedback', data);
        return response.data;
    },

    updatePerformance: async (userId: string, conceptId: string, actualPerformance: number) => {
        const response = await api.post('/cognitive/calibration/update-performance', {
            user_id: userId,
            concept_id: conceptId,
            actual_performance: actualPerformance,
        });
        return response.data;
    },

    // Interventions
    decideIntervention: async (data: {
        learner_state: any;
        events?: any[];
        context?: Record<string, unknown>;
    }) => {
        const response = await api.post('/cognitive/intervention/decide', data);
        return response.data;
    },

    getInterventionHistory: async (userId: string) => {
        const response = await api.get(`/cognitive/intervention/history/${userId}`);
        return response.data;
    },

    // Profile
    getCognitiveProfile: async (userId: string) => {
        const response = await api.get(`/cognitive/profile/${userId}`);
        return response.data;
    },
};

export default api;
