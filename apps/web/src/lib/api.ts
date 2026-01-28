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

// =============================================================================
// CURRICULUM API
// =============================================================================

export const curriculumApi = {
    generateCurriculum: async (data: any) => {
        const response = await api.post('/curriculum/generate', data);
        return response.data;
    },

    generateCurriculumAsync: async (data: any) => {
        const response = await api.post('/curriculum/generate/async', data);
        return response.data;
    },

    getJobStatus: async (jobId: string) => {
        const response = await api.get(`/curriculum/jobs/${jobId}`);
        return response.data;
    },

    getJobResult: async (jobId: string) => {
        const response = await api.get(`/curriculum/jobs/${jobId}/result`);
        return response.data;
    },

    previewArcOfLearning: async (data: any) => {
        const response = await api.post('/curriculum/preview', data);
        return response.data;
    },
};

// =============================================================================
// SOCIAL API
// =============================================================================

export const socialApi = {
    // Coding Challenges
    initSampleChallenges: async () => {
        const response = await api.post('/social/challenges/init');
        return response.data;
    },
    listChallenges: async () => {
        const response = await api.get('/social/coding-challenges');
        return response.data;
    },
    getChallenge: async (challengeId: string) => {
        const response = await api.get(`/social/coding-challenges/${challengeId}`);
        return response.data;
    },
    submitCode: async (data: any) => {
        const response = await api.post('/social/challenges/evaluate', data);
        return response.data;
    },
    getHint: async (data: any) => {
        const response = await api.post('/social/challenges/hint', data);
        return response.data;
    },

    // Debates
    startDebate: async (data: any) => {
        const response = await api.post('/social/debates/start', data);
        return response.data;
    },
    advanceDebate: async (data: any) => {
        const response = await api.post('/social/debates/advance', data);
        return response.data;
    },
    getDebateSummary: async (sessionId: string) => {
        const response = await api.get(`/social/debates/${sessionId}/summary`);
        return response.data;
    },

    // Teaching
    startTeachingSession: async (data: any) => {
        const response = await api.post('/social/teaching/start', data);
        return response.data;
    },
    submitExplanation: async (data: any) => {
        const response = await api.post('/social/teaching/explain', data);
        return response.data;
    },
    endTeachingSession: async (sessionId: string) => {
        const response = await api.post(`/social/teaching/${sessionId}/end`);
        return response.data;
    },
};

// =============================================================================
// MULTIMODAL API
// =============================================================================

export const multimodalApi = {
    // Conceptual State
    getConceptualState: async (userId: string, contentId: string) => {
        const response = await api.get(`/multimodal/state/${userId}/${contentId}`);
        return response.data;
    },
    resetConceptualState: async (userId: string, contentId: string) => {
        const response = await api.post('/multimodal/state/reset', { user_id: userId, content_id: contentId });
        return response.data;
    },
    getLearningSummary: async (userId: string, contentId: string) => {
        const response = await api.get(`/multimodal/summary/${userId}/${contentId}`);
        return response.data;
    },

    // Recommendations
    getModalityRecommendation: async (data: { user_id: string; content_id: string }) => {
        const response = await api.post('/multimodal/recommend', data);
        return response.data;
    },

    // Generation / Morphing
    morphContent: async (data: {
        content: string;
        source_modality: string;
        target_modality: string;
        user_id?: string;
        content_id?: string;
        options?: any;
    }) => {
        const response = await api.post('/multimodal/morph', data);
        return response.data;
    },

    generatePodcast: async (data: {
        content: string;
        topic: string;
        duration_minutes?: number;
        style?: string;
    }) => {
        const response = await api.post('/multimodal/generate/podcast', data);
        return response.data;
    },

    generateDiagram: async (data: {
        content: string;
        diagram_type?: string;
        title?: string;
        focus_concepts?: string[];
    }) => {
        const response = await api.post('/multimodal/generate/diagram', data);
        return response.data;
    },

    generateDiagramFromConcepts: async (data: {
        concepts: string[];
        relation_type?: string;
        title?: string;
    }) => {
        const response = await api.post('/multimodal/generate/diagram-from-concepts', data);
        return response.data;
    },

    // Metadata
    getSupportedModalities: async () => {
        const response = await api.get('/multimodal/modalities');
        return response.data;
    },

    getDiagramTypes: async () => {
        const response = await api.get('/multimodal/diagram-types');
        return response.data;
    },
};

// =============================================================================
// GAMIFICATION API
// =============================================================================

// =============================================================================
// TESTING API
// =============================================================================

export const testingApi = {
    getSummary: async () => {
        const response = await api.get('/testing/summary');
        return response.data;
    },

    getSuite: async (suiteId: string) => {
        const response = await api.get(`/testing/suites/${suiteId}`);
        return response.data;
    },

    getTests: async (category?: string, status?: string) => {
        const response = await api.get('/testing/tests', {
            params: { category, status },
        });
        return response.data;
    },

    getFailedTests: async () => {
        const response = await api.get('/testing/failed');
        return response.data;
    },

    runTests: async (suiteId?: string, testPath?: string) => {
        const response = await api.post('/testing/run', {
            suite_id: suiteId,
            test_path: testPath,
        });
        return response.data;
    },

    getAntigravityResults: async () => {
        const response = await api.get('/testing/antigravity');
        return response.data;
    },

    clearCache: async () => {
        const response = await api.delete('/testing/cache');
        return response.data;
    },
};

// =============================================================================
// GAMIFICATION API
// =============================================================================

export const gamificationApi = {
    getProfile: async (userId: number) => {
        const response = await api.get(`/gamification/profile/${userId}`);
        return response.data;
    },

    getAchievements: async (userId: number) => {
        const response = await api.get('/gamification/achievements', {
            params: { user_id: userId }
        });
        return response.data;
    },

    getSkillTree: async (userId: number, courseId: number) => {
        const response = await api.get('/gamification/skill-tree', {
            params: { user_id: userId, course_id: courseId }
        });
        return response.data;
    },

    getLeaderboard: async (courseId?: number) => {
        const response = await api.get('/gamification/leaderboard', {
            params: { course_id: courseId }
        });
        return response.data;
    },

    triggerReward: async (data: {
        user_id: number;
        mastery_level: number;
        age_group: string;
    }) => {
        const response = await api.post('/gamification/trigger-reward', data);
        return response.data;
    },

    awardXP: async (data: {
        user_id: number;
        action: string;
        multiplier?: number;
        is_first_time?: boolean;
    }) => {
        const response = await api.post('/gamification/xp/award', data);
        return response.data;
    },
};

export default api;
