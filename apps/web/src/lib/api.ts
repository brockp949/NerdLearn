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

export default api;
