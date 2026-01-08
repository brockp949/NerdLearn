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
    getGraph: async () => {
        const response = await api.get('/graph/');
        return response.data;
    },
};

export default api;
