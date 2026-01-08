const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Course {
  id: number;
  title: string;
  description: string;
  instructor_id: number;
  status: "draft" | "published" | "archived";
  price: number;
  difficulty_level: string;
  created_at: string;
  updated_at?: string;
}

export interface Module {
  id: number;
  course_id: number;
  title: string;
  description?: string;
  module_type: "video" | "pdf" | "quiz" | "interactive";
  order: number;
  file_url?: string;
  is_processed: boolean;
  created_at: string;
}

export const api = {
  // Courses
  async getCourses(instructorId?: number): Promise<Course[]> {
    const params = instructorId ? `?instructor_id=${instructorId}` : "";
    const res = await fetch(`${API_BASE_URL}/api/courses${params}`);
    if (!res.ok) throw new Error("Failed to fetch courses");
    return res.json();
  },

  async getCourse(id: number): Promise<Course> {
    const res = await fetch(`${API_BASE_URL}/api/courses/${id}`);
    if (!res.ok) throw new Error("Failed to fetch course");
    return res.json();
  },

  async createCourse(data: {
    title: string;
    description: string;
    instructor_id: number;
    price: number;
    difficulty_level: string;
    tags: string[];
  }): Promise<Course> {
    const res = await fetch(`${API_BASE_URL}/api/courses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create course");
    return res.json();
  },

  async publishCourse(id: number): Promise<Course> {
    const res = await fetch(`${API_BASE_URL}/api/courses/${id}/publish`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Failed to publish course");
    return res.json();
  },

  // Modules
  async getModules(courseId: number): Promise<Module[]> {
    const res = await fetch(`${API_BASE_URL}/api/courses/${courseId}/modules`);
    if (!res.ok) throw new Error("Failed to fetch modules");
    return res.json();
  },

  async uploadModule(
    courseId: number,
    data: {
      title: string;
      description: string;
      module_type: string;
      order: number;
      file: File;
    }
  ): Promise<Module> {
    const formData = new FormData();
    formData.append("title", data.title);
    formData.append("description", data.description || "");
    formData.append("module_type", data.module_type);
    formData.append("order", data.order.toString());
    formData.append("file", data.file);

    const res = await fetch(`${API_BASE_URL}/api/courses/${courseId}/modules`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || "Failed to upload module");
    }

    return res.json();
  },
};
