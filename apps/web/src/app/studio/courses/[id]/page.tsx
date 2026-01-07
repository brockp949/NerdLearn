"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Settings,
  Upload,
  Eye,
  FileVideo,
  FileText,
  CheckCircle,
  Clock,
} from "lucide-react";
import { api, Course, Module } from "@/lib/api/client";
import ModuleUploader from "@/components/studio/ModuleUploader";

export default function CourseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const courseId = parseInt(params.id as string);

  const [course, setCourse] = useState<Course | null>(null);
  const [modules, setModules] = useState<Module[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUploader, setShowUploader] = useState(false);

  useEffect(() => {
    loadCourse();
    loadModules();
  }, [courseId]);

  const loadCourse = async () => {
    try {
      const data = await api.getCourse(courseId);
      setCourse(data);
    } catch (error) {
      console.error("Failed to load course:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadModules = async () => {
    try {
      const data = await api.getModules(courseId);
      setModules(data);
    } catch (error) {
      console.error("Failed to load modules:", error);
    }
  };

  const handlePublish = async () => {
    if (!confirm("Are you sure you want to publish this course?")) return;

    try {
      const updated = await api.publishCourse(courseId);
      setCourse(updated);
    } catch (error) {
      console.error("Failed to publish course:", error);
      alert("Failed to publish course");
    }
  };

  if (loading || !course) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/studio"
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold">{course.title}</h1>
                <p className="text-sm text-muted-foreground">
                  {modules.length} modules
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Link
                href={`/learn/${courseId}`}
                className="inline-flex items-center gap-2 px-4 py-2 border rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                <Eye className="h-4 w-4" />
                Preview
              </Link>

              {course.status === "draft" && (
                <button
                  onClick={handlePublish}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors"
                >
                  <CheckCircle className="h-4 w-4" />
                  Publish
                </button>
              )}

              <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                <Settings className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Course Info */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Course Information</h2>
              <div className="space-y-2">
                <p className="text-sm">
                  <span className="font-medium">Status:</span>{" "}
                  <span
                    className={`capitalize ${
                      course.status === "published"
                        ? "text-green-600"
                        : "text-gray-600"
                    }`}
                  >
                    {course.status}
                  </span>
                </p>
                <p className="text-sm">
                  <span className="font-medium">Price:</span> ${course.price.toFixed(2)}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Difficulty:</span>{" "}
                  {course.difficulty_level || "Not set"}
                </p>
              </div>
            </div>

            {/* Modules */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
              <div className="px-6 py-4 border-b dark:border-gray-700 flex items-center justify-between">
                <h2 className="text-lg font-semibold">Course Modules</h2>
                <button
                  onClick={() => setShowUploader(!showUploader)}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors text-sm"
                >
                  <Upload className="h-4 w-4" />
                  {showUploader ? "Hide Uploader" : "Add Module"}
                </button>
              </div>

              {modules.length === 0 && !showUploader ? (
                <div className="px-6 py-12 text-center">
                  <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">No modules yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Upload videos or PDFs to create your course content
                  </p>
                  <button
                    onClick={() => setShowUploader(true)}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors"
                  >
                    <Upload className="h-4 w-4" />
                    Upload First Module
                  </button>
                </div>
              ) : (
                <div className="divide-y dark:divide-gray-700">
                  {modules.map((module) => (
                    <ModuleListItem key={module.id} module={module} />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {showUploader && (
              <ModuleUploader
                courseId={courseId}
                onUploadComplete={() => {
                  loadModules();
                  setShowUploader(false);
                }}
              />
            )}

            {/* Stats */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-4">Statistics</h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Total Modules</span>
                  <span className="font-medium">{modules.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Videos</span>
                  <span className="font-medium">
                    {modules.filter((m) => m.module_type === "video").length}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">PDFs</span>
                  <span className="font-medium">
                    {modules.filter((m) => m.module_type === "pdf").length}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Processed</span>
                  <span className="font-medium">
                    {modules.filter((m) => m.is_processed).length} /{" "}
                    {modules.length}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ModuleListItem({ module }: { module: Module }) {
  const typeIcons = {
    video: <FileVideo className="h-5 w-5 text-blue-500" />,
    pdf: <FileText className="h-5 w-5 text-red-500" />,
    quiz: <CheckCircle className="h-5 w-5 text-green-500" />,
    interactive: <Settings className="h-5 w-5 text-purple-500" />,
  };

  return (
    <div className="px-6 py-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
      <div className="flex items-center gap-4">
        {typeIcons[module.module_type]}
        <div>
          <h4 className="font-medium">{module.title}</h4>
          {module.description && (
            <p className="text-sm text-muted-foreground line-clamp-1">
              {module.description}
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-4">
        <span className="text-xs text-muted-foreground">#{module.order}</span>
        {module.is_processed ? (
          <CheckCircle className="h-5 w-5 text-green-500" />
        ) : (
          <Clock className="h-5 w-5 text-yellow-500" />
        )}
      </div>
    </div>
  );
}
