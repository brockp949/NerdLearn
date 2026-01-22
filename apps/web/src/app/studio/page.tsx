"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Plus, BookOpen, Users, TrendingUp } from "lucide-react";
import { api, Course } from "@/lib/api/client";

export default function StudioPage() {
  const [courses, setCourses] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCourses();
  }, []);

  const loadCourses = async () => {
    try {
      // TODO: Replace with actual instructor ID from auth
      const data = await api.getCourses(1);
      setCourses(data);
    } catch (error) {
      console.error("Failed to load courses:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
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
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold">Instructor Studio</h1>
              <p className="text-sm text-muted-foreground">
                Create and manage your courses
              </p>
            </div>
            <Link
              href="/studio/courses/new"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors"
            >
              <Plus className="h-4 w-4" />
              Create Course
            </Link>
          </div>
        </div>
      </header>

      {/* Stats */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <StatCard
            icon={<BookOpen className="h-6 w-6" />}
            title="Total Courses"
            value={courses.length.toString()}
            color="blue"
          />
          <StatCard
            icon={<Users className="h-6 w-6" />}
            title="Total Students"
            value="0"
            color="green"
          />
          <StatCard
            icon={<TrendingUp className="h-6 w-6" />}
            title="Published"
            value={courses.filter((c) => c.status === "published").length.toString()}
            color="purple"
          />
        </div>

        {/* Courses List */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="px-6 py-4 border-b dark:border-gray-700">
            <h2 className="text-lg font-semibold">My Courses</h2>
          </div>

          {courses.length === 0 ? (
            <div className="px-6 py-12 text-center">
              <BookOpen className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No courses yet</h3>
              <p className="text-muted-foreground mb-4">
                Create your first course to get started
              </p>
              <Link
                href="/studio/courses/new"
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors"
              >
                <Plus className="h-4 w-4" />
                Create Course
              </Link>
            </div>
          ) : (
            <div className="divide-y dark:divide-gray-700">
              {courses.map((course) => (
                <CourseListItem key={course.id} course={course} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon,
  title,
  value,
  color,
}: {
  icon: React.ReactNode;
  title: string;
  value: string;
  color: "blue" | "green" | "purple";
}) {
  const colorClasses = {
    blue: "bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-400",
    green: "bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-400",
    purple: "bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-400",
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className={`inline-flex p-3 rounded-lg ${colorClasses[color]} mb-4`}>
        {icon}
      </div>
      <p className="text-sm text-muted-foreground mb-1">{title}</p>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  );
}

function CourseListItem({ course }: { course: Course }) {
  const statusColors = {
    draft: "bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300",
    published: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
    archived: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  };

  return (
    <Link
      href={`/studio/courses/${course.id}`}
      className="px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center justify-between"
    >
      <div>
        <h3 className="font-medium mb-1">{course.title}</h3>
        <p className="text-sm text-muted-foreground line-clamp-1">
          {course.description}
        </p>
      </div>
      <div className="flex items-center gap-4">
        <span
          className={`px-3 py-1 rounded-full text-xs font-medium ${
            statusColors[course.status]
          }`}
        >
          {course.status.charAt(0).toUpperCase() + course.status.slice(1)}
        </span>
        <span className="text-sm text-muted-foreground">
          ${course.price.toFixed(2)}
        </span>
      </div>
    </Link>
  );
}
