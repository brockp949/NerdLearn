"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, FileText, X } from "lucide-react";
import { api } from "@/lib/api/client";

interface ModuleUploaderProps {
  courseId: number;
  onUploadComplete: () => void;
}

export default function ModuleUploader({
  courseId,
  onUploadComplete,
}: ModuleUploaderProps) {
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [moduleData, setModuleData] = useState({
    title: "",
    description: "",
    module_type: "video",
    order: 1,
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);

      // Auto-detect module type
      const extension = file.name.split(".").pop()?.toLowerCase();
      if (extension === "pdf") {
        setModuleData((prev) => ({ ...prev, module_type: "pdf" }));
      } else if (["mp4", "mov", "avi", "webm"].includes(extension || "")) {
        setModuleData((prev) => ({ ...prev, module_type: "video" }));
      }

      // Auto-fill title from filename
      if (!moduleData.title) {
        const filename = file.name.replace(/\.[^/.]+$/, "");
        setModuleData((prev) => ({ ...prev, title: filename }));
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/*": [".mp4", ".mov", ".avi", ".webm"],
      "application/pdf": [".pdf"],
    },
    maxFiles: 1,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedFile) {
      alert("Please select a file");
      return;
    }

    if (!moduleData.title) {
      alert("Please enter a module title");
      return;
    }

    setUploading(true);

    try {
      await api.uploadModule(courseId, {
        ...moduleData,
        file: selectedFile,
      });

      // Reset form
      setSelectedFile(null);
      setModuleData({
        title: "",
        description: "",
        module_type: "video",
        order: moduleData.order + 1,
      });

      onUploadComplete();
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Upload Module</h3>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Dropzone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-primary bg-primary/5"
              : "border-gray-300 dark:border-gray-600 hover:border-primary"
          }`}
        >
          <input {...getInputProps()} />

          {selectedFile ? (
            <div className="flex items-center justify-center gap-4">
              {moduleData.module_type === "video" ? (
                <FileVideo className="h-10 w-10 text-primary" />
              ) : (
                <FileText className="h-10 w-10 text-primary" />
              )}
              <div className="text-left">
                <p className="font-medium">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedFile(null);
                }}
                className="ml-4 p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="h-10 w-10 mx-auto text-muted-foreground mb-4" />
              <p className="text-lg font-medium mb-2">
                {isDragActive ? "Drop file here" : "Drag and drop your file here"}
              </p>
              <p className="text-sm text-muted-foreground">
                or click to select (Videos or PDFs)
              </p>
            </>
          )}
        </div>

        {/* Module Details */}
        {selectedFile && (
          <>
            <div>
              <label className="block text-sm font-medium mb-2">Title</label>
              <input
                type="text"
                value={moduleData.title}
                onChange={(e) =>
                  setModuleData({ ...moduleData, title: e.target.value })
                }
                className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Description</label>
              <textarea
                value={moduleData.description}
                onChange={(e) =>
                  setModuleData({ ...moduleData, description: e.target.value })
                }
                className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                rows={3}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Type</label>
                <select
                  value={moduleData.module_type}
                  onChange={(e) =>
                    setModuleData({ ...moduleData, module_type: e.target.value })
                  }
                  className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                >
                  <option value="video">Video</option>
                  <option value="pdf">PDF</option>
                  <option value="quiz">Quiz</option>
                  <option value="interactive">Interactive</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Order</label>
                <input
                  type="number"
                  value={moduleData.order}
                  onChange={(e) =>
                    setModuleData({
                      ...moduleData,
                      order: parseInt(e.target.value),
                    })
                  }
                  className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600"
                  min="1"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={uploading}
              className="w-full px-4 py-2 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {uploading ? "Uploading..." : "Upload Module"}
            </button>
          </>
        )}
      </form>
    </div>
  );
}
