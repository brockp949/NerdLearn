'use client'

import { LearningCard } from './QuestionCard'
import ReactMarkdown from 'react-markdown'

interface ContentViewerProps {
  card: LearningCard
  onContinue: () => void
}

export function ContentViewer({ card, onContinue }: ContentViewerProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <span className="text-3xl">📖</span>
          <div>
            <h2 className="text-xl font-bold text-gray-900">{card.concept_name}</h2>
            <p className="text-sm text-gray-500">Learning Content</p>
          </div>
        </div>
        <div className="flex items-center space-x-2 px-3 py-1 bg-purple-100 rounded-full">
          <span className="text-xs font-medium text-purple-800">Difficulty {card.difficulty.toFixed(1)}</span>
        </div>
      </div>

      {/* Content */}
      <div className="prose prose-lg max-w-none mb-8">
        <div className="p-6 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border border-blue-100">
          <ReactMarkdown
            components={{
              // Custom styling for markdown elements
              p: ({ children }) => <p className="text-gray-800 leading-relaxed mb-4 last:mb-0">{children}</p>,
              strong: ({ children }) => <strong className="text-purple-700 font-bold">{children}</strong>,
              code: ({ children }) => (
                <code className="px-2 py-1 bg-purple-200 text-purple-900 rounded text-sm font-mono">
                  {children}
                </code>
              ),
              pre: ({ children }) => (
                <pre className="p-4 bg-gray-900 text-gray-100 rounded-lg overflow-x-auto my-4">
                  {children}
                </pre>
              ),
            }}
          >
            {card.content}
          </ReactMarkdown>
        </div>
      </div>

      {/* Tips Section */}
      <div className="mb-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
        <div className="flex items-start space-x-3">
          <span className="text-2xl">💡</span>
          <div>
            <p className="text-sm font-medium text-yellow-900 mb-1">Study Tip</p>
            <p className="text-sm text-yellow-800">
              Read the content carefully. You'll be asked a question about this concept next.
            </p>
          </div>
        </div>
      </div>

      {/* Continue Button */}
      <button
        onClick={onContinue}
        className="w-full py-4 px-6 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-lg hover:from-green-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-md"
      >
        I've Read This → Continue to Question
      </button>
    </div>
  )
}
