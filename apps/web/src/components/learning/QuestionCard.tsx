'use client'

import { useState } from 'react'

export interface LearningCard {
  card_id: string
  concept_name: string
  content: string
  question: string
  correct_answer?: string
  difficulty: number
  due_date?: string
}

export type Rating = 'again' | 'hard' | 'good' | 'easy'

interface QuestionCardProps {
  card: LearningCard
  onAnswer: (rating: Rating) => void
  loading?: boolean
}

export function QuestionCard({ card, onAnswer, loading }: QuestionCardProps) {
  const [showAnswer, setShowAnswer] = useState(false)
  const [selectedRating, setSelectedRating] = useState<Rating | null>(null)

  const handleAnswer = (rating: Rating) => {
    setSelectedRating(rating)
    setTimeout(() => {
      onAnswer(rating)
      setShowAnswer(false)
      setSelectedRating(null)
    }, 300)
  }

  const ratingButtons = [
    { rating: 'again' as Rating, label: 'Again', color: 'bg-red-500 hover:bg-red-600', emoji: '❌' },
    { rating: 'hard' as Rating, label: 'Hard', color: 'bg-orange-500 hover:bg-orange-600', emoji: '😓' },
    { rating: 'good' as Rating, label: 'Good', color: 'bg-green-500 hover:bg-green-600', emoji: '✅' },
    { rating: 'easy' as Rating, label: 'Easy', color: 'bg-blue-500 hover:bg-blue-600', emoji: '🎯' },
  ]

  return (
    <div className="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
      {/* Card Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          <span className="text-sm font-medium text-gray-600">{card.concept_name}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-500">Difficulty:</span>
          <div className="flex space-x-1">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((level) => (
              <div
                key={level}
                className={`w-1.5 h-4 rounded ${
                  level <= card.difficulty ? 'bg-purple-500' : 'bg-gray-200'
                }`}
              />
            ))}
          </div>
          <span className="text-xs font-medium text-gray-700">{card.difficulty.toFixed(1)}</span>
        </div>
      </div>

      {/* Question */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">{card.question}</h2>
      </div>

      {/* Show Answer Button */}
      {!showAnswer && (
        <button
          onClick={() => setShowAnswer(true)}
          disabled={loading}
          className="w-full py-3 px-6 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Show Answer
        </button>
      )}

      {/* Answer & Rating Buttons */}
      {showAnswer && (
        <div className="space-y-6">
          {/* Answer Display */}
          {card.correct_answer && (
            <div className="p-6 bg-blue-50 border-l-4 border-blue-500 rounded">
              <p className="text-sm font-medium text-blue-900 mb-2">Answer:</p>
              <p className="text-lg text-gray-900">{card.correct_answer}</p>
            </div>
          )}

          {/* Rating Prompt */}
          <div className="text-center">
            <p className="text-gray-700 font-medium mb-4">How well did you know this?</p>
          </div>

          {/* Rating Buttons */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {ratingButtons.map(({ rating, label, color, emoji }) => (
              <button
                key={rating}
                onClick={() => handleAnswer(rating)}
                disabled={loading}
                className={`py-4 px-4 ${color} text-white font-semibold rounded-lg transition transform ${
                  selectedRating === rating ? 'scale-95' : 'hover:scale-105'
                } disabled:opacity-50 disabled:cursor-not-allowed shadow-md`}
              >
                <div className="flex flex-col items-center space-y-1">
                  <span className="text-2xl">{emoji}</span>
                  <span className="text-sm">{label}</span>
                </div>
              </button>
            ))}
          </div>

          {/* Rating Helper Text */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-gray-500 text-center">
            <div>Completely forgot</div>
            <div>Struggled to recall</div>
            <div>Remembered well</div>
            <div>Too easy</div>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded-lg">
          <div className="flex items-center space-x-3">
            <svg className="animate-spin h-8 w-8 text-purple-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span className="text-gray-700 font-medium">Processing...</span>
          </div>
        </div>
      )}
    </div>
  )
}
