import Link from 'next/link'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-6xl font-bold text-center mb-8 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          NerdLearn
        </h1>

        <p className="text-xl text-center mb-12 text-gray-700 dark:text-gray-300">
          The Cognitive-Adaptive Learning Platform
        </p>

        <div className="grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-3 lg:text-left gap-4">
          <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30">
            <h2 className="mb-3 text-2xl font-semibold">
              🧠 Deep Knowledge Tracing
            </h2>
            <p className="m-0 max-w-[30ch] text-sm opacity-50">
              AI-powered understanding of your knowledge state using SAINT+ architecture
            </p>
          </div>

          <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30">
            <h2 className="mb-3 text-2xl font-semibold">
              🎯 ZPD Regulation
            </h2>
            <p className="m-0 max-w-[30ch] text-sm opacity-50">
              Maintains optimal challenge level - never too hard, never too easy
            </p>
          </div>

          <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30">
            <h2 className="mb-3 text-2xl font-semibold">
              📊 Stealth Assessment
            </h2>
            <p className="m-0 max-w-[30ch] text-sm opacity-50">
              Learn without tests - we assess through behavioral patterns
            </p>
          </div>

          <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30">
            <h2 className="mb-3 text-2xl font-semibold">
              🔄 FSRS Scheduling
            </h2>
            <p className="m-0 max-w-[30ch] text-sm opacity-50">
              99.6% more efficient than Anki - optimal spaced repetition
            </p>
          </div>

          <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30">
            <h2 className="mb-3 text-2xl font-semibold">
              🎮 Smart Gamification
            </h2>
            <p className="m-0 max-w-[30ch] text-sm opacity-50">
              Octalysis framework - intrinsic motivation, not manipulation
            </p>
          </div>

          <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30">
            <h2 className="mb-3 text-2xl font-semibold">
              🌳 Knowledge Graph
            </h2>
            <p className="m-0 max-w-[30ch] text-sm opacity-50">
              Visual skill trees showing your learning journey
            </p>
          </div>
        </div>

        <div className="mt-12 text-center">
          <Link
            href="/dashboard"
            className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold px-8 py-3 rounded-lg transition-colors"
          >
            Start Learning
          </Link>
        </div>

        <div className="mt-16 text-center text-sm text-gray-600 dark:text-gray-400">
          <p className="mb-2">Built on Research-Backed Principles:</p>
          <p className="text-xs">
            Evidence-Centered Design • Cognitive Load Theory • Zone of Proximal Development •
            Spaced Repetition • Interleaving • Transfer Learning
          </p>
        </div>
      </div>
    </main>
  )
}
