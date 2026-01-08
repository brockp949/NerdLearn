import Link from "next/link";
import { BookOpen, Brain, Sparkles, TrendingUp } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-800">
      <nav className="border-b bg-white/50 backdrop-blur-sm dark:bg-gray-900/50">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            <h1 className="text-2xl font-bold">NerdLearn</h1>
          </div>
          <div className="flex gap-4">
            <Link
              href="/studio"
              className="px-4 py-2 text-sm font-medium hover:text-primary transition-colors"
            >
              Instructor Studio
            </Link>
            <Link
              href="/learn"
              className="px-4 py-2 text-sm font-medium bg-primary text-white rounded-md hover:bg-primary/90 transition-colors"
            >
              Start Learning
            </Link>
          </div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Learn Smarter, Not Harder
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-powered adaptive learning platform that combines the best of Udemy and NotebookLM
            with cognitive science
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          <FeatureCard
            icon={<Brain className="h-8 w-8" />}
            title="AI Chat-to-Learn"
            description="Chat with your course content like NotebookLM. Get instant answers with citations."
          />
          <FeatureCard
            icon={<Sparkles className="h-8 w-8" />}
            title="Stealth Assessment"
            description="Invisible learning analytics track your progress without interrupting flow."
          />
          <FeatureCard
            icon={<TrendingUp className="h-8 w-8" />}
            title="Adaptive Engine"
            description="Spaced repetition (FSRS) and Zone of Proximal Development optimization."
          />
          <FeatureCard
            icon={<BookOpen className="h-8 w-8" />}
            title="Knowledge Graphs"
            description="Visualize concept relationships. AI builds prerequisite chains automatically."
          />
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
          <h3 className="text-2xl font-bold mb-4">Ready to Transform Your Learning?</h3>
          <p className="text-muted-foreground mb-6">
            Join instructors and learners building the future of education
          </p>
          <div className="flex gap-4 justify-center">
            <Link
              href="/studio"
              className="px-6 py-3 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors"
            >
              Create Course
            </Link>
            <Link
              href="/learn"
              className="px-6 py-3 bg-primary text-white rounded-md hover:bg-primary/90 transition-colors"
            >
              Browse Courses
            </Link>
          </div>
        </div>
      </main>

      <footer className="border-t mt-16 py-8">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>© 2026 NerdLearn - AI-Powered Adaptive Learning Platform</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
      <div className="text-primary mb-4">{icon}</div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}
