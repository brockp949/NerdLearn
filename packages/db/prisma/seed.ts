import { PrismaClient, BloomLevel, CardType } from '@prisma/client'
import { Neo4jClient } from '../src/neo4j'

const prisma = new PrismaClient()
const neo4j = new Neo4jClient()

// Helper to hash passwords (simple for demo - in production use bcrypt)
function simpleHash(password: string): string {
  // This is just a placeholder - the actual hashing should be done by the API Gateway
  // For now, we'll store a marker that indicates this needs to be hashed
  return `PLACEHOLDER_${password}`
}

async function main() {
  console.log('🌱 Starting seed...')

  // Clean existing data (optional - comment out to preserve data)
  console.log('🧹 Cleaning existing data...')
  await prisma.evidence.deleteMany()
  await prisma.competencyState.deleteMany()
  await prisma.scheduledItem.deleteMany()
  await prisma.card.deleteMany()
  await prisma.concept.deleteMany()
  await prisma.learnerProfile.deleteMany()
  await prisma.user.deleteMany()

  console.log('✅ Cleaned existing data')

  // 1. Create Demo Users
  console.log('👤 Creating demo users...')

  const demoUser = await prisma.user.create({
    data: {
      email: 'demo@nerdlearn.com',
      username: 'demo',
      passwordHash: simpleHash('demo123'),
      fullName: 'Demo User',
      learnerProfile: {
        create: {
          cognitiveEmbedding: JSON.stringify([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), // 6D vector for Bloom's levels
          fsrsStability: 2.5,
          fsrsDifficulty: 5.0,
          currentZpdLower: 0.35,
          currentZpdUpper: 0.70,
          totalXP: 0,
          level: 1,
          streakDays: 0,
          lastReviewDate: new Date(),
        }
      }
    },
    include: { learnerProfile: true }
  })

  const aliceUser = await prisma.user.create({
    data: {
      email: 'alice@example.com',
      username: 'alice',
      passwordHash: simpleHash('alice123'),
      fullName: 'Alice Johnson',
      learnerProfile: {
        create: {
          cognitiveEmbedding: JSON.stringify([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
          fsrsStability: 3.0,
          fsrsDifficulty: 4.5,
          currentZpdLower: 0.35,
          currentZpdUpper: 0.70,
          totalXP: 250,
          level: 2,
          streakDays: 3,
          lastReviewDate: new Date(),
        }
      }
    },
    include: { learnerProfile: true }
  })

  const bobUser = await prisma.user.create({
    data: {
      email: 'bob@example.com',
      username: 'bob',
      passwordHash: simpleHash('bob123'),
      fullName: 'Bob Smith',
      learnerProfile: {
        create: {
          cognitiveEmbedding: JSON.stringify([0.4, 0.4, 0.5, 0.5, 0.3, 0.2]),
          fsrsStability: 2.0,
          fsrsDifficulty: 6.0,
          currentZpdLower: 0.35,
          currentZpdUpper: 0.70,
          totalXP: 500,
          level: 3,
          streakDays: 7,
          lastReviewDate: new Date(),
        }
      }
    },
    include: { learnerProfile: true }
  })

  console.log(`✅ Created 3 users: demo, alice, bob`)

  // 2. Create Python Concepts
  console.log('💡 Creating Python concepts...')

  const concepts = await Promise.all([
    prisma.concept.create({
      data: {
        name: 'Python Variables',
        description: 'Understanding variables and data storage in Python',
        domain: 'Python',
        bloomLevel: BloomLevel.REMEMBER,
        estimatedDifficulty: 3.0,
        conceptEmbedding: JSON.stringify([0.8, 0.2, 0.1, 0.0, 0.0, 0.0]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Functions',
        description: 'Creating and using functions in Python',
        domain: 'Python',
        bloomLevel: BloomLevel.UNDERSTAND,
        estimatedDifficulty: 4.5,
        conceptEmbedding: JSON.stringify([0.6, 0.7, 0.3, 0.2, 0.0, 0.0]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Loops',
        description: 'For loops, while loops, and iteration',
        domain: 'Python',
        bloomLevel: BloomLevel.APPLY,
        estimatedDifficulty: 5.0,
        conceptEmbedding: JSON.stringify([0.5, 0.6, 0.7, 0.3, 0.1, 0.0]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Lists',
        description: 'Working with Python lists and list methods',
        domain: 'Python',
        bloomLevel: BloomLevel.APPLY,
        estimatedDifficulty: 4.0,
        conceptEmbedding: JSON.stringify([0.6, 0.7, 0.6, 0.2, 0.1, 0.0]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Dictionaries',
        description: 'Key-value pairs and dictionary operations',
        domain: 'Python',
        bloomLevel: BloomLevel.APPLY,
        estimatedDifficulty: 4.5,
        conceptEmbedding: JSON.stringify([0.5, 0.6, 0.7, 0.3, 0.1, 0.0]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Control Flow',
        description: 'If statements, elif, and else',
        domain: 'Python',
        bloomLevel: BloomLevel.UNDERSTAND,
        estimatedDifficulty: 3.5,
        conceptEmbedding: JSON.stringify([0.7, 0.6, 0.4, 0.1, 0.0, 0.0]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Recursion',
        description: 'Recursive functions and base cases',
        domain: 'Python',
        bloomLevel: BloomLevel.ANALYZE,
        estimatedDifficulty: 7.5,
        conceptEmbedding: JSON.stringify([0.3, 0.4, 0.5, 0.7, 0.4, 0.2]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Error Handling',
        description: 'Try-except blocks and exception handling',
        domain: 'Python',
        bloomLevel: BloomLevel.APPLY,
        estimatedDifficulty: 5.5,
        conceptEmbedding: JSON.stringify([0.4, 0.5, 0.7, 0.4, 0.2, 0.1]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'File I/O',
        description: 'Reading and writing files in Python',
        domain: 'Python',
        bloomLevel: BloomLevel.APPLY,
        estimatedDifficulty: 5.0,
        conceptEmbedding: JSON.stringify([0.5, 0.6, 0.6, 0.3, 0.1, 0.0]),
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Classes',
        description: 'Object-oriented programming with classes',
        domain: 'Python',
        bloomLevel: BloomLevel.CREATE,
        estimatedDifficulty: 6.5,
        conceptEmbedding: JSON.stringify([0.3, 0.4, 0.5, 0.6, 0.5, 0.7]),
      }
    }),
  ])

  console.log(`✅ Created ${concepts.length} concepts`)

  // 3. Create Learning Cards (3 per concept = 30 total)
  console.log('📝 Creating learning cards...')

  const cardData = [
    // Variables (3 cards)
    {
      conceptId: concepts[0].id,
      content: 'A **variable** in Python is a named storage location that holds a value. Variables are created using the assignment operator `=`.\n\n```python\nname = "Alice"\nage = 25\nis_student = True\n```',
      question: 'What operator is used to assign a value to a variable in Python?',
      correctAnswer: '= (equals sign)',
      difficulty: 2.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[0].id,
      content: 'Python variables are **dynamically typed**, meaning you don\'t need to declare the type. Python infers it from the value.\n\n```python\nx = 5        # int\nx = "hello"  # now a string (valid!)\n```',
      question: 'What does "dynamically typed" mean in Python?',
      correctAnswer: 'Variables can change type, and you don\'t need to declare types explicitly',
      difficulty: 3.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[0].id,
      content: 'Variable names in Python must start with a letter or underscore, and can contain letters, numbers, and underscores. They are **case-sensitive**.\n\n```python\nmy_var = 1\nMyVar = 2   # different variable!\n```',
      question: 'Can a variable name start with a number in Python?',
      correctAnswer: 'No, variable names must start with a letter or underscore',
      difficulty: 3.5,
      cardType: CardType.BASIC,
    },

    // Functions (3 cards)
    {
      conceptId: concepts[1].id,
      content: 'A **function** is a reusable block of code that performs a specific task. Functions are defined using the `def` keyword.\n\n```python\ndef greet(name):\n    return f"Hello, {name}!"\n\nresult = greet("Alice")\n```',
      question: 'What keyword is used to define a function in Python?',
      correctAnswer: 'def',
      difficulty: 4.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[1].id,
      content: 'Functions can accept **parameters** (inputs) and optionally return a value using the `return` statement.\n\n```python\ndef add(a, b):\n    return a + b\n\nsum = add(3, 5)  # returns 8\n```',
      question: 'How do you pass multiple parameters to a function?',
      correctAnswer: 'Separate them with commas in the function definition and call',
      difficulty: 4.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[1].id,
      content: 'If a function doesn\'t have a `return` statement, it returns `None` by default.\n\n```python\ndef print_message(msg):\n    print(msg)\n    # No return statement\n\nresult = print_message("Hi")  # result is None\n```',
      question: 'What does a Python function return if it has no return statement?',
      correctAnswer: 'None',
      difficulty: 5.0,
      cardType: CardType.BASIC,
    },

    // Loops (3 cards)
    {
      conceptId: concepts[2].id,
      content: 'A **for loop** iterates over a sequence (like a list, string, or range).\n\n```python\nfor i in range(5):\n    print(i)  # prints 0, 1, 2, 3, 4\n```',
      question: 'What does range(5) produce in a for loop?',
      correctAnswer: 'Numbers from 0 to 4 (5 is excluded)',
      difficulty: 4.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[2].id,
      content: 'A **while loop** continues executing as long as a condition is True.\n\n```python\ncount = 0\nwhile count < 3:\n    print(count)\n    count += 1\n```',
      question: 'What happens if a while loop\'s condition never becomes False?',
      correctAnswer: 'Infinite loop (the code runs forever)',
      difficulty: 5.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[2].id,
      content: 'You can use `break` to exit a loop early and `continue` to skip to the next iteration.\n\n```python\nfor i in range(10):\n    if i == 5:\n        break  # stops at 5\n    print(i)\n```',
      question: 'What does the break statement do in a loop?',
      correctAnswer: 'Immediately exits the loop',
      difficulty: 5.5,
      cardType: CardType.BASIC,
    },

    // Lists (3 cards)
    {
      conceptId: concepts[3].id,
      content: 'A **list** is an ordered, mutable collection of items enclosed in square brackets.\n\n```python\nfruits = ["apple", "banana", "cherry"]\nprint(fruits[0])  # "apple"\n```',
      question: 'What index is used to access the first element of a list?',
      correctAnswer: '0 (zero)',
      difficulty: 3.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[3].id,
      content: 'Lists support many operations: append, remove, insert, and more.\n\n```python\nnumbers = [1, 2, 3]\nnumbers.append(4)  # [1, 2, 3, 4]\nnumbers.remove(2)  # [1, 3, 4]\n```',
      question: 'What method adds an item to the end of a list?',
      correctAnswer: 'append()',
      difficulty: 4.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[3].id,
      content: 'You can use **negative indices** to access elements from the end of a list.\n\n```python\nitems = ["a", "b", "c"]\nprint(items[-1])  # "c" (last item)\n```',
      question: 'What does items[-2] return for a list with 5 elements?',
      correctAnswer: 'The second-to-last element',
      difficulty: 4.5,
      cardType: CardType.BASIC,
    },

    // Dictionaries (3 cards)
    {
      conceptId: concepts[4].id,
      content: 'A **dictionary** stores key-value pairs in curly braces.\n\n```python\nperson = {"name": "Alice", "age": 25}\nprint(person["name"])  # "Alice"\n```',
      question: 'What are the two components of each item in a dictionary?',
      correctAnswer: 'Key and value',
      difficulty: 4.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[4].id,
      content: 'Dictionary keys must be **immutable** (strings, numbers, tuples). Values can be any type.\n\n```python\ndata = {"count": 5, "items": [1, 2, 3]}\n```',
      question: 'Can you use a list as a dictionary key?',
      correctAnswer: 'No, keys must be immutable (lists are mutable)',
      difficulty: 5.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[4].id,
      content: 'Access dictionary values using keys. Use `.get()` to avoid KeyError if key doesn\'t exist.\n\n```python\nperson = {"name": "Bob"}\nage = person.get("age", 0)  # returns 0 if "age" not found\n```',
      question: 'What does .get() return if the key doesn\'t exist and no default is provided?',
      correctAnswer: 'None',
      difficulty: 4.5,
      cardType: CardType.BASIC,
    },

    // Control Flow (3 cards)
    {
      conceptId: concepts[5].id,
      content: '**If statements** allow conditional execution of code.\n\n```python\nif temperature > 30:\n    print("Hot")\nelif temperature > 20:\n    print("Warm")\nelse:\n    print("Cold")\n```',
      question: 'What keyword is used for an alternative condition after if?',
      correctAnswer: 'elif',
      difficulty: 3.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[5].id,
      content: 'Python uses **indentation** (usually 4 spaces) to define code blocks.\n\n```python\nif x > 0:\n    print("Positive")\n    print("This is also in the if block")\nprint("This is outside")\n```',
      question: 'What does Python use to define code blocks (instead of curly braces)?',
      correctAnswer: 'Indentation',
      difficulty: 3.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[5].id,
      content: 'You can use **logical operators** (and, or, not) to combine conditions.\n\n```python\nif age >= 18 and has_license:\n    print("Can drive")\n```',
      question: 'Which logical operator requires both conditions to be True?',
      correctAnswer: 'and',
      difficulty: 4.0,
      cardType: CardType.BASIC,
    },

    // Recursion (3 cards)
    {
      conceptId: concepts[6].id,
      content: '**Recursion** is when a function calls itself. Every recursive function needs a **base case** to stop.\n\n```python\ndef factorial(n):\n    if n <= 1:  # base case\n        return 1\n    return n * factorial(n - 1)\n```',
      question: 'What is the essential component that stops a recursive function?',
      correctAnswer: 'Base case',
      difficulty: 7.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[6].id,
      content: 'Without a base case, recursion causes a **stack overflow** error.\n\n```python\ndef infinite():\n    return infinite()  # no base case!\n```',
      question: 'What error occurs if a recursive function has no base case?',
      correctAnswer: 'Stack overflow or RecursionError',
      difficulty: 7.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[6].id,
      content: 'Recursion is useful for problems with **self-similar** structure (trees, factorial, Fibonacci).\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```',
      question: 'When is recursion most appropriate?',
      correctAnswer: 'For problems with self-similar or hierarchical structure',
      difficulty: 8.0,
      cardType: CardType.BASIC,
    },

    // Error Handling (3 cards)
    {
      conceptId: concepts[7].id,
      content: '**Try-except** blocks handle errors gracefully without crashing.\n\n```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print("Cannot divide by zero!")\n```',
      question: 'What block catches and handles exceptions in Python?',
      correctAnswer: 'except block',
      difficulty: 5.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[7].id,
      content: 'You can catch multiple exception types or use a general `except` clause.\n\n```python\ntry:\n    value = int("abc")\nexcept (ValueError, TypeError):\n    print("Invalid input")\n```',
      question: 'How do you catch multiple exception types in one except block?',
      correctAnswer: 'Put them in parentheses separated by commas',
      difficulty: 5.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[7].id,
      content: 'The **finally** block always executes, whether an exception occurred or not.\n\n```python\ntry:\n    file = open("data.txt")\nfinally:\n    file.close()  # always runs\n```',
      question: 'When does the finally block execute?',
      correctAnswer: 'Always, whether an exception occurred or not',
      difficulty: 6.0,
      cardType: CardType.BASIC,
    },

    // File I/O (3 cards)
    {
      conceptId: concepts[8].id,
      content: 'Open files with `open()` and use modes: "r" (read), "w" (write), "a" (append).\n\n```python\nwith open("file.txt", "r") as f:\n    content = f.read()\n```',
      question: 'What file mode is used to read a file?',
      correctAnswer: '"r" or "r" (read mode)',
      difficulty: 4.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[8].id,
      content: 'The **with statement** automatically closes the file when done.\n\n```python\nwith open("file.txt", "w") as f:\n    f.write("Hello")  # file auto-closes after\n```',
      question: 'What advantage does the "with" statement provide for file operations?',
      correctAnswer: 'Automatically closes the file when done',
      difficulty: 5.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[8].id,
      content: 'Use `.read()`, `.readline()`, or `.readlines()` to read file content.\n\n```python\nwith open("file.txt") as f:\n    lines = f.readlines()  # list of all lines\n```',
      question: 'Which method reads a file into a list of lines?',
      correctAnswer: 'readlines()',
      difficulty: 5.5,
      cardType: CardType.BASIC,
    },

    // Classes (3 cards)
    {
      conceptId: concepts[9].id,
      content: 'A **class** is a blueprint for creating objects. Use `class` keyword.\n\n```python\nclass Dog:\n    def __init__(self, name):\n        self.name = name\n\nmy_dog = Dog("Buddy")\n```',
      question: 'What method is called when a new instance of a class is created?',
      correctAnswer: '__init__ (constructor)',
      difficulty: 6.0,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[9].id,
      content: '**self** refers to the instance of the class. It must be the first parameter of instance methods.\n\n```python\nclass Cat:\n    def speak(self):\n        return "Meow"\n```',
      question: 'What does "self" refer to in a class method?',
      correctAnswer: 'The instance of the class',
      difficulty: 6.5,
      cardType: CardType.BASIC,
    },
    {
      conceptId: concepts[9].id,
      content: 'Classes can have **attributes** (data) and **methods** (functions).\n\n```python\nclass Person:\n    def __init__(self, name, age):\n        self.name = name  # attribute\n        self.age = age\n    \n    def greet(self):  # method\n        return f"Hi, I\'m {self.name}"\n```',
      question: 'What are the two main components of a class?',
      correctAnswer: 'Attributes (data) and methods (functions)',
      difficulty: 7.0,
      cardType: CardType.BASIC,
    },
  ]

  const cards = await Promise.all(
    cardData.map(card => prisma.card.create({ data: card }))
  )

  console.log(`✅ Created ${cards.length} learning cards`)

  // 4. Create Knowledge Graph in Neo4j
  console.log('🌳 Building Knowledge Graph in Neo4j...')

  try {
    // Create concept nodes
    for (const concept of concepts) {
      await neo4j.createConcept({
        id: concept.id,
        name: concept.name,
        domain: concept.domain,
        bloomLevel: concept.bloomLevel,
        difficulty: concept.estimatedDifficulty,
      })
    }

    // Create prerequisite relationships
    const prerequisites = [
      { from: 0, to: 1, weight: 0.8, strict: true },  // Variables → Functions
      { from: 1, to: 6, weight: 0.9, strict: true },  // Functions → Recursion
      { from: 3, to: 4, weight: 0.7, strict: false }, // Lists → Dictionaries
      { from: 5, to: 2, weight: 0.6, strict: false }, // Control Flow → Loops
      { from: 0, to: 5, weight: 0.7, strict: true },  // Variables → Control Flow
      { from: 2, to: 7, weight: 0.5, strict: false }, // Loops → Error Handling
      { from: 1, to: 9, weight: 0.8, strict: true },  // Functions → Classes
    ]

    for (const prereq of prerequisites) {
      await neo4j.createPrerequisite({
        conceptId: concepts[prereq.to].id,
        prerequisiteId: concepts[prereq.from].id,
        weight: prereq.weight,
        isStrict: prereq.strict,
      })
    }

    console.log(`✅ Created Knowledge Graph with ${concepts.length} nodes and ${prerequisites.length} edges`)
  } catch (error) {
    console.warn('⚠️  Neo4j not available - skipping Knowledge Graph creation')
    console.warn('   Error:', error)
  }

  // 5. Create initial scheduled items for demo user
  console.log('📅 Creating initial scheduled items...')

  const now = new Date()
  const demoProfileId = demoUser.learnerProfile?.id

  if (demoProfileId) {
    // Schedule first 10 cards for immediate review
    for (let i = 0; i < Math.min(10, cards.length); i++) {
      await prisma.scheduledItem.create({
        data: {
          learnerId: demoProfileId,
          cardId: cards[i].id,
          currentStability: 2.5,
          currentDifficulty: cards[i].difficulty,
          retrievability: 0.9,
          intervalDays: 0,
          nextDueDate: now, // Due now
          reviewCount: 0,
        }
      })
    }

    // Schedule remaining cards for future dates
    for (let i = 10; i < cards.length; i++) {
      const daysFromNow = Math.floor((i - 10) / 3) + 1
      const dueDate = new Date(now)
      dueDate.setDate(dueDate.getDate() + daysFromNow)

      await prisma.scheduledItem.create({
        data: {
          learnerId: demoProfileId,
          cardId: cards[i].id,
          currentStability: 2.5,
          currentDifficulty: cards[i].difficulty,
          retrievability: 0.9,
          intervalDays: daysFromNow,
          nextDueDate: dueDate,
          reviewCount: 0,
        }
      })
    }

    console.log(`✅ Created ${cards.length} scheduled items for demo user`)
  }

  // 6. Create initial competency states
  console.log('📊 Creating initial competency states...')

  if (demoProfileId) {
    for (const concept of concepts.slice(0, 5)) {
      await prisma.competencyState.create({
        data: {
          learnerId: demoProfileId,
          conceptId: concept.id,
          knowledgeProbability: 0.5,
          masteryLevel: 0.0,
          lastAssessed: now,
          evidenceCount: 0,
        }
      })
    }

    console.log('✅ Created initial competency states')
  }

  console.log('\n🎉 Seed completed successfully!')
  console.log('\n📊 Summary:')
  console.log(`   👤 Users: 3 (demo@nerdlearn.com, alice@example.com, bob@example.com)`)
  console.log(`   💡 Concepts: ${concepts.length}`)
  console.log(`   📝 Cards: ${cards.length}`)
  console.log(`   📅 Scheduled items: ${cards.length} (for demo user)`)
  console.log(`   🌳 Knowledge Graph: ${concepts.length} nodes, ${prerequisites.length} edges (if Neo4j available)`)
  console.log('\n🔐 Login credentials:')
  console.log('   Email: demo@nerdlearn.com')
  console.log('   Password: demo123')
  console.log('\n   Note: Passwords are placeholders - they need to be hashed by the API Gateway on first use')
}

main()
  .catch((e) => {
    console.error('❌ Seed failed:', e)
    process.exit(1)
  })
  .finally(async () => {
    await prisma.$disconnect()
    await neo4j.close()
  })
