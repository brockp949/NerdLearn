import { PrismaClient, BloomLevel, ResourceType, UserRole, ScheduleStatus } from '@prisma/client'

const prisma = new PrismaClient()

// Helper to initialize AGE graph
async function initGraph() {
  try {
    // Load AGE extension
    await prisma.$executeRaw`CREATE EXTENSION IF NOT EXISTS age;`
    await prisma.$executeRaw`LOAD 'age';`
    await prisma.$executeRaw`SET search_path = ag_catalog, "$user", public;`

    // Create graph if not exists
    const graphExists = await prisma.$queryRaw`SELECT count(*) FROM ag_graph WHERE name = 'nerdlearn_graph'`
    // @ts-ignore
    if (graphExists[0].count === 0n || graphExists[0].count === 0) {
      await prisma.$executeRaw`SELECT create_graph('nerdlearn_graph');`
      console.log('✅ Created AGE graph: nerdlearn_graph')
    } else {
      console.log('ℹ️  AGE graph exists')
    }
  } catch (e) {
    console.warn('⚠️  Failed to initialize AGE graph:', e)
  }
}

async function createPrerequisite(fromId: string, toId: string, weight: number) {
  try {
    const cypher = `
      MATCH (a:Concept {id: "${fromId}"}), (b:Concept {id: "${toId}"})
      MERGE (a)-[r:HAS_PREREQUISITE]->(b)
      SET r.weight = ${weight}
      RETURN r
    `
    await prisma.$executeRawUnsafe(`SELECT * FROM cypher('nerdlearn_graph', $$ ${cypher} $$) as (r agtype);`)
  } catch (e) {
    console.error(`❌ Failed to create edge ${fromId}->${toId}`, e)
  }
}

async function createConceptNode(concept: any) {
  try {
    const cypher = `
      MERGE (c:Concept {id: "${concept.id}"})
      SET c.name = "${concept.name}",
          c.domain = "${concept.domain}",
          c.difficulty = ${concept.avgDifficulty}
      RETURN c
    `
    await prisma.$executeRawUnsafe(`SELECT * FROM cypher('nerdlearn_graph', $$ ${cypher} $$) as (c agtype);`)
  } catch (e) {
    console.error(`❌ Failed to create node ${concept.name}`, e)
  }
}

function simpleHash(password: string): string {
  return `PLACEHOLDER_${password}`
}

async function main() {
  console.log('🌱 Starting seed...')
  console.log('🧹 Cleaning existing data...')

  await initGraph()

  // Clean data in correct order
  await prisma.scheduledItem.deleteMany()
  await prisma.evidence.deleteMany()
  await prisma.competencyState.deleteMany()
  await prisma.resource.deleteMany()
  await prisma.concept.deleteMany()
  await prisma.learnerProfile.deleteMany()
  await prisma.user.deleteMany()

  try {
    await prisma.$executeRaw`SELECT drop_graph('nerdlearn_graph', true);`
    await prisma.$executeRaw`SELECT create_graph('nerdlearn_graph');`
    console.log('✅ Reset AGE graph')
  } catch (e) { }

  console.log('✅ Cleaned existing data')

  // 1. Create Demo Users
  console.log('👤 Creating demo users...')

  const demoUser = await prisma.user.create({
    data: {
      email: 'demo@nerdlearn.com',
      username: 'demo',
      passwordHash: simpleHash('demo123'),
      role: UserRole.STUDENT,
      profile: {
        create: {
          cognitiveEmbedding: JSON.stringify([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
          fsrsStability: 2.5,
          fsrsDifficulty: 5.0,
          currentZpdLower: 0.35,
          currentZpdUpper: 0.70,
          totalXP: 0,
          level: 1,
          streakDays: 0,
          lastActivityDate: new Date(),
        }
      }
    },
    include: { profile: true }
  })

  // 2. Create Python Concepts
  console.log('💡 Creating Python concepts...')

  const concepts = await Promise.all([
    // [0]
    prisma.concept.create({
      data: { name: 'Python Variables', description: 'Variables and data storage', domain: 'Python', taxonomyLevel: BloomLevel.REMEMBER, avgDifficulty: 2.0, neoId: 'c_py_vars' }
    }),
    // [1]
    prisma.concept.create({
      data: { name: 'Data Types', description: 'Strings, Integers, Booleans', domain: 'Python', taxonomyLevel: BloomLevel.UNDERSTAND, avgDifficulty: 2.5, neoId: 'c_py_types' }
    }),
    // [2]
    prisma.concept.create({
      data: { name: 'Control Flow', description: 'If, Elif, Else conditionals', domain: 'Python', taxonomyLevel: BloomLevel.APPLY, avgDifficulty: 3.5, neoId: 'c_py_control' }
    }),
    // [3]
    prisma.concept.create({
      data: { name: 'Python Loops', description: 'For and While loops', domain: 'Python', taxonomyLevel: BloomLevel.APPLY, avgDifficulty: 4.0, neoId: 'c_py_loops' }
    }),
    // [4]
    prisma.concept.create({
      data: { name: 'Python Functions', description: 'Defining and invoking functions', domain: 'Python', taxonomyLevel: BloomLevel.CREATE, avgDifficulty: 5.0, neoId: 'c_py_funcs' }
    }),
    // [5]
    prisma.concept.create({
      data: { name: 'Python Lists', description: 'Ordered mutable sequences', domain: 'Python', taxonomyLevel: BloomLevel.APPLY, avgDifficulty: 3.5, neoId: 'c_py_lists' }
    }),
    // [6]
    prisma.concept.create({
      data: { name: 'Dictionaries', description: 'Key-value stores', domain: 'Python', taxonomyLevel: BloomLevel.ANALYZE, avgDifficulty: 4.5, neoId: 'c_py_dicts' }
    }),
    // [7]
    prisma.concept.create({
      data: { name: 'Error Handling', description: 'Try, Except blocks', domain: 'Python', taxonomyLevel: BloomLevel.EVALUATE, avgDifficulty: 5.5, neoId: 'c_py_errors' }
    }),
    // [8]
    prisma.concept.create({
      data: { name: 'File I/O', description: 'Reading and writing files', domain: 'Python', taxonomyLevel: BloomLevel.APPLY, avgDifficulty: 5.0, neoId: 'c_py_fileio' }
    }),
    // [9]
    prisma.concept.create({
      data: { name: 'Object-Oriented Programming', description: 'Classes and Objects', domain: 'Python', taxonomyLevel: BloomLevel.CREATE, avgDifficulty: 7.0, neoId: 'c_py_oop' }
    })
  ])

  console.log(`✅ Created ${concepts.length} concepts`)

  // 3. Create Resources (Curriculum)
  console.log('📝 Creating curriculum resources...')

  const curriculum = [
    // 0: Variables
    {
      conceptDiff: 0, title: 'Variable Assignment', type: ResourceType.EXERCISE, diff: 2.0,
      content: { type: 'flashcard', front: 'How do you assign the value 10 to variable x?', back: 'x = 10' }
    },
    {
      conceptDiff: 0, title: 'Variable Naming Rules', type: ResourceType.EXERCISE, diff: 2.5,
      content: { type: 'flashcard', front: 'Can a variable name start with a number?', back: 'No, it must start with a letter or underscore.' }
    },
    {
      conceptDiff: 0, title: 'Reassignment', type: ResourceType.EXERCISE, diff: 2.5,
      content: { type: 'multiple_choice', question: 'What happens if you run:\nx = 5\nx = "Hello"', options: ['Error', 'x is now "Hello"', 'x keeps value 5'], answer: 'x is now "Hello"' }
    },

    // 1: Data Types
    {
      conceptDiff: 1, title: 'Integer vs String', type: ResourceType.EXERCISE, diff: 2.5,
      content: { type: 'flashcard', front: 'What is the type of "42"?', back: 'String (str)' }
    },
    {
      conceptDiff: 1, title: 'Boolean Values', type: ResourceType.EXERCISE, diff: 2.5,
      content: { type: 'multiple_choice', question: 'Which is a valid Boolean in Python?', options: ['true', 'True', 'TRUE'], answer: 'True' }
    },

    // 2: Control Flow
    {
      conceptDiff: 2, title: 'If Statement Syntax', type: ResourceType.EXERCISE, diff: 3.0,
      content: { type: 'flashcard', front: 'What character MUST end an if statement line?', back: ': (Colon)' }
    },
    {
      conceptDiff: 2, title: 'Elif Usage', type: ResourceType.EXERCISE, diff: 3.5,
      content: { type: 'flashcard', front: 'When do you use "elif"?', back: 'To check another condition if the previous "if" was False.' }
    },
    {
      conceptDiff: 2, title: 'Logic Operators', type: ResourceType.EXERCISE, diff: 3.5,
      content: { type: 'multiple_choice', question: 'Which operator returns True if BOTH operands are True?', options: ['or', 'not', 'and'], answer: 'and' }
    },

    // 3: Loops
    {
      conceptDiff: 3, title: 'For Loop Range', type: ResourceType.EXERCISE, diff: 3.5,
      content: { type: 'flashcard', front: 'What does range(3) produce?', back: '0, 1, 2' }
    },
    {
      conceptDiff: 3, title: 'While Loop Condition', type: ResourceType.EXERCISE, diff: 4.0,
      content: { type: 'multiple_choice', question: 'When does a while loop stop?', options: ['When condition is True', 'When condition is False', 'Never'], answer: 'When condition is False' }
    },
    {
      conceptDiff: 3, title: 'Break Statement', type: ResourceType.EXERCISE, diff: 4.0,
      content: { type: 'flashcard', front: 'What does "break" do inside a loop?', back: 'Immediately exits the loop.' }
    },

    // 4: Functions
    {
      conceptDiff: 4, title: 'Function Definition', type: ResourceType.EXERCISE, diff: 4.5,
      content: { type: 'flashcard', front: 'Keyword to define a function?', back: 'def' }
    },
    {
      conceptDiff: 4, title: 'Return Value', type: ResourceType.EXERCISE, diff: 4.5,
      content: { type: 'multiple_choice', question: 'What does a function return if no "return" statement is used?', options: ['0', 'None', 'Error'], answer: 'None' }
    },
    {
      conceptDiff: 4, title: 'Arguments', type: ResourceType.EXERCISE, diff: 5.0,
      content: { type: 'flashcard', front: 'What are values passed into functions called?', back: 'Arguments (or Parameters)' }
    },

    // 5: Lists
    {
      conceptDiff: 5, title: 'List Indexing', type: ResourceType.EXERCISE, diff: 3.5,
      content: { type: 'flashcard', front: 'How to access the first element of list L?', back: 'L[0]' }
    },
    {
      conceptDiff: 5, title: 'List Append', type: ResourceType.EXERCISE, diff: 3.5,
      content: { type: 'flashcard', front: 'Method to add element to end of list?', back: '.append()' }
    },
    {
      conceptDiff: 5, title: 'List Slicing', type: ResourceType.EXERCISE, diff: 4.0,
      content: { type: 'multiple_choice', question: 'What is L[1:]?', options: ['First element', 'All except first', 'Last element'], answer: 'All except first' }
    },

    // 6: Dictionaries
    {
      conceptDiff: 6, title: 'Dict Syntax', type: ResourceType.EXERCISE, diff: 4.5,
      content: { type: 'flashcard', front: 'What brackets are used for dictionaries?', back: '{} (Curly braces)' }
    },
    {
      conceptDiff: 6, title: 'Key Lookup', type: ResourceType.EXERCISE, diff: 4.5,
      content: { type: 'flashcard', front: 'How to get value for key "k" in dict D?', back: 'D["k"]' }
    },

    // 9: OOP
    {
      conceptDiff: 9, title: 'Class Definition', type: ResourceType.EXERCISE, diff: 6.0,
      content: { type: 'flashcard', front: 'Keyword to create a class?', back: 'class' }
    },
    {
      conceptDiff: 9, title: 'The __init__ method', type: ResourceType.EXERCISE, diff: 6.5,
      content: { type: 'flashcard', front: 'What is __init__ used for?', back: 'Initializing new objects (Constructor)' }
    },
    {
      conceptDiff: 9, title: 'Self Parameter', type: ResourceType.EXERCISE, diff: 7.0,
      content: { type: 'multiple_choice', question: 'What is the first parameter of instance methods?', options: ['this', 'self', 'me'], answer: 'self' }
    }
  ]

  const createdResources = []
  for (const item of curriculum) {
    const res = await prisma.resource.create({
      data: {
        conceptId: concepts[item.conceptDiff].id,
        title: item.title,
        type: item.type,
        contentData: item.content,
        difficulty: item.diff
      }
    })
    createdResources.push(res)
  }

  console.log(`✅ Created ${createdResources.length} resources`)

  // 4. Create Knowledge Graph in Postgres (Apache AGE)
  console.log('🌳 Building Knowledge Graph in AGE...')

  try {
    // Create concept nodes
    for (const concept of concepts) {
      await createConceptNode(concept)
    }

    // Create prerequisite relationships
    // 0:Vars, 1:DataTypes, 2:Control, 3:Loops, 4:Funcs, 5:Lists, 6:Dicts, 7:Errors, 8:FileIO, 9:OOP
    const prerequisites = [
      { from: 0, to: 1, weight: 0.9 }, // Vars -> Types
      { from: 1, to: 2, weight: 0.8 }, // Types -> Control
      { from: 2, to: 3, weight: 0.8 }, // Control -> Loops
      { from: 1, to: 5, weight: 0.7 }, // Types -> Lists
      { from: 3, to: 5, weight: 0.6 }, // Loops -> Lists (iteration)
      { from: 5, to: 6, weight: 0.7 }, // Lists -> Dicts
      { from: 2, to: 4, weight: 0.9 }, // Control -> Funcs
      { from: 4, to: 7, weight: 0.6 }, // Funcs -> Errors
      { from: 4, to: 9, weight: 0.8 }, // Funcs -> OOP
      { from: 6, to: 9, weight: 0.7 }, // Dicts -> OOP
    ]

    for (const prereq of prerequisites) {
      await createPrerequisite(concepts[prereq.from].id, concepts[prereq.to].id, prereq.weight)
    }

    console.log(`✅ Created Knowledge Graph`)
  } catch (error) {
    console.warn('⚠️  Graph creation failed:', error)
  }

  // 5. Create Scheduled Items for Demo User
  console.log('📅 Creating scheduled items...')
  const profileId = demoUser.profile?.id
  const now = new Date()

  if (profileId) {
    // Shedule first 10 items
    for (let i = 0; i < 10 && i < createdResources.length; i++) {
      const res = createdResources[i]
      await prisma.scheduledItem.create({
        data: {
          learnerId: profileId,
          resourceId: res.id,
          dueDate: now,
          stability: 2.0,
          difficulty: res.difficulty,
          status: ScheduleStatus.DUE
        }
      })
    }
    console.log(`✅ Scheduled initial items`)
  }

  console.log('\n🎉 Seed completed successfully!')
}

main()
  .catch((e) => {
    console.error(e)
    process.exit(1)
  })
  .finally(async () => {
    await prisma.$disconnect()
  })
