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
    prisma.concept.create({
      data: {
        name: 'Python Variables',
        description: 'Understanding variables and data storage in Python',
        domain: 'Python',
        taxonomyLevel: BloomLevel.REMEMBER,
        avgDifficulty: 3.0,
        neoId: 'c_py_vars' // Explicit ID for graph correlation if needed
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Functions',
        description: 'Creating and using functions in Python',
        domain: 'Python',
        taxonomyLevel: BloomLevel.UNDERSTAND,
        avgDifficulty: 4.5,
        neoId: 'c_py_funcs'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Loops',
        description: 'For loops, while loops, and iteration',
        domain: 'Python',
        taxonomyLevel: BloomLevel.APPLY,
        avgDifficulty: 5.0,
        neoId: 'c_py_loops'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Lists',
        domain: 'Python',
        taxonomyLevel: BloomLevel.APPLY,
        avgDifficulty: 4.0,
        neoId: 'c_py_lists'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Dictionaries',
        domain: 'Python',
        taxonomyLevel: BloomLevel.APPLY,
        avgDifficulty: 4.5,
        neoId: 'c_py_dicts'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Control Flow',
        domain: 'Python',
        taxonomyLevel: BloomLevel.UNDERSTAND,
        avgDifficulty: 3.5,
        neoId: 'c_py_control'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Recursion',
        domain: 'Python',
        taxonomyLevel: BloomLevel.ANALYZE,
        avgDifficulty: 7.5,
        neoId: 'c_py_recursion'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Error Handling',
        domain: 'Python',
        taxonomyLevel: BloomLevel.APPLY,
        avgDifficulty: 5.5,
        neoId: 'c_py_errors'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'File I/O',
        domain: 'Python',
        taxonomyLevel: BloomLevel.APPLY,
        avgDifficulty: 5.0,
        neoId: 'c_py_fileio'
      }
    }),
    prisma.concept.create({
      data: {
        name: 'Python Classes',
        domain: 'Python',
        taxonomyLevel: BloomLevel.CREATE,
        avgDifficulty: 6.5,
        neoId: 'c_py_classes'
      }
    })
  ])

  console.log(`✅ Created ${concepts.length} concepts`)

  // 3. Create Resources (Flashcards)
  console.log('📝 Creating learning resources...')

  // Mapping first concept 'Variables' with 3 cards
  const c0 = concepts[0]
  const resources = await Promise.all([
    prisma.resource.create({
      data: {
        conceptId: c0.id,
        title: 'Variable Assignment',
        type: ResourceType.EXERCISE,
        contentData: {
          type: 'flashcard',
          front: 'What operator is used to assign a value to a variable in Python?',
          back: '= (equals sign)',
          content: 'A **variable** in Python is a named storage location...'
        },
        difficulty: 2.5
      }
    }),
    prisma.resource.create({
      data: {
        conceptId: c0.id,
        title: 'Dynamic Typing',
        type: ResourceType.EXERCISE,
        contentData: {
          type: 'flashcard',
          front: 'What does "dynamically typed" mean in Python?',
          back: 'Variables can change type, and you don\'t need to declare types explicitly'
        },
        difficulty: 3.0
      }
    }),
    prisma.resource.create({
      data: {
        conceptId: c0.id,
        title: 'Variable Naming',
        type: ResourceType.EXERCISE,
        contentData: {
          type: 'flashcard',
          front: 'Can a variable name start with a number in Python?',
          back: 'No, variable names must start with a letter or underscore'
        },
        difficulty: 3.5
      }
    })
  ])

  console.log(`✅ Created ${resources.length} resources`)

  // 4. Create Knowledge Graph in Postgres (Apache AGE)
  console.log('🌳 Building Knowledge Graph in AGE...')

  try {
    // Create concept nodes
    for (const concept of concepts) {
      await createConceptNode(concept)
    }

    // Create prerequisite relationships
    const prerequisites = [
      { from: 0, to: 1, weight: 0.8 },
      { from: 1, to: 6, weight: 0.9 },
      { from: 3, to: 4, weight: 0.7 },
      { from: 5, to: 2, weight: 0.6 },
      { from: 0, to: 5, weight: 0.7 },
      { from: 2, to: 7, weight: 0.5 },
      { from: 1, to: 9, weight: 0.8 },
    ]

    for (const prereq of prerequisites) {
      await createPrerequisite(concepts[prereq.from].id, concepts[prereq.to].id, prereq.weight)
    }

    console.log(`✅ Created Knowledge Graph`)
  } catch (error) {
    console.warn('⚠️  Graph creation failed:', error)
  }

  // 5. Create Scheduled Items
  console.log('📅 Creating scheduled items...')
  const profileId = demoUser.profile?.id
  const now = new Date()

  if (profileId) {
    for (const res of resources) {
      await prisma.scheduledItem.create({
        data: {
          learnerId: profileId,
          resourceId: res.id,
          dueDate: now,
          stability: 2.5,
          difficulty: res.difficulty,
          status: ScheduleStatus.DUE
        }
      })
    }
    console.log(`✅ Scheduled ${resources.length} items`)
  }

  // 6. Competency State
  if (profileId) {
    await prisma.competencyState.create({
      data: {
        learnerId: profileId,
        conceptId: c0.id,
        masteryProbability: 0.2,
        lastPracticed: now
      }
    })
    console.log('✅ Created initial competency')
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
