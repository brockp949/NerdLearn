/**
 * Neo4j Knowledge Graph Client
 * Manages the concept relationships and prerequisite structures
 */

import neo4j, { Driver, Session } from 'neo4j-driver';

class KnowledgeGraph {
  private driver: Driver;

  constructor() {
    const uri = process.env.NEO4J_URI || 'bolt://localhost:7687';
    const user = process.env.NEO4J_USER || 'neo4j';
    const password = process.env.NEO4J_PASSWORD || 'nerdlearn_dev_password';

    this.driver = neo4j.driver(uri, neo4j.auth.basic(user, password));
  }

  async getSession(): Promise<Session> {
    return this.driver.session();
  }

  async close(): Promise<void> {
    await this.driver.close();
  }

  /**
   * Initialize the Knowledge Graph schema with constraints and indexes
   */
  async initializeSchema(): Promise<void> {
    const session = await this.getSession();
    try {
      // Create constraints
      await session.run(`
        CREATE CONSTRAINT concept_id IF NOT EXISTS
        FOR (c:Concept) REQUIRE c.id IS UNIQUE
      `);

      await session.run(`
        CREATE CONSTRAINT resource_id IF NOT EXISTS
        FOR (r:Resource) REQUIRE r.id IS UNIQUE
      `);

      // Create indexes
      await session.run(`
        CREATE INDEX concept_name IF NOT EXISTS
        FOR (c:Concept) ON (c.name)
      `);

      await session.run(`
        CREATE INDEX concept_domain IF NOT EXISTS
        FOR (c:Concept) ON (c.domain)
      `);

      console.log('✅ Neo4j schema initialized');
    } finally {
      await session.close();
    }
  }

  /**
   * Create a new concept node in the Knowledge Graph
   */
  async createConcept(params: {
    id: string;
    name: string;
    domain: string;
    subdomain?: string;
    taxonomyLevel: string;
    difficulty: number;
  }): Promise<void> {
    const session = await this.getSession();
    try {
      await session.run(
        `
        CREATE (c:Concept {
          id: $id,
          name: $name,
          domain: $domain,
          subdomain: $subdomain,
          taxonomyLevel: $taxonomyLevel,
          difficulty: $difficulty,
          createdAt: datetime()
        })
        `,
        params
      );
    } finally {
      await session.close();
    }
  }

  /**
   * Create a prerequisite relationship between concepts
   * Creates edge: prerequisite -> concept (must learn A before B)
   */
  async createPrerequisite(params: {
    conceptId: string;
    prerequisiteId: string;
    weight: number;
    isStrict: boolean;
  }): Promise<void> {
    const session = await this.getSession();
    try {
      await session.run(
        `
        MATCH (c:Concept {id: $conceptId})
        MATCH (p:Concept {id: $prerequisiteId})
        CREATE (p)-[r:HAS_PREREQUISITE {
          weight: $weight,
          isStrict: $isStrict,
          createdAt: datetime()
        }]->(c)
        `,
        params
      );
    } finally {
      await session.close();
    }
  }

  /**
   * Get all prerequisites for a concept (with depth)
   * Returns the full prerequisite tree up to specified depth
   */
  async getPrerequisites(conceptId: string, depth: number = 3): Promise<any[]> {
    const session = await this.getSession();
    try {
      const result = await session.run(
        `
        MATCH path = (c:Concept {id: $conceptId})<-[:HAS_PREREQUISITE*1..$depth]-(p:Concept)
        RETURN p, length(path) as depth
        ORDER BY depth ASC
        `,
        { conceptId, depth }
      );

      return result.records.map((record) => ({
        concept: record.get('p').properties,
        depth: record.get('depth').toNumber(),
      }));
    } finally {
      await session.close();
    }
  }

  /**
   * Find learning path between two concepts
   * Uses shortest path algorithm to find optimal learning sequence
   */
  async findLearningPath(fromConceptId: string, toConceptId: string): Promise<any[]> {
    const session = await this.getSession();
    try {
      const result = await session.run(
        `
        MATCH (start:Concept {id: $fromConceptId}),
              (end:Concept {id: $toConceptId}),
              path = shortestPath((start)-[:HAS_PREREQUISITE*]->(end))
        RETURN [node in nodes(path) | node.name] as path,
               [rel in relationships(path) | rel.weight] as weights
        `,
        { fromConceptId, toConceptId }
      );

      if (result.records.length === 0) return [];

      return result.records.map((record) => ({
        path: record.get('path'),
        weights: record.get('weights').map((w: any) => w.toNumber()),
      }));
    } finally {
      await session.close();
    }
  }

  /**
   * Get concept difficulty distribution across domain
   * Useful for curriculum balancing
   */
  async getDomainDifficultyDistribution(domain: string): Promise<any> {
    const session = await this.getSession();
    try {
      const result = await session.run(
        `
        MATCH (c:Concept {domain: $domain})
        RETURN
          avg(c.difficulty) as avgDifficulty,
          stdev(c.difficulty) as stdDifficulty,
          min(c.difficulty) as minDifficulty,
          max(c.difficulty) as maxDifficulty,
          count(c) as conceptCount
        `,
        { domain }
      );

      const record = result.records[0];
      return {
        avgDifficulty: record.get('avgDifficulty'),
        stdDifficulty: record.get('stdDifficulty'),
        minDifficulty: record.get('minDifficulty'),
        maxDifficulty: record.get('maxDifficulty'),
        conceptCount: record.get('conceptCount').toNumber(),
      };
    } finally {
      await session.close();
    }
  }

  /**
   * Recommend next concepts based on learner's mastered concepts
   * Uses graph traversal to find concepts where prerequisites are met
   */
  async recommendNextConcepts(
    masteredConceptIds: string[],
    limit: number = 5
  ): Promise<any[]> {
    const session = await this.getSession();
    try {
      const result = await session.run(
        `
        MATCH (mastered:Concept)
        WHERE mastered.id IN $masteredConceptIds
        MATCH (mastered)-[:HAS_PREREQUISITE]->(next:Concept)
        WHERE NOT next.id IN $masteredConceptIds

        // Check if all prerequisites are mastered
        OPTIONAL MATCH (next)<-[:HAS_PREREQUISITE]-(prereq:Concept)
        WITH next, collect(prereq.id) as allPrereqs
        WHERE all(p IN allPrereqs WHERE p IN $masteredConceptIds)

        RETURN next, next.difficulty as difficulty
        ORDER BY difficulty ASC
        LIMIT $limit
        `,
        { masteredConceptIds, limit }
      );

      return result.records.map((record) => ({
        concept: record.get('next').properties,
        difficulty: record.get('difficulty'),
      }));
    } finally {
      await session.close();
    }
  }

  /**
   * Health check for Neo4j connection
   */
  async healthCheck(): Promise<boolean> {
    const session = await this.getSession();
    try {
      await session.run('RETURN 1');
      return true;
    } catch (error) {
      console.error('Neo4j health check failed:', error);
      return false;
    } finally {
      await session.close();
    }
  }
}

export const knowledgeGraph = new KnowledgeGraph();
export default KnowledgeGraph;
