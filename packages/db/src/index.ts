/**
 * NerdLearn Database Client
 * Exports Prisma client and utility functions
 */

import { PrismaClient } from '@prisma/client';

// Singleton pattern for Prisma client
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
  });

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma;

// Export Prisma types
export * from '@prisma/client';

// Utility function to disconnect
export const disconnect = async () => {
  await prisma.$disconnect();
};

// Health check
export const healthCheck = async (): Promise<boolean> => {
  try {
    await prisma.$queryRaw`SELECT 1`;
    return true;
  } catch {
    return false;
  }
};
