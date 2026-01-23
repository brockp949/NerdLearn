-- CreateEnum
CREATE TYPE "UserRole" AS ENUM ('STUDENT', 'INSTRUCTOR', 'ADMIN', 'RESEARCHER');

-- CreateEnum
CREATE TYPE "BloomLevel" AS ENUM ('REMEMBER', 'UNDERSTAND', 'APPLY', 'ANALYZE', 'EVALUATE', 'CREATE');

-- CreateEnum
CREATE TYPE "EvidenceType" AS ENUM ('EXPLICIT_CORRECT', 'EXPLICIT_INCORRECT', 'IMPLICIT_ENGAGEMENT', 'IMPLICIT_STRUGGLE', 'IMPLICIT_MASTERY', 'CODE_SUBMISSION', 'PEER_INTERACTION');

-- CreateEnum
CREATE TYPE "ResourceType" AS ENUM ('VIDEO', 'ARTICLE', 'INTERACTIVE', 'EXERCISE', 'ASSESSMENT', 'WORKED_EXAMPLE', 'SIMULATION');

-- CreateEnum
CREATE TYPE "ScheduleStatus" AS ENUM ('PENDING', 'DUE', 'OVERDUE', 'COMPLETED');

-- CreateEnum
CREATE TYPE "AchievementType" AS ENUM ('STREAK_MILESTONE', 'XP_MILESTONE', 'CONCEPT_MASTERY', 'SPEED_DEMON', 'PERFECTIONIST', 'EXPLORER', 'HELPER', 'CREATIVE_SOLUTION');

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "passwordHash" TEXT NOT NULL,
    "role" "UserRole" NOT NULL DEFAULT 'STUDENT',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "lastLoginAt" TIMESTAMP(3),

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LearnerProfile" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "cognitiveEmbedding" JSONB NOT NULL,
    "fsrsStability" DOUBLE PRECISION NOT NULL DEFAULT 2.5,
    "fsrsDifficulty" DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    "fsrsRetrievability" DOUBLE PRECISION NOT NULL DEFAULT 0.9,
    "avgResponseTime" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "avgAccuracy" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "engagementScore" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "persistenceIndex" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "currentZpdLower" DOUBLE PRECISION NOT NULL DEFAULT 0.35,
    "currentZpdUpper" DOUBLE PRECISION NOT NULL DEFAULT 0.70,
    "optimalDifficulty" DOUBLE PRECISION NOT NULL DEFAULT 0.50,
    "totalXP" INTEGER NOT NULL DEFAULT 0,
    "level" INTEGER NOT NULL DEFAULT 1,
    "streakDays" INTEGER NOT NULL DEFAULT 0,
    "lastActivityDate" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "freezesAvailable" INTEGER NOT NULL DEFAULT 2,
    "preferredModality" TEXT NOT NULL DEFAULT 'mixed',
    "dailyGoalMinutes" INTEGER NOT NULL DEFAULT 30,
    "notificationsEnabled" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LearnerProfile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Concept" (
    "id" TEXT NOT NULL,
    "neoId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "taxonomyLevel" "BloomLevel" NOT NULL DEFAULT 'REMEMBER',
    "avgDifficulty" DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    "lexicalDensity" DOUBLE PRECISION,
    "conceptualDensity" DOUBLE PRECISION,
    "domain" TEXT NOT NULL,
    "subdomain" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Concept_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ConceptPrerequisite" (
    "id" TEXT NOT NULL,
    "conceptId" TEXT NOT NULL,
    "prerequisiteId" TEXT NOT NULL,
    "weight" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "isStrict" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "ConceptPrerequisite_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CompetencyState" (
    "id" TEXT NOT NULL,
    "learnerId" TEXT NOT NULL,
    "conceptId" TEXT NOT NULL,
    "masteryProbability" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "confidence" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "knowledgeState" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "successRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "totalAttempts" INTEGER NOT NULL DEFAULT 0,
    "successfulAttempts" INTEGER NOT NULL DEFAULT 0,
    "lastPracticed" TIMESTAMP(3),
    "nextReviewDue" TIMESTAMP(3),
    "reviewCount" INTEGER NOT NULL DEFAULT 0,
    "itemStability" DOUBLE PRECISION NOT NULL DEFAULT 2.5,
    "itemDifficulty" DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "CompetencyState_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Evidence" (
    "id" TEXT NOT NULL,
    "competencyId" TEXT NOT NULL,
    "type" "EvidenceType" NOT NULL,
    "weight" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "data" JSONB NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "decayRate" DOUBLE PRECISION NOT NULL DEFAULT 0.95,
    "sessionId" TEXT,
    "activityId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Evidence_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Resource" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "type" "ResourceType" NOT NULL,
    "contentUrl" TEXT,
    "contentData" JSONB,
    "estimatedMinutes" INTEGER,
    "difficulty" DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    "conceptId" TEXT,
    "embeddingId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Resource_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ScheduledItem" (
    "id" TEXT NOT NULL,
    "learnerId" TEXT NOT NULL,
    "resourceId" TEXT NOT NULL,
    "dueDate" TIMESTAMP(3) NOT NULL,
    "stability" DOUBLE PRECISION NOT NULL,
    "difficulty" DOUBLE PRECISION NOT NULL,
    "status" "ScheduleStatus" NOT NULL DEFAULT 'PENDING',
    "priority" INTEGER NOT NULL DEFAULT 0,
    "blockId" TEXT,
    "interleavingIndex" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "completedAt" TIMESTAMP(3),

    CONSTRAINT "ScheduledItem_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LearningSession" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "startTime" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "endTime" TIMESTAMP(3),
    "durationSeconds" INTEGER,
    "itemsCompleted" INTEGER NOT NULL DEFAULT 0,
    "avgAccuracy" DOUBLE PRECISION,
    "avgResponseTime" DOUBLE PRECISION,
    "mouseEvents" INTEGER NOT NULL DEFAULT 0,
    "keystrokes" INTEGER NOT NULL DEFAULT 0,
    "pauseCount" INTEGER NOT NULL DEFAULT 0,
    "deviceType" TEXT,
    "browserAgent" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LearningSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LearningResponse" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "sessionId" TEXT,
    "resourceId" TEXT NOT NULL,
    "isCorrect" BOOLEAN,
    "responseData" JSONB NOT NULL,
    "responseTime" INTEGER,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "dwellTime" INTEGER,
    "hesitationCount" INTEGER,
    "confidenceScore" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LearningResponse_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Achievement" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "type" "AchievementType" NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "iconUrl" TEXT,
    "xpReward" INTEGER NOT NULL DEFAULT 0,
    "rarity" DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    "unlockedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Achievement_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Course" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "domain" TEXT NOT NULL,
    "instructorId" TEXT,
    "isPublished" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Course_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CourseEnrollment" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "courseId" TEXT NOT NULL,
    "enrolledAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),
    "progress" DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    CONSTRAINT "CourseEnrollment_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "User_username_key" ON "User"("username");

-- CreateIndex
CREATE INDEX "User_email_idx" ON "User"("email");

-- CreateIndex
CREATE INDEX "User_username_idx" ON "User"("username");

-- CreateIndex
CREATE UNIQUE INDEX "LearnerProfile_userId_key" ON "LearnerProfile"("userId");

-- CreateIndex
CREATE INDEX "LearnerProfile_userId_idx" ON "LearnerProfile"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "Concept_neoId_key" ON "Concept"("neoId");

-- CreateIndex
CREATE INDEX "Concept_domain_subdomain_idx" ON "Concept"("domain", "subdomain");

-- CreateIndex
CREATE INDEX "Concept_taxonomyLevel_idx" ON "Concept"("taxonomyLevel");

-- CreateIndex
CREATE INDEX "ConceptPrerequisite_conceptId_idx" ON "ConceptPrerequisite"("conceptId");

-- CreateIndex
CREATE INDEX "ConceptPrerequisite_prerequisiteId_idx" ON "ConceptPrerequisite"("prerequisiteId");

-- CreateIndex
CREATE UNIQUE INDEX "ConceptPrerequisite_conceptId_prerequisiteId_key" ON "ConceptPrerequisite"("conceptId", "prerequisiteId");

-- CreateIndex
CREATE INDEX "CompetencyState_learnerId_idx" ON "CompetencyState"("learnerId");

-- CreateIndex
CREATE INDEX "CompetencyState_conceptId_idx" ON "CompetencyState"("conceptId");

-- CreateIndex
CREATE INDEX "CompetencyState_nextReviewDue_idx" ON "CompetencyState"("nextReviewDue");

-- CreateIndex
CREATE UNIQUE INDEX "CompetencyState_learnerId_conceptId_key" ON "CompetencyState"("learnerId", "conceptId");

-- CreateIndex
CREATE INDEX "Evidence_competencyId_idx" ON "Evidence"("competencyId");

-- CreateIndex
CREATE INDEX "Evidence_timestamp_idx" ON "Evidence"("timestamp");

-- CreateIndex
CREATE INDEX "Evidence_type_idx" ON "Evidence"("type");

-- CreateIndex
CREATE UNIQUE INDEX "Resource_embeddingId_key" ON "Resource"("embeddingId");

-- CreateIndex
CREATE INDEX "Resource_type_idx" ON "Resource"("type");

-- CreateIndex
CREATE INDEX "Resource_conceptId_idx" ON "Resource"("conceptId");

-- CreateIndex
CREATE INDEX "ScheduledItem_learnerId_dueDate_idx" ON "ScheduledItem"("learnerId", "dueDate");

-- CreateIndex
CREATE INDEX "ScheduledItem_status_idx" ON "ScheduledItem"("status");

-- CreateIndex
CREATE INDEX "LearningSession_userId_startTime_idx" ON "LearningSession"("userId", "startTime");

-- CreateIndex
CREATE INDEX "LearningResponse_userId_idx" ON "LearningResponse"("userId");

-- CreateIndex
CREATE INDEX "LearningResponse_resourceId_idx" ON "LearningResponse"("resourceId");

-- CreateIndex
CREATE INDEX "LearningResponse_sessionId_idx" ON "LearningResponse"("sessionId");

-- CreateIndex
CREATE INDEX "Achievement_userId_idx" ON "Achievement"("userId");

-- CreateIndex
CREATE INDEX "Achievement_type_idx" ON "Achievement"("type");

-- CreateIndex
CREATE INDEX "Course_domain_idx" ON "Course"("domain");

-- CreateIndex
CREATE INDEX "Course_isPublished_idx" ON "Course"("isPublished");

-- CreateIndex
CREATE INDEX "CourseEnrollment_userId_idx" ON "CourseEnrollment"("userId");

-- CreateIndex
CREATE INDEX "CourseEnrollment_courseId_idx" ON "CourseEnrollment"("courseId");

-- CreateIndex
CREATE UNIQUE INDEX "CourseEnrollment_userId_courseId_key" ON "CourseEnrollment"("userId", "courseId");

-- AddForeignKey
ALTER TABLE "LearnerProfile" ADD CONSTRAINT "LearnerProfile_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ConceptPrerequisite" ADD CONSTRAINT "ConceptPrerequisite_conceptId_fkey" FOREIGN KEY ("conceptId") REFERENCES "Concept"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ConceptPrerequisite" ADD CONSTRAINT "ConceptPrerequisite_prerequisiteId_fkey" FOREIGN KEY ("prerequisiteId") REFERENCES "Concept"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CompetencyState" ADD CONSTRAINT "CompetencyState_learnerId_fkey" FOREIGN KEY ("learnerId") REFERENCES "LearnerProfile"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CompetencyState" ADD CONSTRAINT "CompetencyState_conceptId_fkey" FOREIGN KEY ("conceptId") REFERENCES "Concept"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Evidence" ADD CONSTRAINT "Evidence_competencyId_fkey" FOREIGN KEY ("competencyId") REFERENCES "CompetencyState"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Resource" ADD CONSTRAINT "Resource_conceptId_fkey" FOREIGN KEY ("conceptId") REFERENCES "Concept"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ScheduledItem" ADD CONSTRAINT "ScheduledItem_learnerId_fkey" FOREIGN KEY ("learnerId") REFERENCES "LearnerProfile"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ScheduledItem" ADD CONSTRAINT "ScheduledItem_resourceId_fkey" FOREIGN KEY ("resourceId") REFERENCES "Resource"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LearningSession" ADD CONSTRAINT "LearningSession_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LearningResponse" ADD CONSTRAINT "LearningResponse_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LearningResponse" ADD CONSTRAINT "LearningResponse_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "LearningSession"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LearningResponse" ADD CONSTRAINT "LearningResponse_resourceId_fkey" FOREIGN KEY ("resourceId") REFERENCES "Resource"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Achievement" ADD CONSTRAINT "Achievement_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CourseEnrollment" ADD CONSTRAINT "CourseEnrollment_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CourseEnrollment" ADD CONSTRAINT "CourseEnrollment_courseId_fkey" FOREIGN KEY ("courseId") REFERENCES "Course"("id") ON DELETE CASCADE ON UPDATE CASCADE;
