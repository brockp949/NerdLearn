export type TestStatus = 'passed' | 'failed' | 'skipped' | 'running' | 'pending' | 'error';

export interface TestResult {
    id: string;
    name: string;
    status: TestStatus;
    duration: number; // in milliseconds
    category: string;
    filePath?: string;
    lineNumber?: number;
    errorMessage?: string;
    errorType?: string;
    stackTrace?: string;
    shortTrace?: string;
    stdout?: string;
    stderr?: string;
    timestamp: string;
}

export interface TestSuite {
    id: string;
    name: string;
    category: string;
    tests: TestResult[];
    totalTests: number;
    passed: number;
    failed: number;
    skipped: number;
    errors: number;
    duration: number;
    lastRun: string;
}

export interface TestSummary {
    totalSuites: number;
    totalTests: number;
    passed: number;
    failed: number;
    skipped: number;
    errors: number;
    passRate: number;
    duration: number;
    lastRun: string;
    suites: TestSuite[];
    running: boolean;
}

export interface AntigravityTestResult extends TestResult {
    goalVector: {
        primaryObjective: string;
        successCriteria: string[];
        failureConditions: string[];
    };
    gravityIntensity: 'LOW' | 'MEDIUM' | 'HIGH';
    driftDetected: boolean;
}

export interface FailedTestsResponse {
    tests: (TestResult & { suite: string })[];
    total: number;
    message: string;
}
