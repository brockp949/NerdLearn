'use client';

import React, { useState } from 'react';
import { useTestSummary, useRunTests } from '@/hooks/use-tests';
import { TestSuite, TestStatus, TestResult } from '@/types/testing';
import {
    CheckCircle2,
    XCircle,
    AlertCircle,
    Play,
    RefreshCw,
    ChevronDown,
    ChevronRight,
    Clock,
    Beaker,
    Shield,
    FileCode,
    AlertTriangle,
    Copy,
    ExternalLink
} from 'lucide-react';

const statusColors: Record<TestStatus, string> = {
    passed: 'text-green-500',
    failed: 'text-red-500',
    skipped: 'text-yellow-500',
    running: 'text-blue-500',
    pending: 'text-gray-400',
    error: 'text-red-600',
};

const statusBgColors: Record<TestStatus, string> = {
    passed: 'bg-green-50 border-green-200',
    failed: 'bg-red-50 border-red-200',
    skipped: 'bg-yellow-50 border-yellow-200',
    running: 'bg-blue-50 border-blue-200',
    pending: 'bg-gray-50 border-gray-200',
    error: 'bg-red-100 border-red-300',
};

const statusIcons: Record<TestStatus, React.ReactNode> = {
    passed: <CheckCircle2 size={14} className="text-green-500" />,
    failed: <XCircle size={14} className="text-red-500" />,
    skipped: <AlertCircle size={14} className="text-yellow-500" />,
    running: <RefreshCw size={14} className="text-blue-500 animate-spin" />,
    pending: <Clock size={14} className="text-gray-400" />,
    error: <AlertTriangle size={14} className="text-red-600" />,
};

interface TestDetailProps {
    test: TestResult;
    onClose: () => void;
}

function TestDetail({ test, onClose }: TestDetailProps) {
    const [showFullTrace, setShowFullTrace] = useState(false);
    const [copied, setCopied] = useState(false);

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const isFailed = test.status === 'failed' || test.status === 'error';

    return (
        <div className={`mt-2 p-3 rounded-lg border ${statusBgColors[test.status]}`}>
            {/* Header with file info */}
            <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2 text-xs text-gray-600">
                    <FileCode size={12} />
                    <span className="font-mono">
                        {test.filePath}
                        {test.lineNumber && `:${test.lineNumber}`}
                    </span>
                </div>
                <button
                    onClick={onClose}
                    className="text-gray-400 hover:text-gray-600 text-xs"
                >
                    Close
                </button>
            </div>

            {/* Error info for failed tests */}
            {isFailed && (
                <div className="space-y-2">
                    {/* Error type and message */}
                    {(test.errorType || test.errorMessage) && (
                        <div className="bg-white rounded p-2 border border-red-200">
                            {test.errorType && (
                                <div className="text-xs font-semibold text-red-700 mb-1">
                                    {test.errorType}
                                </div>
                            )}
                            {test.errorMessage && (
                                <div className="text-sm text-red-600 font-mono break-words">
                                    {test.errorMessage}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Stack trace */}
                    {(test.shortTrace || test.stackTrace) && (
                        <div className="bg-gray-900 rounded p-2 overflow-hidden">
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-gray-400">Stack Trace</span>
                                <div className="flex gap-1">
                                    <button
                                        onClick={() => copyToClipboard(test.stackTrace || test.shortTrace || '')}
                                        className="text-xs text-gray-400 hover:text-white flex items-center gap-1"
                                    >
                                        <Copy size={10} />
                                        {copied ? 'Copied!' : 'Copy'}
                                    </button>
                                    {test.stackTrace && test.stackTrace !== test.shortTrace && (
                                        <button
                                            onClick={() => setShowFullTrace(!showFullTrace)}
                                            className="text-xs text-blue-400 hover:text-blue-300 ml-2"
                                        >
                                            {showFullTrace ? 'Show less' : 'Show full'}
                                        </button>
                                    )}
                                </div>
                            </div>
                            <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-x-auto max-h-48 overflow-y-auto">
                                {showFullTrace ? test.stackTrace : (test.shortTrace || test.stackTrace)}
                            </pre>
                        </div>
                    )}

                    {/* Stdout/Stderr */}
                    {test.stdout && (
                        <div className="bg-gray-100 rounded p-2">
                            <div className="text-xs text-gray-500 mb-1">Captured Output</div>
                            <pre className="text-xs text-gray-700 font-mono whitespace-pre-wrap max-h-24 overflow-y-auto">
                                {test.stdout}
                            </pre>
                        </div>
                    )}
                    {test.stderr && (
                        <div className="bg-yellow-50 rounded p-2 border border-yellow-200">
                            <div className="text-xs text-yellow-700 mb-1">Stderr</div>
                            <pre className="text-xs text-yellow-800 font-mono whitespace-pre-wrap max-h-24 overflow-y-auto">
                                {test.stderr}
                            </pre>
                        </div>
                    )}
                </div>
            )}

            {/* Success info for passed tests */}
            {test.status === 'passed' && (
                <div className="text-sm text-green-700">
                    Test passed in {test.duration}ms
                </div>
            )}

            {/* Skipped info */}
            {test.status === 'skipped' && (
                <div className="text-sm text-yellow-700">
                    Test was skipped
                    {test.errorMessage && `: ${test.errorMessage}`}
                </div>
            )}

            {/* Duration */}
            <div className="mt-2 text-xs text-gray-500">
                Duration: {test.duration}ms | Run at: {new Date(test.timestamp).toLocaleString()}
            </div>
        </div>
    );
}

interface TestSuiteCardProps {
    suite: TestSuite;
    expanded: boolean;
    onToggle: () => void;
}

function TestSuiteCard({ suite, expanded, onToggle }: TestSuiteCardProps) {
    const [selectedTest, setSelectedTest] = useState<string | null>(null);

    const passRate = suite.totalTests > 0
        ? Math.round((suite.passed / suite.totalTests) * 100)
        : 0;

    const hasFailed = suite.failed > 0 || suite.errors > 0;

    // Sort tests: failed/error first
    const sortedTests = [...suite.tests].sort((a, b) => {
        const priority: Record<TestStatus, number> = {
            failed: 0,
            error: 1,
            skipped: 2,
            running: 3,
            pending: 4,
            passed: 5,
        };
        return priority[a.status] - priority[b.status];
    });

    return (
        <div className={`border rounded-lg overflow-hidden bg-white ${hasFailed ? 'border-red-300' : 'border-gray-200'}`}>
            <button
                onClick={onToggle}
                className={`w-full flex items-center justify-between p-3 hover:bg-gray-50 transition ${hasFailed ? 'bg-red-50' : ''}`}
            >
                <div className="flex items-center gap-2">
                    {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    <span className="font-medium text-sm text-gray-900">{suite.name}</span>
                    <span className="text-xs text-gray-500 px-2 py-0.5 bg-gray-100 rounded">
                        {suite.category}
                    </span>
                    {hasFailed && (
                        <span className="text-xs text-red-600 px-2 py-0.5 bg-red-100 rounded font-medium">
                            {suite.failed + suite.errors} failed
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1 text-xs">
                        <span className="text-green-600">{suite.passed}</span>
                        <span className="text-gray-400">/</span>
                        <span className="text-gray-600">{suite.totalTests}</span>
                    </div>
                    <div
                        className={`w-16 h-1.5 rounded-full bg-gray-200 overflow-hidden`}
                    >
                        <div
                            className={`h-full rounded-full transition-all ${
                                passRate === 100 ? 'bg-green-500' :
                                passRate >= 80 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${passRate}%` }}
                        />
                    </div>
                </div>
            </button>

            {expanded && (
                <div className="border-t border-gray-100 divide-y divide-gray-50">
                    {sortedTests.map((test) => (
                        <div key={test.id}>
                            <button
                                onClick={() => setSelectedTest(selectedTest === test.id ? null : test.id)}
                                className={`w-full px-3 py-2 flex items-center justify-between hover:bg-gray-50 text-left ${
                                    test.status === 'failed' || test.status === 'error' ? 'bg-red-25' : ''
                                }`}
                            >
                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                    {statusIcons[test.status]}
                                    <span className={`text-xs truncate ${
                                        test.status === 'failed' || test.status === 'error'
                                            ? 'text-red-700 font-medium'
                                            : 'text-gray-700'
                                    }`}>
                                        {test.name}
                                    </span>
                                </div>
                                <div className="flex items-center gap-2">
                                    {test.errorMessage && (
                                        <span className="text-xs text-red-500 max-w-32 truncate hidden sm:block">
                                            {test.errorMessage}
                                        </span>
                                    )}
                                    <span className="text-xs text-gray-400 whitespace-nowrap">
                                        {test.duration}ms
                                    </span>
                                    {(test.status === 'failed' || test.status === 'error' || test.status === 'skipped') && (
                                        <ChevronRight
                                            size={12}
                                            className={`text-gray-400 transition-transform ${
                                                selectedTest === test.id ? 'rotate-90' : ''
                                            }`}
                                        />
                                    )}
                                </div>
                            </button>

                            {selectedTest === test.id && (
                                <div className="px-3 pb-3">
                                    <TestDetail
                                        test={test}
                                        onClose={() => setSelectedTest(null)}
                                    />
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export function TestingPanel() {
    const { data: summary, loading, error, refetch } = useTestSummary();
    const runTests = useRunTests();
    const [expandedSuites, setExpandedSuites] = useState<Set<string>>(new Set());

    const toggleSuite = (suiteId: string) => {
        setExpandedSuites(prev => {
            const next = new Set(prev);
            if (next.has(suiteId)) {
                next.delete(suiteId);
            } else {
                next.add(suiteId);
            }
            return next;
        });
    };

    // Auto-expand suites with failures
    React.useEffect(() => {
        if (summary?.suites) {
            const failedSuites = summary.suites
                .filter(s => s.failed > 0 || s.errors > 0)
                .map(s => s.id);
            if (failedSuites.length > 0) {
                setExpandedSuites(new Set(failedSuites));
            }
        }
    }, [summary?.suites]);

    const handleRunTests = async () => {
        try {
            await runTests.mutateAsync({});
            // Poll for results - the hook auto-refreshes when running
        } catch (e) {
            console.error('Failed to run tests:', e);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-4">
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                    <p className="text-sm text-red-700">Failed to load tests</p>
                    <button
                        onClick={() => refetch()}
                        className="mt-2 text-xs text-red-600 hover:text-red-800"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    const failedCount = (summary?.failed || 0) + (summary?.errors || 0);

    return (
        <div className="h-full flex flex-col">
            {/* Header Stats */}
            <div className={`p-4 border-b border-gray-200 ${
                failedCount > 0
                    ? 'bg-gradient-to-r from-red-50 to-orange-50'
                    : 'bg-gradient-to-r from-green-50 to-emerald-50'
            }`}>
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                        <Shield className={`w-5 h-5 ${failedCount > 0 ? 'text-red-600' : 'text-green-600'}`} />
                        <h3 className="font-semibold text-gray-900">Antigravity Testing</h3>
                        {summary?.running && (
                            <span className="flex items-center gap-1 text-xs text-blue-600 bg-blue-100 px-2 py-0.5 rounded">
                                <RefreshCw size={10} className="animate-spin" />
                                Running...
                            </span>
                        )}
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={() => refetch()}
                            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-white rounded transition"
                            title="Refresh"
                        >
                            <RefreshCw size={16} />
                        </button>
                        <button
                            onClick={handleRunTests}
                            disabled={runTests.isPending || summary?.running}
                            className={`flex items-center gap-1.5 px-3 py-1.5 text-white text-xs font-medium rounded transition ${
                                failedCount > 0
                                    ? 'bg-red-600 hover:bg-red-700'
                                    : 'bg-green-600 hover:bg-green-700'
                            } disabled:opacity-50`}
                        >
                            {runTests.isPending || summary?.running ? (
                                <RefreshCw size={14} className="animate-spin" />
                            ) : (
                                <Play size={14} />
                            )}
                            Run All
                        </button>
                    </div>
                </div>

                {/* Summary Stats */}
                <div className="grid grid-cols-5 gap-2">
                    <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                        <p className="text-lg font-bold text-gray-900">{summary?.totalTests || 0}</p>
                        <p className="text-xs text-gray-500">Total</p>
                    </div>
                    <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                        <p className="text-lg font-bold text-green-600">{summary?.passed || 0}</p>
                        <p className="text-xs text-gray-500">Passed</p>
                    </div>
                    <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                        <p className="text-lg font-bold text-red-600">{failedCount}</p>
                        <p className="text-xs text-gray-500">Failed</p>
                    </div>
                    <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                        <p className="text-lg font-bold text-yellow-600">{summary?.skipped || 0}</p>
                        <p className="text-xs text-gray-500">Skipped</p>
                    </div>
                    <div className="bg-white rounded-lg p-2 text-center shadow-sm">
                        <p className={`text-lg font-bold ${
                            (summary?.passRate || 0) === 100 ? 'text-green-600' :
                            (summary?.passRate || 0) >= 80 ? 'text-yellow-600' : 'text-red-600'
                        }`}>{summary?.passRate || 0}%</p>
                        <p className="text-xs text-gray-500">Pass Rate</p>
                    </div>
                </div>

                {/* Failure summary banner */}
                {failedCount > 0 && (
                    <div className="mt-3 p-2 bg-red-100 border border-red-200 rounded-lg">
                        <div className="flex items-center gap-2 text-sm text-red-700">
                            <AlertTriangle size={14} />
                            <span className="font-medium">{failedCount} test(s) need attention</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Test Suites */}
            <div className="flex-1 overflow-auto p-4 space-y-2">
                {!summary?.suites || summary.suites.length === 0 ? (
                    <div className="text-center py-8">
                        <Beaker className="w-12 h-12 mx-auto text-gray-300 mb-3" />
                        <p className="text-sm text-gray-500">No test results yet</p>
                        <button
                            onClick={handleRunTests}
                            className="mt-3 text-xs text-orange-600 hover:text-orange-700"
                        >
                            Run tests to get started
                        </button>
                    </div>
                ) : (
                    summary.suites.map((suite) => (
                        <TestSuiteCard
                            key={suite.id}
                            suite={suite}
                            expanded={expandedSuites.has(suite.id)}
                            onToggle={() => toggleSuite(suite.id)}
                        />
                    ))
                )}
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-gray-200 bg-gray-50">
                <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>
                        Last run: {summary?.lastRun ? new Date(summary.lastRun).toLocaleString() : 'Never'}
                    </span>
                    {summary?.duration && summary.duration > 0 && (
                        <span>Duration: {(summary.duration / 1000).toFixed(1)}s</span>
                    )}
                </div>
            </div>
        </div>
    );
}
