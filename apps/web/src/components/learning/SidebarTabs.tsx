'use client';

import * as React from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { cn } from '@/lib/utils';
import { Bot, LineChart, Network, Beaker } from 'lucide-react';

const Tabs = TabsPrimitive.Root;
const TabsList = TabsPrimitive.List;
const TabsTrigger = TabsPrimitive.Trigger;
const TabsContent = TabsPrimitive.Content;

interface SidebarTabsProps {
    chatContent: React.ReactNode;
    progressContent: React.ReactNode;
    graphContent: React.ReactNode;
    testingContent?: React.ReactNode;
}

export function SidebarTabs({ chatContent, progressContent, graphContent, testingContent }: SidebarTabsProps) {
    return (
        <Tabs defaultValue="chat" className="flex flex-col h-full">
            <TabsList className="flex border-b border-zinc-200 bg-gray-50/50">
                <TabsTrigger
                    value="chat"
                    className={cn(
                        "flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-medium text-gray-500 transition-all hover:text-gray-900 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600 data-[state=active]:bg-white"
                    )}
                >
                    <Bot size={16} />
                    <span className="hidden sm:inline">AI Tutor</span>
                </TabsTrigger>
                <TabsTrigger
                    value="progress"
                    className={cn(
                        "flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-medium text-gray-500 transition-all hover:text-gray-900 border-b-2 border-transparent data-[state=active]:border-purple-500 data-[state=active]:text-purple-600 data-[state=active]:bg-white"
                    )}
                >
                    <LineChart size={16} />
                    <span className="hidden sm:inline">Progress</span>
                </TabsTrigger>
                <TabsTrigger
                    value="graph"
                    className={cn(
                        "flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-medium text-gray-500 transition-all hover:text-gray-900 border-b-2 border-transparent data-[state=active]:border-green-500 data-[state=active]:text-green-600 data-[state=active]:bg-white"
                    )}
                >
                    <Network size={16} />
                    <span className="hidden sm:inline">Skills</span>
                </TabsTrigger>
                <TabsTrigger
                    value="testing"
                    className={cn(
                        "flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-medium text-gray-500 transition-all hover:text-gray-900 border-b-2 border-transparent data-[state=active]:border-orange-500 data-[state=active]:text-orange-600 data-[state=active]:bg-white"
                    )}
                >
                    <Beaker size={16} />
                    <span className="hidden sm:inline">Testing</span>
                </TabsTrigger>
            </TabsList>
            <TabsContent value="chat" className="flex-1 overflow-hidden data-[state=active]:flex flex-col">
                {chatContent}
            </TabsContent>
            <TabsContent value="progress" className="flex-1 overflow-auto p-4 bg-gray-50/50 data-[state=active]:flex flex-col">
                {progressContent}
            </TabsContent>
            <TabsContent value="graph" className="flex-1 overflow-hidden data-[state=active]:flex flex-col bg-slate-50">
                {graphContent}
            </TabsContent>
            <TabsContent value="testing" className="flex-1 overflow-hidden data-[state=active]:flex flex-col bg-orange-50/30">
                {testingContent}
            </TabsContent>
        </Tabs>
    );
}
