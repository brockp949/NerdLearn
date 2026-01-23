import React from 'react';

interface SplitScreenLayoutProps {
    children: React.ReactNode;
    sidebar: React.ReactNode;
    sidebarOpen?: boolean;
}

export function SplitScreenLayout({ children, sidebar, sidebarOpen = true }: SplitScreenLayoutProps) {
    return (
        <div className="flex h-[calc(100vh-64px)] overflow-hidden bg-gray-100">
            {/* Main Content Area */}
            <div className="flex-1 overflow-auto p-6 transition-all duration-300">
                <div className="mx-auto max-w-5xl h-full">
                    {children}
                </div>
            </div>

            {/* Sidebar (Chat) */}
            <div
                className={`bg-white border-l border-zinc-200 transition-all duration-300 ease-in-out flex flex-col ${sidebarOpen ? 'w-[400px]' : 'w-0 overflow-hidden'
                    }`}
            >
                <div className="h-full flex flex-col">
                    {/* We ensure the sidebar content takes full height */}
                    {sidebar}
                </div>
            </div>
        </div>
    );
}
