"use client";

import { useState, useRef, useEffect } from "react";
import { submitUserMessage } from "@/app/actions";
import { Button } from "@/components/ui/button"; // Assuming these exist or I'll use standard ones
import { Input } from "@/components/ui/input";
import { Send, Bot, User } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";

// Define the structure for our messages
interface Message {
    id: number;
    role: "user" | "assistant";
    display: React.ReactNode;
}

import { cn } from "@/lib/utils";

export function ChatInterface({ className }: { className?: string }) {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 0,
            role: "assistant",
            display: <div>Hello! accurate retention requires understanding stability. Ask me about the FSRS curve.</div>
        }
    ]);
    const [inputValue, setInputValue] = useState("");
    const [isPending, setIsPending] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!inputValue.trim()) return;

        const userMessage: Message = {
            id: Date.now(),
            role: "user",
            display: <div>{inputValue}</div>,
        };

        setMessages((prev) => [...prev, userMessage]);
        setInputValue("");
        setIsPending(true);

        try {
            const response = await submitUserMessage(inputValue);
            setMessages((prev) => [
                ...prev,
                {
                    id: response.id,
                    role: "assistant",
                    display: response.display,
                },
            ]);
        } catch (error) {
            console.error(error);
        } finally {
            setIsPending(false);
        }
    };

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    return (
        <Card className={`flex flex-col bg-zinc-950/90 backdrop-blur-md border-zinc-800 shadow-2xl ${className || "h-[600px] w-[400px]"}`}>
            <div className="p-4 border-b border-zinc-800 flex items-center gap-2">
                <Bot className="size-5 text-blue-500" />
                <h2 className="font-semibold text-zinc-100">NerdLearn AI</h2>
            </div>

            <ScrollArea className="flex-1 p-4" ref={scrollRef}>
                <div className="flex flex-col gap-4">
                    {messages.map((message) => (
                        <div
                            key={message.id}
                            className={`flex items-start gap-3 ${message.role === "user" ? "flex-row-reverse" : ""
                                }`}
                        >
                            <div
                                className={`size-8 rounded-full flex items-center justify-center shrink-0 ${message.role === "user"
                                    ? "bg-zinc-700 text-zinc-100"
                                    : "bg-blue-900/30 text-blue-400"
                                    }`}
                            >
                                {message.role === "user" ? <User size={14} /> : <Bot size={14} />}
                            </div>
                            <div
                                className={`rounded-lg p-3 max-w-[85%] text-sm ${message.role === "user"
                                    ? "bg-zinc-800 text-zinc-100"
                                    : "text-zinc-300"
                                    }`}
                            >
                                {message.display}
                            </div>
                        </div>
                    ))}
                    {isPending && (
                        <div className="flex items-start gap-3">
                            <div className="size-8 rounded-full bg-blue-900/30 flex items-center justify-center shrink-0">
                                <Bot size={14} className="text-blue-400" />
                            </div>
                            <div className="text-zinc-400 text-sm animate-pulse flex items-center mt-2">
                                Thinking...
                            </div>
                        </div>
                    )}
                </div>
            </ScrollArea>

            <form onSubmit={handleSubmit} className="p-4 border-t border-zinc-800 flex gap-2">
                <Input
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Ask about retention curves..."
                    className="bg-zinc-900/50 border-zinc-700 text-zinc-100 focus-visible:ring-blue-500/50"
                />
                <Button
                    type="submit"
                    size="icon"
                    disabled={isPending}
                    className="bg-blue-600 hover:bg-blue-500 text-white"
                >
                    <Send size={18} />
                </Button>
            </form>
        </Card>
    );
}
