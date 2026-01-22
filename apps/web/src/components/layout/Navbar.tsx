"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "../ui/button";
import { Brain, Swords, Map, User } from "lucide-react";

const navItems = [
    { name: "Dojo", href: "/dojo", icon: Swords },
    { name: "Quests", href: "/quests", icon: Map },
    { name: "Brain", href: "/brain", icon: Brain },
    { name: "Profile", href: "/profile", icon: User },
];

export function Navbar() {
    const pathname = usePathname();

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-white/10 bg-background/60 backdrop-blur-xl">
            <div className="container mx-auto h-full flex items-center justify-between px-4">
                <div className="flex items-center gap-2">
                    <div className="size-8 rounded bg-primary flex items-center justify-center font-bold text-primary-foreground">
                        NL
                    </div>
                    <span className="text-xl font-bold bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
                        NerdLearn
                    </span>
                </div>

                <div className="flex items-center gap-1">
                    {navItems.map((item) => {
                        const isActive = pathname.startsWith(item.href);
                        return (
                            <Link key={item.href} href={item.href}>
                                <Button
                                    variant={isActive ? "secondary" : "ghost"}
                                    size="sm"
                                    className={cn(
                                        "gap-2 transition-all",
                                        isActive && "bg-white/10 text-white shadow-[0_0_15px_rgba(255,255,255,0.1)]"
                                    )}
                                >
                                    <item.icon className="size-4" />
                                    {item.name}
                                </Button>
                            </Link>
                        );
                    })}
                </div>

                <div className="flex items-center gap-4">
                    <div className="text-xs text-muted-foreground">
                        <span className="font-mono text-primary">Level 4</span> Architect
                    </div>
                    <div className="size-8 rounded-full bg-gradient-to-br from-primary to-purple-600 ring-2 ring-white/10" />
                </div>
            </div>
        </nav>
    );
}
