
"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from "recharts";
import { Loader2 } from "lucide-react";

interface DNAProfile {
    user_id: string;
    resilience: number;
    impulsivity: number;
    curiosity: number;
    modality_preference: {
        visual: number;
        text: number;
        interactive: number;
    };
    traits: string[];
}

export function LearningDNACard({ userId }: { userId: string }) {
    const [data, setData] = useState<DNAProfile | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate fetching data (replace with actual API call)
        // In production: fetch(`/api/analytics/dna/${userId}`)
        setTimeout(() => {
            setData({
                user_id: userId,
                resilience: 0.8,
                impulsivity: 0.4,
                curiosity: 0.7,
                modality_preference: {
                    visual: 0.5,
                    text: 0.2,
                    interactive: 0.3
                },
                traits: ["Grit Master", "Visual Learner", "Explorer"]
            });
            setLoading(false);
        }, 1000);
    }, [userId]);

    if (loading) {
        return (
            <Card className="w-full h-[400px] flex items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </Card>
        );
    }

    if (!data) return null;

    const chartData = [
        { subject: "Resilience", A: data.resilience * 100, fullMark: 100 },
        { subject: "Focus", A: (1 - data.impulsivity) * 100, fullMark: 100 }, // Invert impulsivity for positive "Focus"
        { subject: "Curiosity", A: data.curiosity * 100, fullMark: 100 },
        { subject: "Visual", A: data.modality_preference.visual * 100, fullMark: 100 },
        { subject: "Interactive", A: data.modality_preference.interactive * 100, fullMark: 100 },
        { subject: "Textual", A: data.modality_preference.text * 100, fullMark: 100 },
    ];

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle>Learning DNA</CardTitle>
                <CardDescription>Your unique learning profile</CardDescription>
            </CardHeader>
            <CardContent>
                <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                            <PolarGrid />
                            <PolarAngleAxis dataKey="subject" />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} />
                            <Radar
                                name="You"
                                dataKey="A"
                                stroke="#8884d8"
                                fill="#8884d8"
                                fillOpacity={0.6}
                            />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>

                <div className="mt-4 flex flex-wrap gap-2 justify-center">
                    {data.traits.map((trait) => (
                        <span key={trait} className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium">
                            {trait}
                        </span>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
