"use client";

import { useMemo, useState } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";

export function RetentionGraph({ initialStability = 5 }: { initialStability?: number }) {
    const [stability, setStability] = useState(initialStability);

    // Calculate the exponential decay curve: R = S^t (simplified approximation) or R = (1 + factor * t/s)^decay
    // Using the actual FSRS approximation: R = (1 + factor * t / S) ^ -1 (power law forgetting)
    // Let's use the standard exponential forgetting curve for simplicity if FSRS specific formula is complex to implement without more params: R = e^(-t/S)
    // Or the one from the research paper: R = (1 + factor * t / 9 * S) ^ -1
    // Let's use a standard decay function that looks good: R = 100 * (1 + t / (9 * S))^-1

    const data = useMemo(() => {
        return Array.from({ length: 31 }, (_, i) => {
            // FSRS inspired curve
            const ret = 100 * Math.pow(1 + i / (9 * stability), -1);
            return {
                day: i,
                retention: ret
            };
        });
    }, [stability]);

    return (
        <Card className="w-full max-w-md bg-zinc-950 border-zinc-800">
            <CardHeader>
                <CardTitle className="text-zinc-100">Forgetting Curve</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="h-[200px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="day" stroke="#666" fontSize={12} />
                            <YAxis stroke="#666" fontSize={12} domain={[0, 100]} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Line
                                type="monotone"
                                dataKey="retention"
                                stroke="#3b82f6"
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="mt-4 space-y-2">
                    <div className="flex justify-between text-sm text-zinc-400">
                        <span>Stability (Days to 90%)</span>
                        <span>{stability.toFixed(1)}</span>
                    </div>
                    <Slider
                        value={[stability]}
                        min={0.1}
                        max={30}
                        step={0.1}
                        onValueChange={(vals) => setStability(vals[0])}
                        className="py-4"
                    />
                </div>
            </CardContent>
        </Card>
    );
}
