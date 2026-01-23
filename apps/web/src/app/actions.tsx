"use server";

import { createStreamableUI } from "@ai-sdk/rsc";
import { openai } from "@ai-sdk/openai";
import { streamUI } from "@ai-sdk/rsc";
import { z } from "zod";
import { RetentionGraph } from "@/components/chat/retention-graph";

export async function submitUserMessage(input: string) {
    "use server";

    const ui = createStreamableUI();

    (async () => {
        const result = await streamUI({
            model: openai("gpt-4o"), // Use a strong model for tool calling
            initial: <div className="animate-pulse">Thinking...</div>,
            system: `You are an expert in Spaced Repetition Systems (SRS) and the FSRS algorithm.
      If the user asks about retention curves, forgetting curves, or stability, you SHOULD render a graph using the 'renderRetentionGraph' tool.
      Do not just explain it in text if a visualization would be better.
      If the user asks to modify the stability, call the tool with the new value.`,
            messages: [
                {
                    role: "user",
                    content: input,
                },
            ],
            text: ({ content, done }) => {
                if (done) {
                    return <div className="mb-4 text-zinc-200 whitespace-pre-wrap">{content}</div>
                }
                return <div className="mb-4 text-zinc-200 whitespace-pre-wrap">{content}</div>
            },
            tools: {
                renderRetentionGraph: {
                    description: "Render an interactive FSRS retention curve graph (forgetting curve).",
                    // @ts-ignore
                    parameters: z.object({
                        initialStability: z.number().default(5).describe("The stability (S) in days."),
                    }),
                    generate: async ({ initialStability }) => {
                        return <RetentionGraph initialStability={initialStability} />;
                    },
                },
            },
        });

        ui.done(result.value);
    })();

    return {
        id: Date.now(),
        display: ui.value,
    };
}
