import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@radix-ui/react-tabs';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, BookOpen, Share2, Headphones, Activity } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { DiagramGenerator } from '../diagram/DiagramGenerator';

interface ContentMorphingUIProps {
    topic: string;
    initialContent: string;
}

type Modality = 'text' | 'diagram' | 'podcast';
type Persona = 'ELI5' | 'Academic' | 'Socratic';

/**
 * ContentMorphingUI
 * 
 * Research alignment:
 * - Holistic Content Morphing: Context-aware modality switching
 * - Conceptual State Vector: Maintains learning state across formats
 */
export const ContentMorphingUI: React.FC<ContentMorphingUIProps> = ({ topic, initialContent }) => {
    const [modality, setModality] = useState<Modality>('text');
    const [persona, setPersona] = useState<Persona>('Academic');
    const [content, setContent] = useState<string>(initialContent);
    const [loading, setLoading] = useState<boolean>(false);

    // Data containers
    const [diagramData, setDiagramData] = useState<{ nodes: any[], edges: any[] } | null>(null);
    const [podcastData, setPodcastData] = useState<{ audio_url: string, script: any[] } | null>(null);

    // Handle Text Style Transfer
    const handleStyleChange = async (newPersona: Persona) => {
        if (newPersona === persona) return;
        setLoading(true);
        setPersona(newPersona);

        try {
            const res = await axios.post('/api/transformation/style-transfer', {
                content: initialContent, // Always source from original to prevent drift
                target_persona: newPersona
            });
            setContent(res.data.transformed_content);
        } catch (err) {
            console.error("Style transfer failed", err);
        } finally {
            setLoading(false);
        }
    };

    // Handle Modality Switch
    const handleModalityChange = async (newModality: Modality) => {
        setModality(newModality);

        if (newModality === 'diagram' && !diagramData) {
            setLoading(true);
            try {
                const res = await axios.post('/api/transformation/diagram', {
                    content: content,
                    topic: topic
                });
                setDiagramData(res.data);
            } catch (err) {
                console.error("Diagram gen failed", err);
            } finally {
                setLoading(false);
            }
        } else if (newModality === 'podcast' && !podcastData) {
            setLoading(true);
            try {
                const res = await axios.post('/api/transformation/podcast', {
                    content: content,
                    topic: topic
                });
                setPodcastData(res.data);
            } catch (err) {
                console.error("Podcast gen failed", err);
            } finally {
                setLoading(false);
            }
        }
    };

    return (
        <Card className="w-full max-w-4xl mx-auto shadow-xl border-t-4 border-t-indigo-500">
            <CardHeader className="bg-slate-50">
                <div className="flex justify-between items-center">
                    <div>
                        <CardTitle className="text-2xl font-bold flex items-center gap-2">
                            <Activity className="h-6 w-6 text-indigo-600" />
                            {topic}
                        </CardTitle>
                        <CardDescription>Multi-Modal Learning Interface</CardDescription>
                    </div>

                    {/* Format Slider / Tabs */}
                    <div className="flex bg-slate-200 p-1 rounded-lg">
                        <Button
                            variant={modality === 'text' ? 'default' : 'ghost'}
                            size="sm"
                            onClick={() => handleModalityChange('text')}
                        >
                            <BookOpen className="h-4 w-4 mr-2" /> Text
                        </Button>
                        <Button
                            variant={modality === 'diagram' ? 'default' : 'ghost'}
                            size="sm"
                            onClick={() => handleModalityChange('diagram')}
                        >
                            <Share2 className="h-4 w-4 mr-2" /> Diagram
                        </Button>
                        <Button
                            variant={modality === 'podcast' ? 'default' : 'ghost'}
                            size="sm"
                            onClick={() => handleModalityChange('podcast')}
                        >
                            <Headphones className="h-4 w-4 mr-2" /> Podcast
                        </Button>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="p-6 min-h-[500px]">
                {loading ? (
                    <div className="flex flex-col items-center justify-center h-[400px]">
                        <Loader2 className="h-12 w-12 animate-spin text-indigo-500 mb-4" />
                        <p className="text-slate-500">Transforming content...</p>
                    </div>
                ) : (
                    <>
                        {/* Text View */}
                        {modality === 'text' && (
                            <div className="space-y-4">
                                <div className="flex gap-2 border-b pb-2">
                                    {(['ELI5', 'Academic', 'Socratic'] as Persona[]).map((p) => (
                                        <Button
                                            key={p}
                                            variant={persona === p ? "secondary" : "ghost"}
                                            size="sm"
                                            onClick={() => handleStyleChange(p)}
                                            className="text-xs"
                                        >
                                            {p}
                                        </Button>
                                    ))}
                                </div>
                                <ScrollArea className="h-[400px] w-full rounded-md border p-4">
                                    <div className="prose prose-slate max-w-none">
                                        <ReactMarkdown>
                                            {content}
                                        </ReactMarkdown>
                                    </div>
                                </ScrollArea>
                            </div>
                        )}

                        {/* Diagram View */}
                        {modality === 'diagram' && diagramData && (
                            <DiagramGenerator
                                initialNodes={diagramData.nodes}
                                initialEdges={diagramData.edges}
                            />
                        )}

                        {/* Podcast View */}
                        {modality === 'podcast' && podcastData && (
                            <div className="flex flex-col items-center gap-6 pt-10">
                                <div className="w-full max-w-md bg-indigo-50 p-6 rounded-2xl border border-indigo-100 text-center">
                                    <Headphones className="h-16 w-16 text-indigo-600 mx-auto mb-4" />
                                    <h3 className="text-xl font-semibold mb-2">{topic} - Deep Dive</h3>
                                    <audio controls className="w-full mt-4" src={podcastData.audio_url} />
                                </div>

                                <div className="w-full max-w-2xl">
                                    <h4 className="font-semibold mb-2 text-slate-700">Transcript</h4>
                                    <ScrollArea className="h-[200px] bg-slate-50 p-4 rounded border">
                                        {podcastData.script.map((line: any, i: number) => (
                                            <div key={i} className="mb-3">
                                                <span className="font-bold text-xs uppercase tracking-wide text-indigo-500">
                                                    {line.speaker}
                                                </span>
                                                <p className="text-sm text-slate-700">{line.text}</p>
                                            </div>
                                        ))}
                                    </ScrollArea>
                                </div>
                            </div>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );
};
