# Track 004: Frontend Development (Phase 4)

**Goal**: Implement the frontend interfaces for the newly created backend features (Gamification, Causal Discovery, Adaptive Learning) and polish the overall UX.

## Context
The backend now supports complex adaptive learning (Interleaved Practice, RL), causal discovery, and research-backed gamification (Variable Rewards, Streak Shields). The frontend (`apps/web`) currently lacks the specific UI components to visualize and interact with these advanced features.

## Plan

### 1. Gamification UI
- [ ] **Reward Modal**: Create a "Loot Drop" component to display variable rewards (XP boosts, Badges, Streak Shields) with age-appropriate visuals (`apps/web/src/components/gamification/RewardModal.tsx`).
- [ ] **Dashboard Updates**: Update the main dashboard to show:
    - Current Level (with age-appropriate curve visualization).
    - Active Streak & Streak Shields available.
    - Recent Achievements.
- [ ] **Feedback Systems**: Implement visual cues for XP gain and level-ups that match the backend's age-group logic.

### 2. Adaptive Learning UI
- [ ] **Interleaved Session View**: Update the learning session interface to indicate *why* a specific card is shown (e.g., "Spaced Repetition due", "Interleaved Practice", "Prerequisite Check").
- [ ] **Knowledge Graph Visualization**: Create a visual graph component (using `react-force-graph` or similar) to render the Causal Discovery results (Concepts and Edges).

### 3. Integration
- [ ] **API Clients**: Update `apps/web/src/lib/api.ts` (or equivalent) to include methods for:
    - Triggering/Claiming rewards.
    - Fetching Causal Graph data.
    - Fetching detailed User Gamification Profile.
- [ ] **State Management**: Ensure local state reflects backend updates (e.g., XP goes up immediately after a session).

### 4. Polish
- [ ] Ensure responsive design for mobile users.
- [ ] Verify accessibility (ARIA labels) for new components.

## References
- `apps/api/app/gamification/variable_rewards.py` (Backend logic for rewards)
- `apps/api/app/adaptive/causal_discovery` (Backend logic for graph)
