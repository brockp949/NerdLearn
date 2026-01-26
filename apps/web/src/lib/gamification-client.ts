
import axios, { AxiosInstance } from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface Reward {
  id: str;
  name: str;
  rarity: "common" | "rare" | "epic" | "legendary";
  reward_type: "xp" | "badge" | "streak_shield" | "cosmetic";
  value: any;
}

export interface RewardFeedback {
  item: Reward;
  visual_effect: string;
  audio_cue: string;
  message: string;
}

export interface RewardTriggerResponse {
  reward_earned: boolean;
  reward?: Reward;
  feedback?: RewardFeedback;
}

export interface UserProfile {
  user_id: number;
  username: string;
  level: number;
  total_xp: number;
  xp_to_next_level: number;
  level_progress: number;
  streak_days: number;
  stats: Record<string, any>;
  achievements: any[];
}

export interface LeaderboardEntry {
  rank: number;
  username: string;
  level: number;
  total_xp: number;
  streak_days: number;
}

// ============================================================================
// CLIENT CLASS
// ============================================================================

class GamificationClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
    })
  }

  async triggerReward(userId: number, masteryLevel: number, ageGroup: string = "adult"): Promise<RewardTriggerResponse> {
    const response = await this.client.post<RewardTriggerResponse>('/api/gamification/trigger-reward', {
      user_id: userId,
      mastery_level: masteryLevel,
      age_group: ageGroup
    })
    return response.data
  }

  async getUserProfile(userId: number): Promise<UserProfile> {
    const response = await this.client.get<UserProfile>(`/api/gamification/profile/${userId}`)
    return response.data
  }

  async getAchievements(userId: number) {
    const response = await this.client.get(`/api/gamification/achievements?user_id=${userId}`)
    return response.data
  }

  async getLeaderboard(timePeriod: string = "all_time") {
    const response = await this.client.get(`/api/gamification/leaderboard?time_period=${timePeriod}`)
    return response.data
  }
  
  async getSkillTree(userId: number, courseId: number) {
      const response = await this.client.get(`/api/gamification/skill-tree?user_id=${userId}&course_id=${courseId}`)
      return response.data
  }
}

export const gamificationClient = new GamificationClient()
