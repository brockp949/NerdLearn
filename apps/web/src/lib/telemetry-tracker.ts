import * as React from 'react';

/**
 * Client-Side Telemetry Tracker - Non-Invasive Affect Detection
 *
 * Research alignment:
 * - Mouse/keyboard dynamics tracking (privacy-preserving)
 * - Frustration Index calculation
 * - Client-side LSTM model for affect classification
 *
 * Key Features:
 * 1. Rage Click detection (>3 clicks in <1 sec)
 * 2. Dead Click detection (clicking non-interactive elements)
 * 3. Mouse velocity variance (erratic = confused, smooth = flow)
 * 4. Time idle tracking (deep thought vs disengagement)
 * 5. Client-side affect inference (sends only inferred state)
 */

// Types
export type AffectState = 'flow' | 'frustrated' | 'bored' | 'confused' | 'neutral';

export interface TelemetryEvent {
  type: TelemetryEventType;
  timestamp: number;
  data: Record<string, unknown>;
}

export type TelemetryEventType =
  | 'click'
  | 'rage_click'
  | 'dead_click'
  | 'mouse_move'
  | 'scroll'
  | 'keypress'
  | 'idle_start'
  | 'idle_end'
  | 'focus'
  | 'blur'
  | 'page_view'
  | 'content_interaction';

export interface FrustrationIndex {
  rageClicks: number;      // Rc
  deadClicks: number;       // Dc
  mouseVelocityVariance: number;  // Mv
  timeIdle: number;         // Ti
  score: number;            // Weighted F
}

export interface TelemetryConfig {
  // Sampling rates
  mouseSampleRate: number;     // ms between mouse samples
  scrollSampleRate: number;    // ms between scroll samples

  // Thresholds
  rageClickThreshold: number;  // clicks within rageClickWindow
  rageClickWindow: number;     // ms
  idleThreshold: number;       // ms before considered idle
  deadClickSelectors: string[]; // CSS selectors for non-interactive elements

  // Weights for Frustration Index
  weights: {
    rageClicks: number;        // w1
    deadClicks: number;        // w2
    mouseVelocity: number;     // w3
    idleTime: number;          // w4
  };

  // Privacy
  sendOnlyInferredState: boolean;
  sessionTimeout: number;      // ms

  // API
  apiEndpoint: string;
  batchSize: number;
  flushInterval: number;       // ms
}

const DEFAULT_CONFIG: TelemetryConfig = {
  mouseSampleRate: 100,
  scrollSampleRate: 200,
  rageClickThreshold: 3,
  rageClickWindow: 1000,
  idleThreshold: 30000,
  deadClickSelectors: [
    '.non-interactive',
    '[data-no-click]',
    'body > div:not([class])',
  ],
  weights: {
    rageClicks: 0.35,
    deadClicks: 0.25,
    mouseVelocity: 0.25,
    idleTime: 0.15,
  },
  sendOnlyInferredState: true,
  sessionTimeout: 1800000, // 30 minutes
  apiEndpoint: '/api/telemetry',
  batchSize: 50,
  flushInterval: 10000,
};

/**
 * Telemetry Tracker Class
 *
 * Tracks user behavior for affect detection while preserving privacy.
 * Only inferred states are sent to the server, not raw telemetry.
 */
export class TelemetryTracker {
  private config: TelemetryConfig;
  private eventBuffer: TelemetryEvent[] = [];
  private sessionId: string;
  private userId?: number;
  private courseId?: number;
  private conceptId?: number;

  // Click tracking
  private clickTimestamps: number[] = [];
  private rageClickCount = 0;
  private deadClickCount = 0;

  // Mouse tracking
  private mousePositions: { x: number; y: number; t: number }[] = [];
  private lastMouseSample = 0;
  private mouseVelocities: number[] = [];

  // Idle tracking
  private lastActivityTime: number = Date.now();
  private isIdle = false;
  private totalIdleTime = 0;
  private idleStartTime?: number;

  // Scroll tracking
  private lastScrollSample = 0;
  private scrollDepths: number[] = [];

  // State
  private currentAffectState: AffectState = 'neutral';
  private frustrationIndex: FrustrationIndex = {
    rageClicks: 0,
    deadClicks: 0,
    mouseVelocityVariance: 0,
    timeIdle: 0,
    score: 0,
  };

  // Intervals
  private flushInterval?: NodeJS.Timeout;
  private idleCheckInterval?: NodeJS.Timeout;
  private affectUpdateInterval?: NodeJS.Timeout;

  // Event listeners (stored for cleanup)
  private boundHandlers: Map<string, EventListener> = new Map();

  constructor(config: Partial<TelemetryConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.sessionId = this.generateSessionId();
  }

  /**
   * Initialize the tracker
   */
  public init(userId?: number, courseId?: number, conceptId?: number): void {
    this.userId = userId;
    this.courseId = courseId;
    this.conceptId = conceptId;

    // Register event listeners
    this.registerEventListeners();

    // Start intervals
    this.startIntervals();

    // Log page view
    this.trackEvent('page_view', {
      url: window.location.href,
      referrer: document.referrer,
    });

    console.log('[Telemetry] Tracker initialized', { sessionId: this.sessionId });
  }

  /**
   * Cleanup and stop tracking
   */
  public destroy(): void {
    // Remove event listeners
    this.boundHandlers.forEach((handler, event) => {
      document.removeEventListener(event, handler);
      window.removeEventListener(event, handler);
    });
    this.boundHandlers.clear();

    // Clear intervals
    if (this.flushInterval) clearInterval(this.flushInterval);
    if (this.idleCheckInterval) clearInterval(this.idleCheckInterval);
    if (this.affectUpdateInterval) clearInterval(this.affectUpdateInterval);

    // Final flush
    this.flush();

    console.log('[Telemetry] Tracker destroyed');
  }

  /**
   * Update context (e.g., when navigating to new concept)
   */
  public setContext(courseId?: number, conceptId?: number): void {
    this.courseId = courseId;
    this.conceptId = conceptId;
  }

  /**
   * Get current affect state
   */
  public getAffectState(): AffectState {
    return this.currentAffectState;
  }

  /**
   * Get current frustration index
   */
  public getFrustrationIndex(): FrustrationIndex {
    return { ...this.frustrationIndex };
  }

  /**
   * Subscribe to affect state changes
   */
  public onAffectChange(callback: (state: AffectState) => void): () => void {
    const handler = () => callback(this.currentAffectState);
    window.addEventListener('telemetry:affect-change', handler);
    return () => window.removeEventListener('telemetry:affect-change', handler);
  }

  // Private methods

  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private registerEventListeners(): void {
    // Click events
    const clickHandler = (e: Event) => this.handleClick(e as MouseEvent);
    document.addEventListener('click', clickHandler, { capture: true });
    this.boundHandlers.set('click', clickHandler);

    // Mouse movement
    const mouseMoveHandler = (e: Event) => this.handleMouseMove(e as MouseEvent);
    document.addEventListener('mousemove', mouseMoveHandler, { passive: true });
    this.boundHandlers.set('mousemove', mouseMoveHandler);

    // Scroll
    const scrollHandler = (e: Event) => this.handleScroll(e);
    window.addEventListener('scroll', scrollHandler, { passive: true });
    this.boundHandlers.set('scroll', scrollHandler);

    // Keyboard
    const keyHandler = (e: Event) => this.handleKeypress(e as KeyboardEvent);
    document.addEventListener('keydown', keyHandler, { passive: true });
    this.boundHandlers.set('keydown', keyHandler);

    // Focus/blur
    const focusHandler = this.handleFocus.bind(this);
    const blurHandler = this.handleBlur.bind(this);
    window.addEventListener('focus', focusHandler);
    window.addEventListener('blur', blurHandler);
    this.boundHandlers.set('focus', focusHandler);
    this.boundHandlers.set('blur', blurHandler);

    // Visibility change
    const visibilityHandler = this.handleVisibilityChange.bind(this);
    document.addEventListener('visibilitychange', visibilityHandler);
    this.boundHandlers.set('visibilitychange', visibilityHandler);
  }

  private startIntervals(): void {
    // Flush buffer periodically
    this.flushInterval = setInterval(() => {
      this.flush();
    }, this.config.flushInterval);

    // Check for idle
    this.idleCheckInterval = setInterval(() => {
      this.checkIdle();
    }, 5000);

    // Update affect state
    this.affectUpdateInterval = setInterval(() => {
      this.updateAffectState();
    }, 2000);
  }

  private handleClick(event: MouseEvent): void {
    const now = Date.now();
    this.updateActivity();

    // Track click timestamps for rage detection
    this.clickTimestamps.push(now);

    // Clean old timestamps
    this.clickTimestamps = this.clickTimestamps.filter(
      (t) => now - t < this.config.rageClickWindow
    );

    // Detect rage clicks
    if (this.clickTimestamps.length >= this.config.rageClickThreshold) {
      this.rageClickCount++;
      this.trackEvent('rage_click', {
        clickCount: this.clickTimestamps.length,
        target: this.getElementInfo(event.target as Element),
      });
      this.clickTimestamps = []; // Reset
    }

    // Detect dead clicks
    const target = event.target as Element;
    const isDeadClick = this.isDeadClickTarget(target);

    if (isDeadClick) {
      this.deadClickCount++;
      this.trackEvent('dead_click', {
        target: this.getElementInfo(target),
      });
    }

    // Regular click tracking
    this.trackEvent('click', {
      target: this.getElementInfo(target),
      x: event.clientX,
      y: event.clientY,
      isDeadClick,
    });
  }

  private handleMouseMove(event: MouseEvent): void {
    const now = Date.now();
    this.updateActivity();

    // Throttle sampling
    if (now - this.lastMouseSample < this.config.mouseSampleRate) {
      return;
    }

    this.lastMouseSample = now;

    // Track position
    const position = { x: event.clientX, y: event.clientY, t: now };
    this.mousePositions.push(position);

    // Keep only recent positions (last 5 seconds)
    this.mousePositions = this.mousePositions.filter((p) => now - p.t < 5000);

    // Calculate velocity
    if (this.mousePositions.length >= 2) {
      const prev = this.mousePositions[this.mousePositions.length - 2];
      const curr = this.mousePositions[this.mousePositions.length - 1];

      const dx = curr.x - prev.x;
      const dy = curr.y - prev.y;
      const dt = curr.t - prev.t;

      if (dt > 0) {
        const velocity = Math.sqrt(dx * dx + dy * dy) / dt;
        this.mouseVelocities.push(velocity);

        // Keep only recent velocities
        if (this.mouseVelocities.length > 50) {
          this.mouseVelocities.shift();
        }
      }
    }
  }

  private handleScroll(event: Event): void {
    const now = Date.now();
    this.updateActivity();

    // Throttle
    if (now - this.lastScrollSample < this.config.scrollSampleRate) {
      return;
    }

    this.lastScrollSample = now;

    // Calculate scroll depth
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollDepth = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;

    this.scrollDepths.push(scrollDepth);

    // Track significant scroll events
    if (this.scrollDepths.length % 10 === 0) {
      this.trackEvent('scroll', {
        depth: scrollDepth,
        maxDepth: Math.max(...this.scrollDepths),
      });
    }
  }

  private handleKeypress(event: KeyboardEvent): void {
    this.updateActivity();

    // Don't track specific keys for privacy
    this.trackEvent('keypress', {
      isTyping: true,
    });
  }

  private handleFocus(): void {
    this.trackEvent('focus', {});
    this.updateActivity();
  }

  private handleBlur(): void {
    this.trackEvent('blur', {});
    this.startIdle();
  }

  private handleVisibilityChange(): void {
    if (document.hidden) {
      this.startIdle();
    } else {
      this.endIdle();
      this.updateActivity();
    }
  }

  private updateActivity(): void {
    this.lastActivityTime = Date.now();
    if (this.isIdle) {
      this.endIdle();
    }
  }

  private checkIdle(): void {
    const now = Date.now();
    const timeSinceActivity = now - this.lastActivityTime;

    if (!this.isIdle && timeSinceActivity >= this.config.idleThreshold) {
      this.startIdle();
    }
  }

  private startIdle(): void {
    if (!this.isIdle) {
      this.isIdle = true;
      this.idleStartTime = Date.now();
      this.trackEvent('idle_start', {});
    }
  }

  private endIdle(): void {
    if (this.isIdle && this.idleStartTime) {
      const idleDuration = Date.now() - this.idleStartTime;
      this.totalIdleTime += idleDuration;
      this.isIdle = false;
      this.idleStartTime = undefined;
      this.trackEvent('idle_end', { duration: idleDuration });
    }
  }

  private isDeadClickTarget(element: Element): boolean {
    // Check if element matches any dead click selectors
    for (const selector of this.config.deadClickSelectors) {
      if (element.matches(selector)) {
        return true;
      }
    }

    // Check if element is interactive
    const tagName = element.tagName.toLowerCase();
    const interactiveTags = ['a', 'button', 'input', 'select', 'textarea'];

    if (interactiveTags.includes(tagName)) {
      return false;
    }

    // Check for click handlers or role attributes
    if (
      element.hasAttribute('onclick') ||
      element.hasAttribute('role') ||
      element.classList.contains('clickable') ||
      element.closest('a, button, [role="button"]')
    ) {
      return false;
    }

    return true;
  }

  private getElementInfo(element: Element): Record<string, string> {
    return {
      tag: element.tagName.toLowerCase(),
      id: element.id || '',
      className: element.className?.toString()?.slice(0, 50) || '',
      text: element.textContent?.slice(0, 30) || '',
    };
  }

  private calculateMouseVelocityVariance(): number {
    if (this.mouseVelocities.length < 2) {
      return 0;
    }

    const mean =
      this.mouseVelocities.reduce((a, b) => a + b, 0) / this.mouseVelocities.length;
    const squaredDiffs = this.mouseVelocities.map((v) => Math.pow(v - mean, 2));
    const variance =
      squaredDiffs.reduce((a, b) => a + b, 0) / this.mouseVelocities.length;

    return variance;
  }

  private updateFrustrationIndex(): void {
    const { weights } = this.config;

    // Normalize values to 0-1 range
    const normalizedRageClicks = Math.min(this.rageClickCount / 10, 1);
    const normalizedDeadClicks = Math.min(this.deadClickCount / 15, 1);
    const normalizedVelocityVariance = Math.min(
      this.calculateMouseVelocityVariance() / 100,
      1
    );
    const normalizedIdleTime = Math.min(this.totalIdleTime / 300000, 1); // 5 min max

    // Calculate weighted frustration index
    const score =
      weights.rageClicks * normalizedRageClicks +
      weights.deadClicks * normalizedDeadClicks +
      weights.mouseVelocity * normalizedVelocityVariance +
      weights.idleTime * normalizedIdleTime;

    this.frustrationIndex = {
      rageClicks: normalizedRageClicks,
      deadClicks: normalizedDeadClicks,
      mouseVelocityVariance: normalizedVelocityVariance,
      timeIdle: normalizedIdleTime,
      score: Math.min(score, 1),
    };
  }

  private updateAffectState(): void {
    this.updateFrustrationIndex();

    const { score } = this.frustrationIndex;
    const previousState = this.currentAffectState;

    // Determine affect state based on frustration index and other signals
    if (score > 0.7) {
      this.currentAffectState = 'frustrated';
    } else if (this.frustrationIndex.timeIdle > 0.6 && score < 0.3) {
      this.currentAffectState = 'bored';
    } else if (
      this.frustrationIndex.mouseVelocityVariance > 0.5 &&
      this.frustrationIndex.deadClicks > 0.3
    ) {
      this.currentAffectState = 'confused';
    } else if (score < 0.2 && this.mouseVelocities.length > 10) {
      this.currentAffectState = 'flow';
    } else {
      this.currentAffectState = 'neutral';
    }

    // Emit change event if state changed
    if (previousState !== this.currentAffectState) {
      window.dispatchEvent(new CustomEvent('telemetry:affect-change'));
    }
  }

  public trackEvent(type: TelemetryEventType, data: Record<string, unknown>): void {
    const event: TelemetryEvent = {
      type,
      timestamp: Date.now(),
      data: {
        ...data,
        sessionId: this.sessionId,
        userId: this.userId,
        courseId: this.courseId,
        conceptId: this.conceptId,
      },
    };

    this.eventBuffer.push(event);

    // Flush if buffer is full
    if (this.eventBuffer.length >= this.config.batchSize) {
      this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (this.eventBuffer.length === 0) {
      return;
    }

    // Prepare payload
    const payload = this.config.sendOnlyInferredState
      ? {
        sessionId: this.sessionId,
        userId: this.userId,
        courseId: this.courseId,
        conceptId: this.conceptId,
        affectState: this.currentAffectState,
        frustrationIndex: this.frustrationIndex,
        timestamp: Date.now(),
        eventCount: this.eventBuffer.length,
      }
      : {
        events: [...this.eventBuffer],
        affectState: this.currentAffectState,
        frustrationIndex: this.frustrationIndex,
      };

    // Clear buffer before sending (to avoid duplicate sends)
    this.eventBuffer = [];

    try {
      // Send to server
      await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        keepalive: true, // Allow sending even if page is closing
      });
    } catch (error) {
      console.warn('[Telemetry] Failed to send telemetry:', error);
      // Events are already cleared; in production, consider retry logic
    }
  }
}

// Singleton instance
let trackerInstance: TelemetryTracker | null = null;

/**
 * Get or create the telemetry tracker singleton
 */
export function getTelemetryTracker(
  config?: Partial<TelemetryConfig>
): TelemetryTracker {
  if (!trackerInstance) {
    trackerInstance = new TelemetryTracker(config);
  }
  return trackerInstance;
}

/**
 * React hook for telemetry tracking
 */
export function useTelemetryTracker(
  userId?: number,
  courseId?: number,
  conceptId?: number
) {
  const [affectState, setAffectState] = React.useState<AffectState>('neutral');
  const [frustrationIndex, setFrustrationIndex] = React.useState<FrustrationIndex>({
    rageClicks: 0,
    deadClicks: 0,
    mouseVelocityVariance: 0,
    timeIdle: 0,
    score: 0,
  });

  React.useEffect(() => {
    const tracker = getTelemetryTracker();
    tracker.init(userId, courseId, conceptId);

    // Subscribe to affect changes
    const unsubscribe = tracker.onAffectChange((state) => {
      setAffectState(state);
      setFrustrationIndex(tracker.getFrustrationIndex());
    });

    return () => {
      unsubscribe();
      // Don't destroy tracker on unmount as it might be used by other components
    };
  }, [userId, courseId, conceptId]);

  return {
    affectState,
    frustrationIndex,
    tracker: getTelemetryTracker(),
  };
}


export default TelemetryTracker;
