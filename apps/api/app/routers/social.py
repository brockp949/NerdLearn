"""
Social Gamification API Router

Endpoints for friends, challenges, study groups, and leaderboards.
Enables social learning features and competitive elements.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func, desc
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import secrets

from app.core.database import get_db
from app.models.social import (
    Friendship,
    FriendshipStatus,
    Challenge,
    ChallengeType,
    ChallengeStatus,
    ChallengeParticipant,
    StudyGroup,
    GroupMessage,
    GroupRole,
    Leaderboard,
    UserActivity,
    study_group_members,
)
from app.models.user import User

from app.services.social_agent_service import SocialAgentService
from app.schemas.social_agent import (
    CodingChallenge, EvaluationResult, EvaluationRequest, HintRequest, HintResponse,
    StartDebateRequest, AdvanceDebateRequest, DebateSessionResponse, DebateRoundResponse, DebateSummary,
    TeachingSessionStartRequest, ExplanationRequest, TeachingResponse, TeachingSessionSummary, TeachingSessionResponse
)

agent_service = SocialAgentService()

router = APIRouter()


# ==================== Schemas ====================

class FriendRequestCreate(BaseModel):
    addressee_id: int


class FriendRequestResponse(BaseModel):
    id: int
    requester_id: int
    addressee_id: int
    status: str
    created_at: datetime


class FriendResponse(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    total_xp: int
    level: int
    streak_days: int


class ChallengeCreate(BaseModel):
    challenge_type: ChallengeType
    title: str
    description: Optional[str]
    target_value: int
    course_id: Optional[int]
    end_date: datetime
    participant_ids: List[int]
    xp_reward: int = 100


class ChallengeResponse(BaseModel):
    id: int
    challenge_type: str
    title: str
    description: Optional[str]
    target_value: int
    status: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    xp_reward: int
    creator_id: int
    winner_id: Optional[int]
    participants: List[dict]


class StudyGroupCreate(BaseModel):
    name: str
    description: Optional[str]
    course_id: Optional[int]
    is_public: bool = False
    max_members: int = 50


class StudyGroupResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    is_public: bool
    max_members: int
    member_count: int
    total_xp: int
    owner_id: int
    invite_code: Optional[str]


class GroupMessageCreate(BaseModel):
    content: str
    shared_module_id: Optional[int]


class GroupMessageResponse(BaseModel):
    id: int
    user_id: int
    username: str
    content: str
    created_at: datetime
    shared_module_id: Optional[int]


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: int
    username: str
    score: int
    level: int


# ==================== Friends Endpoints ====================

@router.post("/friends/request", response_model=FriendRequestResponse)
async def send_friend_request(
    request: FriendRequestCreate,
    current_user_id: int = Query(..., description="Current user ID"),
    db: AsyncSession = Depends(get_db),
):
    """Send a friend request to another user."""
    if current_user_id == request.addressee_id:
        raise HTTPException(status_code=400, detail="Cannot send friend request to yourself")

    # Check if addressee exists
    result = await db.execute(select(User).where(User.id == request.addressee_id))
    addressee = result.scalar_one_or_none()
    if not addressee:
        raise HTTPException(status_code=404, detail="User not found")

    # Check for existing friendship
    existing = await db.execute(
        select(Friendship).where(
            or_(
                and_(
                    Friendship.requester_id == current_user_id,
                    Friendship.addressee_id == request.addressee_id,
                ),
                and_(
                    Friendship.requester_id == request.addressee_id,
                    Friendship.addressee_id == current_user_id,
                ),
            )
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Friend request already exists")

    # Create friendship request
    friendship = Friendship(
        requester_id=current_user_id,
        addressee_id=request.addressee_id,
        status=FriendshipStatus.PENDING,
    )
    db.add(friendship)
    await db.commit()
    await db.refresh(friendship)

    return FriendRequestResponse(
        id=friendship.id,
        requester_id=friendship.requester_id,
        addressee_id=friendship.addressee_id,
        status=friendship.status.value,
        created_at=friendship.created_at,
    )


@router.get("/friends/requests", response_model=List[FriendRequestResponse])
async def get_friend_requests(
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get pending friend requests for the current user."""
    result = await db.execute(
        select(Friendship).where(
            Friendship.addressee_id == current_user_id,
            Friendship.status == FriendshipStatus.PENDING,
        )
    )
    requests = result.scalars().all()

    return [
        FriendRequestResponse(
            id=r.id,
            requester_id=r.requester_id,
            addressee_id=r.addressee_id,
            status=r.status.value,
            created_at=r.created_at,
        )
        for r in requests
    ]


@router.post("/friends/requests/{request_id}/accept")
async def accept_friend_request(
    request_id: int,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Accept a friend request."""
    result = await db.execute(
        select(Friendship).where(
            Friendship.id == request_id,
            Friendship.addressee_id == current_user_id,
            Friendship.status == FriendshipStatus.PENDING,
        )
    )
    friendship = result.scalar_one_or_none()

    if not friendship:
        raise HTTPException(status_code=404, detail="Friend request not found")

    friendship.status = FriendshipStatus.ACCEPTED
    await db.commit()

    return {"message": "Friend request accepted"}


@router.post("/friends/requests/{request_id}/decline")
async def decline_friend_request(
    request_id: int,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Decline a friend request."""
    result = await db.execute(
        select(Friendship).where(
            Friendship.id == request_id,
            Friendship.addressee_id == current_user_id,
            Friendship.status == FriendshipStatus.PENDING,
        )
    )
    friendship = result.scalar_one_or_none()

    if not friendship:
        raise HTTPException(status_code=404, detail="Friend request not found")

    friendship.status = FriendshipStatus.DECLINED
    await db.commit()

    return {"message": "Friend request declined"}


@router.get("/friends", response_model=List[FriendResponse])
async def get_friends(
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get the user's friends list."""
    result = await db.execute(
        select(Friendship).where(
            or_(
                Friendship.requester_id == current_user_id,
                Friendship.addressee_id == current_user_id,
            ),
            Friendship.status == FriendshipStatus.ACCEPTED,
        )
    )
    friendships = result.scalars().all()

    friend_ids = set()
    for f in friendships:
        if f.requester_id == current_user_id:
            friend_ids.add(f.addressee_id)
        else:
            friend_ids.add(f.requester_id)

    if not friend_ids:
        return []

    friends_result = await db.execute(
        select(User).where(User.id.in_(friend_ids))
    )
    friends = friends_result.scalars().all()

    return [
        FriendResponse(
            id=f.id,
            username=f.username,
            full_name=f.full_name,
            total_xp=f.total_xp or 0,
            level=f.level or 1,
            streak_days=f.streak_days or 0,
        )
        for f in friends
    ]


@router.delete("/friends/{friend_id}")
async def remove_friend(
    friend_id: int,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Remove a friend."""
    result = await db.execute(
        select(Friendship).where(
            or_(
                and_(
                    Friendship.requester_id == current_user_id,
                    Friendship.addressee_id == friend_id,
                ),
                and_(
                    Friendship.requester_id == friend_id,
                    Friendship.addressee_id == current_user_id,
                ),
            ),
            Friendship.status == FriendshipStatus.ACCEPTED,
        )
    )
    friendship = result.scalar_one_or_none()

    if not friendship:
        raise HTTPException(status_code=404, detail="Friendship not found")

    await db.delete(friendship)
    await db.commit()

    return {"message": "Friend removed"}


# ==================== Challenges Endpoints ====================

@router.post("/challenges", response_model=ChallengeResponse)
async def create_challenge(
    challenge: ChallengeCreate,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Create a new challenge and invite participants."""
    # Create challenge
    new_challenge = Challenge(
        challenge_type=challenge.challenge_type,
        title=challenge.title,
        description=challenge.description,
        target_value=challenge.target_value,
        course_id=challenge.course_id,
        end_date=challenge.end_date,
        start_date=datetime.utcnow(),
        creator_id=current_user_id,
        xp_reward=challenge.xp_reward,
        status=ChallengeStatus.PENDING,
    )
    db.add(new_challenge)
    await db.flush()

    # Add creator as participant (auto-accepted)
    creator_participant = ChallengeParticipant(
        challenge_id=new_challenge.id,
        user_id=current_user_id,
        accepted=True,
    )
    db.add(creator_participant)

    # Add invited participants
    for participant_id in challenge.participant_ids:
        if participant_id != current_user_id:
            participant = ChallengeParticipant(
                challenge_id=new_challenge.id,
                user_id=participant_id,
                accepted=False,
            )
            db.add(participant)

    await db.commit()
    await db.refresh(new_challenge)

    # Get participants
    participants_result = await db.execute(
        select(ChallengeParticipant).where(
            ChallengeParticipant.challenge_id == new_challenge.id
        )
    )
    participants = participants_result.scalars().all()

    return ChallengeResponse(
        id=new_challenge.id,
        challenge_type=new_challenge.challenge_type.value,
        title=new_challenge.title,
        description=new_challenge.description,
        target_value=new_challenge.target_value,
        status=new_challenge.status.value,
        start_date=new_challenge.start_date,
        end_date=new_challenge.end_date,
        xp_reward=new_challenge.xp_reward,
        creator_id=new_challenge.creator_id,
        winner_id=new_challenge.winner_id,
        participants=[
            {
                "user_id": p.user_id,
                "accepted": p.accepted,
                "current_value": p.current_value,
                "completed": p.completed,
            }
            for p in participants
        ],
    )


@router.get("/challenges", response_model=List[ChallengeResponse])
async def get_challenges(
    current_user_id: int = Query(...),
    status_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get challenges for the current user."""
    # Get challenges where user is a participant
    subquery = (
        select(ChallengeParticipant.challenge_id)
        .where(ChallengeParticipant.user_id == current_user_id)
    )

    query = select(Challenge).where(Challenge.id.in_(subquery))

    if status_filter:
        query = query.where(Challenge.status == ChallengeStatus(status_filter))

    result = await db.execute(query.options(selectinload(Challenge.participants)))
    challenges = result.scalars().all()

    return [
        ChallengeResponse(
            id=c.id,
            challenge_type=c.challenge_type.value,
            title=c.title,
            description=c.description,
            target_value=c.target_value,
            status=c.status.value,
            start_date=c.start_date,
            end_date=c.end_date,
            xp_reward=c.xp_reward,
            creator_id=c.creator_id,
            winner_id=c.winner_id,
            participants=[
                {
                    "user_id": p.user_id,
                    "accepted": p.accepted,
                    "current_value": p.current_value,
                    "completed": p.completed,
                }
                for p in c.participants
            ],
        )
        for c in challenges
    ]


@router.post("/challenges/{challenge_id}/accept")
async def accept_challenge(
    challenge_id: int,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Accept a challenge invitation."""
    result = await db.execute(
        select(ChallengeParticipant).where(
            ChallengeParticipant.challenge_id == challenge_id,
            ChallengeParticipant.user_id == current_user_id,
            ChallengeParticipant.accepted == False,
        )
    )
    participant = result.scalar_one_or_none()

    if not participant:
        raise HTTPException(status_code=404, detail="Challenge invitation not found")

    participant.accepted = True

    # Check if all participants have accepted
    all_participants_result = await db.execute(
        select(ChallengeParticipant).where(
            ChallengeParticipant.challenge_id == challenge_id
        )
    )
    all_participants = all_participants_result.scalars().all()

    all_accepted = all(p.accepted for p in all_participants)

    if all_accepted:
        # Activate challenge
        challenge_result = await db.execute(
            select(Challenge).where(Challenge.id == challenge_id)
        )
        challenge = challenge_result.scalar_one()
        challenge.status = ChallengeStatus.ACTIVE
        challenge.start_date = datetime.utcnow()

    await db.commit()

    return {"message": "Challenge accepted", "status": "active" if all_accepted else "pending"}


@router.post("/challenges/{challenge_id}/progress")
async def update_challenge_progress(
    challenge_id: int,
    progress_value: int = Query(...),
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Update progress in a challenge."""
    result = await db.execute(
        select(ChallengeParticipant).where(
            ChallengeParticipant.challenge_id == challenge_id,
            ChallengeParticipant.user_id == current_user_id,
            ChallengeParticipant.accepted == True,
        )
    )
    participant = result.scalar_one_or_none()

    if not participant:
        raise HTTPException(status_code=404, detail="Not a participant in this challenge")

    # Get challenge
    challenge_result = await db.execute(
        select(Challenge).where(
            Challenge.id == challenge_id,
            Challenge.status == ChallengeStatus.ACTIVE,
        )
    )
    challenge = challenge_result.scalar_one_or_none()

    if not challenge:
        raise HTTPException(status_code=404, detail="Active challenge not found")

    participant.current_value = progress_value

    # Check if target reached
    if progress_value >= challenge.target_value and not participant.completed:
        participant.completed = True
        participant.completed_at = datetime.utcnow()

        # First to complete wins (for racing challenges)
        if not challenge.winner_id:
            challenge.winner_id = current_user_id
            challenge.status = ChallengeStatus.COMPLETED

            # Award XP to winner
            winner_result = await db.execute(
                select(User).where(User.id == current_user_id)
            )
            winner = winner_result.scalar_one()
            winner.total_xp = (winner.total_xp or 0) + challenge.xp_reward

    await db.commit()

    return {
        "current_value": participant.current_value,
        "completed": participant.completed,
        "challenge_status": challenge.status.value,
    }


# ==================== Study Groups Endpoints ====================

@router.post("/groups", response_model=StudyGroupResponse)
async def create_study_group(
    group: StudyGroupCreate,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Create a new study group."""
    # Generate invite code for private groups
    invite_code = None
    if not group.is_public:
        invite_code = secrets.token_urlsafe(8)

    new_group = StudyGroup(
        name=group.name,
        description=group.description,
        course_id=group.course_id,
        is_public=group.is_public,
        max_members=group.max_members,
        owner_id=current_user_id,
        invite_code=invite_code,
        member_count=1,
    )
    db.add(new_group)
    await db.flush()

    # Add owner as member
    await db.execute(
        study_group_members.insert().values(
            group_id=new_group.id,
            user_id=current_user_id,
            role=GroupRole.OWNER,
        )
    )

    await db.commit()
    await db.refresh(new_group)

    return StudyGroupResponse(
        id=new_group.id,
        name=new_group.name,
        description=new_group.description,
        is_public=new_group.is_public,
        max_members=new_group.max_members,
        member_count=new_group.member_count,
        total_xp=new_group.total_xp,
        owner_id=new_group.owner_id,
        invite_code=new_group.invite_code,
    )


@router.get("/groups", response_model=List[StudyGroupResponse])
async def get_study_groups(
    current_user_id: int = Query(...),
    include_public: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    """Get study groups the user is a member of, optionally including public groups."""
    # Get user's groups
    subquery = (
        select(study_group_members.c.group_id)
        .where(study_group_members.c.user_id == current_user_id)
    )

    if include_public:
        query = select(StudyGroup).where(
            or_(
                StudyGroup.id.in_(subquery),
                StudyGroup.is_public == True,
            )
        )
    else:
        query = select(StudyGroup).where(StudyGroup.id.in_(subquery))

    result = await db.execute(query)
    groups = result.scalars().all()

    return [
        StudyGroupResponse(
            id=g.id,
            name=g.name,
            description=g.description,
            is_public=g.is_public,
            max_members=g.max_members,
            member_count=g.member_count,
            total_xp=g.total_xp,
            owner_id=g.owner_id,
            invite_code=g.invite_code if g.owner_id == current_user_id else None,
        )
        for g in groups
    ]


@router.post("/groups/{group_id}/join")
async def join_study_group(
    group_id: int,
    invite_code: Optional[str] = None,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Join a study group (public or with invite code)."""
    result = await db.execute(
        select(StudyGroup).where(StudyGroup.id == group_id)
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(status_code=404, detail="Study group not found")

    # Check if already a member
    member_check = await db.execute(
        select(study_group_members).where(
            study_group_members.c.group_id == group_id,
            study_group_members.c.user_id == current_user_id,
        )
    )
    if member_check.first():
        raise HTTPException(status_code=400, detail="Already a member")

    # Check capacity
    if group.member_count >= group.max_members:
        raise HTTPException(status_code=400, detail="Group is full")

    # Check access
    if not group.is_public:
        if not invite_code or invite_code != group.invite_code:
            raise HTTPException(status_code=403, detail="Invalid invite code")

    # Add member
    await db.execute(
        study_group_members.insert().values(
            group_id=group_id,
            user_id=current_user_id,
            role=GroupRole.MEMBER,
        )
    )
    group.member_count += 1

    await db.commit()

    return {"message": "Joined study group successfully"}


@router.delete("/groups/{group_id}/leave")
async def leave_study_group(
    group_id: int,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Leave a study group."""
    result = await db.execute(
        select(StudyGroup).where(StudyGroup.id == group_id)
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(status_code=404, detail="Study group not found")

    if group.owner_id == current_user_id:
        raise HTTPException(status_code=400, detail="Owner cannot leave. Transfer ownership or delete the group.")

    await db.execute(
        study_group_members.delete().where(
            study_group_members.c.group_id == group_id,
            study_group_members.c.user_id == current_user_id,
        )
    )
    group.member_count -= 1

    await db.commit()

    return {"message": "Left study group"}


@router.get("/groups/{group_id}/messages", response_model=List[GroupMessageResponse])
async def get_group_messages(
    group_id: int,
    current_user_id: int = Query(...),
    limit: int = Query(default=50, le=100),
    before_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get messages from a study group."""
    # Verify membership
    member_check = await db.execute(
        select(study_group_members).where(
            study_group_members.c.group_id == group_id,
            study_group_members.c.user_id == current_user_id,
        )
    )
    if not member_check.first():
        raise HTTPException(status_code=403, detail="Not a member of this group")

    query = (
        select(GroupMessage, User.username)
        .join(User, GroupMessage.user_id == User.id)
        .where(GroupMessage.group_id == group_id)
    )

    if before_id:
        query = query.where(GroupMessage.id < before_id)

    query = query.order_by(desc(GroupMessage.id)).limit(limit)

    result = await db.execute(query)
    messages = result.all()

    return [
        GroupMessageResponse(
            id=m.GroupMessage.id,
            user_id=m.GroupMessage.user_id,
            username=m.username,
            content=m.GroupMessage.content,
            created_at=m.GroupMessage.created_at,
            shared_module_id=m.GroupMessage.shared_module_id,
        )
        for m in reversed(messages)
    ]


@router.post("/groups/{group_id}/messages", response_model=GroupMessageResponse)
async def send_group_message(
    group_id: int,
    message: GroupMessageCreate,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Send a message to a study group."""
    # Verify membership
    member_check = await db.execute(
        select(study_group_members).where(
            study_group_members.c.group_id == group_id,
            study_group_members.c.user_id == current_user_id,
        )
    )
    if not member_check.first():
        raise HTTPException(status_code=403, detail="Not a member of this group")

    # Get username
    user_result = await db.execute(select(User).where(User.id == current_user_id))
    user = user_result.scalar_one()

    new_message = GroupMessage(
        group_id=group_id,
        user_id=current_user_id,
        content=message.content,
        shared_module_id=message.shared_module_id,
    )
    db.add(new_message)
    await db.commit()
    await db.refresh(new_message)

    return GroupMessageResponse(
        id=new_message.id,
        user_id=new_message.user_id,
        username=user.username,
        content=new_message.content,
        created_at=new_message.created_at,
        shared_module_id=new_message.shared_module_id,
    )


# ==================== Leaderboard Endpoints ====================

@router.get("/leaderboard/global", response_model=List[LeaderboardEntry])
async def get_global_leaderboard(
    period: str = Query(default="weekly", pattern="^(daily|weekly|monthly|all_time)$"),
    limit: int = Query(default=10, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Get global leaderboard rankings."""
    result = await db.execute(
        select(User)
        .where(User.is_active == True)
        .order_by(desc(User.total_xp))
        .limit(limit)
    )
    users = result.scalars().all()

    return [
        LeaderboardEntry(
            rank=i + 1,
            user_id=u.id,
            username=u.username,
            score=u.total_xp or 0,
            level=u.level or 1,
        )
        for i, u in enumerate(users)
    ]


@router.get("/leaderboard/friends", response_model=List[LeaderboardEntry])
async def get_friends_leaderboard(
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get leaderboard among friends."""
    # Get friends
    friendships_result = await db.execute(
        select(Friendship).where(
            or_(
                Friendship.requester_id == current_user_id,
                Friendship.addressee_id == current_user_id,
            ),
            Friendship.status == FriendshipStatus.ACCEPTED,
        )
    )
    friendships = friendships_result.scalars().all()

    friend_ids = {current_user_id}  # Include self
    for f in friendships:
        if f.requester_id == current_user_id:
            friend_ids.add(f.addressee_id)
        else:
            friend_ids.add(f.requester_id)

    # Get users and rank
    result = await db.execute(
        select(User)
        .where(User.id.in_(friend_ids))
        .order_by(desc(User.total_xp))
    )
    users = result.scalars().all()

    return [
        LeaderboardEntry(
            rank=i + 1,
            user_id=u.id,
            username=u.username,
            score=u.total_xp or 0,
            level=u.level or 1,
        )
        for i, u in enumerate(users)
    ]


@router.get("/leaderboard/group/{group_id}", response_model=List[LeaderboardEntry])
async def get_group_leaderboard(
    group_id: int,
    current_user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get leaderboard for a study group."""
    # Verify membership
    member_check = await db.execute(
        select(study_group_members).where(
            study_group_members.c.group_id == group_id,
            study_group_members.c.user_id == current_user_id,
        )
    )
    if not member_check.first():
        raise HTTPException(status_code=403, detail="Not a member of this group")

    # Get group members
    members_result = await db.execute(
        select(study_group_members.c.user_id).where(
            study_group_members.c.group_id == group_id
        )
    )
    member_ids = [row[0] for row in members_result]

    # Get users and rank
    result = await db.execute(
        select(User)
        .where(User.id.in_(member_ids))
        .order_by(desc(User.total_xp))
    )
    users = result.scalars().all()

    return [
        LeaderboardEntry(
            rank=i + 1,
            user_id=u.id,
            username=u.username,
            score=u.total_xp or 0,
            level=u.level or 1,
        )
        for i, u in enumerate(users)
    ]


# ==================== Activity Feed ====================

@router.get("/activity/friends")
async def get_friends_activity(
    current_user_id: int = Query(...),
    limit: int = Query(default=20, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Get activity feed from friends."""
    # Get friends
    friendships_result = await db.execute(
        select(Friendship).where(
            or_(
                Friendship.requester_id == current_user_id,
                Friendship.addressee_id == current_user_id,
            ),
            Friendship.status == FriendshipStatus.ACCEPTED,
        )
    )
    friendships = friendships_result.scalars().all()

    friend_ids = set()
    for f in friendships:
        if f.requester_id == current_user_id:
            friend_ids.add(f.addressee_id)
        else:
            friend_ids.add(f.requester_id)

    if not friend_ids:
        return []

    # Get recent activities
    result = await db.execute(
        select(UserActivity, User.username)
        .join(User, UserActivity.user_id == User.id)
        .where(
            UserActivity.user_id.in_(friend_ids),
            UserActivity.is_public == True,
        )
        .order_by(desc(UserActivity.created_at))
        .limit(limit)
    )
    activities = result.all()

    return [
        {
            "id": a.UserActivity.id,
            "user_id": a.UserActivity.user_id,
            "username": a.username,
            "activity_type": a.UserActivity.activity_type,
            "title": a.UserActivity.title,
            "description": a.UserActivity.description,
            "created_at": a.UserActivity.created_at.isoformat(),
        }
        for a in activities
    ]


# ==================== Agentic Social Features ====================

# --- Coding Challenges ---

@router.post("/challenges/init")
async def init_sample_challenges(
    current_user_id: int = Query(...),
):
    """Initialize sample coding challenges (mock)."""
    return {"message": "Sample challenges initialized"}

@router.get("/coding-challenges", response_model=Dict[str, List[CodingChallenge]])
async def list_coding_challenges(
    current_user_id: int = Query(...),
):
    """List available AI coding challenges."""
    challenges = await agent_service.get_challenges()
    return {"challenges": challenges}

@router.get("/coding-challenges/{challenge_id}", response_model=CodingChallenge)
async def get_coding_challenge(
    challenge_id: str,
    current_user_id: int = Query(...),
):
    """Get details for a specific coding challenge."""
    challenge = await agent_service.get_challenge(challenge_id)
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge

@router.post("/challenges/evaluate", response_model=EvaluationResult)
async def evaluate_code_submission(
    request: EvaluationRequest,
):
    """Evaluate code submission using AI agent."""
    return await agent_service.evaluate_code(request.challenge_id, request.code)

@router.post("/challenges/hint", response_model=HintResponse)
async def get_challenge_hint(
    request: HintRequest,
):
    """Get a contextual hint for the current code."""
    hint_text = await agent_service.get_hint(request.challenge_id, request.code, request.hint_level)
    return HintResponse(
        hint_level=request.hint_level,
        hint=hint_text,
        cost=10 # Mock cost
    )

# --- Debates ---

@router.post("/debates/start", response_model=DebateSessionResponse)
async def start_debate(
    request: StartDebateRequest,
):
    """Start a new multi-agent debate session."""
    return await agent_service.start_debate(
        request.topic, 
        request.format, 
        request.panel_preset, 
        request.max_rounds
    )

@router.post("/debates/advance", response_model=DebateRoundResponse)
async def advance_debate(
    request: AdvanceDebateRequest,
):
    """Advance the debate to the next round."""
    return await agent_service.advance_debate(request.session_id, request.learner_contribution)

@router.get("/debates/{session_id}/summary", response_model=DebateSummary)
async def get_debate_summary(
    session_id: str,
):
    """Get the summary of a completed debate."""
    return await agent_service.get_debate_summary(session_id)

# --- Teaching ---

@router.post("/teaching/start", response_model=TeachingSessionResponse)
async def start_teaching_session(
    request: TeachingSessionStartRequest,
):
    """Start a new teaching session (Feynman Technique)."""
    return await agent_service.start_teaching_session(
        request.user_id, 
        request.concept_name, 
        request.persona
    )

@router.post("/teaching/explain", response_model=TeachingResponse)
async def submit_explanation(
    request: ExplanationRequest,
):
    """Submit an explanation to the student agent."""
    return await agent_service.submit_explanation(request.session_id, request.explanation)

@router.post("/teaching/{session_id}/end", response_model=TeachingSessionSummary)
async def end_teaching_session(
    session_id: str,
):
    """End the teaching session and get feedback."""
    return await agent_service.end_teaching_session(session_id)
