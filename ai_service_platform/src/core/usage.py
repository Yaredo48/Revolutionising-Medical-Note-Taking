"""
Usage tracking service for AI Service Platform.
Records and aggregates usage metrics for billing and analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.db.database import UsageRecord
from src.models.schemas import UsageMetricType, ServiceType, UsageSummary


@dataclass
class UsageTracker:
    """Usage tracking service."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def record_usage(
        self,
        tenant_id: int,
        service_type: ServiceType,
        metric_type: UsageMetricType,
        value: int,
        user_id: Optional[int] = None,
        api_key_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """
        Record a usage event.
        
        Args:
            tenant_id: The tenant ID
            service_type: Type of AI service used
            metric_type: Type of metric (tokens, calls, etc.)
            value: The value to record
            user_id: Optional user ID
            api_key_id: Optional API key ID
            metadata: Additional metadata
        
        Returns:
            The created UsageRecord
        """
        record = UsageRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            api_key_id=api_key_id,
            service_type=service_type.value,
            metric_type=metric_type.value,
            value=value,
            metadata=metadata or {}
        )
        
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        
        return record
    
    async def record_api_call(
        self,
        tenant_id: int,
        service_type: ServiceType,
        user_id: Optional[int] = None,
        api_key_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """Record an API call."""
        return await self.record_usage(
            tenant_id=tenant_id,
            service_type=service_type,
            metric_type=UsageMetricType.API_CALLS,
            value=1,
            user_id=user_id,
            api_key_id=api_key_id,
            metadata=metadata
        )
    
    async def record_tokens(
        self,
        tenant_id: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        service_type: ServiceType = ServiceType.LLM,
        user_id: Optional[int] = None,
        api_key_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[UsageRecord]:
        """
        Record token usage for a request.
        
        Returns:
            List of created UsageRecords
        """
        records = []
        
        if prompt_tokens > 0:
            records.append(await self.record_usage(
                tenant_id=tenant_id,
                service_type=service_type,
                metric_type=UsageMetricType.TOKENS_PROMPT,
                value=prompt_tokens,
                user_id=user_id,
                api_key_id=api_key_id,
                metadata=metadata
            ))
        
        if completion_tokens > 0:
            records.append(await self.record_usage(
                tenant_id=tenant_id,
                service_type=service_type,
                metric_type=UsageMetricType.TOKENS_COMPLETION,
                value=completion_tokens,
                user_id=user_id,
                api_key_id=api_key_id,
                metadata=metadata
            ))
        
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            records.append(await self.record_usage(
                tenant_id=tenant_id,
                service_type=service_type,
                metric_type=UsageMetricType.TOKENS_TOTAL,
                value=total_tokens,
                user_id=user_id,
                api_key_id=api_key_id,
                metadata=metadata
            ))
        
        return records
    
    async def get_usage_summary(
        self,
        tenant_id: int,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> UsageSummary:
        """
        Get usage summary for a tenant within a date range.
        
        Args:
            tenant_id: The tenant ID
            start_date: Start of the period (defaults to 30 days ago)
            end_date: End of the period (defaults to now)
        
        Returns:
            UsageSummary with aggregated metrics
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Query for aggregated metrics
        result = await self.db.execute(
            select(
                func.sum(UsageRecord.value),
                func.count(UsageRecord.id)
            ).where(
                and_(
                    UsageRecord.tenant_id == tenant_id,
                    UsageRecord.timestamp >= start_date,
                    UsageRecord.timestamp <= end_date
                )
            )
        )
        
        row = result.one()
        total_value = row[0] or 0
        total_calls = row[1] or 0
        
        # Get breakdown by service and metric
        breakdown_result = await self.db.execute(
            select(
                UsageRecord.service_type,
                UsageRecord.metric_type,
                func.sum(UsageRecord.value),
                func.count(UsageRecord.id)
            ).where(
                and_(
                    UsageRecord.tenant_id == tenant_id,
                    UsageRecord.timestamp >= start_date,
                    UsageRecord.timestamp <= end_date
                )
            ).group_by(
                UsageRecord.service_type,
                UsageRecord.metric_type
            )
        )
        
        breakdown = {}
        for service, metric, value, count in breakdown_result.all():
            if service not in breakdown:
                breakdown[service] = {}
            breakdown[service][metric] = {
                "total": value,
                "count": count
            }
        
        return UsageSummary(
            tenant_id=tenant_id,
            period_start=start_date,
            period_end=end_date,
            total_api_calls=total_calls,
            total_tokens=total_value,
            breakdown=breakdown
        )
    
    async def get_daily_usage(
        self,
        tenant_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get daily usage breakdown for a tenant.
        
        Returns:
            List of daily usage records
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.db.execute(
            select(
                func.date(UsageRecord.timestamp).label('date'),
                UsageRecord.service_type,
                UsageRecord.metric_type,
                func.sum(UsageRecord.value),
                func.count(UsageRecord.id)
            ).where(
                and_(
                    UsageRecord.tenant_id == tenant_id,
                    UsageRecord.timestamp >= start_date
                )
            ).group_by(
                func.date(UsageRecord.timestamp),
                UsageRecord.service_type,
                UsageRecord.metric_type
            ).order_by(
                func.date(UsageRecord.timestamp)
            )
        )
        
        daily_usage = []
        for row in result.all():
            daily_usage.append({
                "date": row[0],
                "service_type": row[1],
                "metric_type": row[2],
                "total_value": row[3],
                "request_count": row[4]
            })
        
        return daily_usage
    
    async def get_usage_by_api_key(
        self,
        tenant_id: int,
        api_key_id: int,
        days: int = 30
    ) -> UsageSummary:
        """Get usage summary for a specific API key."""
        start_date = datetime.utcnow() - timedelta(days=days)
        end_date = datetime.utcnow()
        
        result = await self.db.execute(
            select(
                func.sum(UsageRecord.value),
                func.count(UsageRecord.id)
            ).where(
                and_(
                    UsageRecord.tenant_id == tenant_id,
                    UsageRecord.api_key_id == api_key_id,
                    UsageRecord.timestamp >= start_date,
                    UsageRecord.timestamp <= end_date
                )
            )
        )
        
        row = result.one()
        
        return UsageSummary(
            tenant_id=tenant_id,
            period_start=start_date,
            period_end=end_date,
            total_api_calls=row[1] or 0,
            total_tokens=row[0] or 0,
            breakdown={}
        )
