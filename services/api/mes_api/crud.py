from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import models


# Template
async def create_template(session: AsyncSession, data: dict) -> models.Template:
    template = models.Template(**data)
    session.add(template)
    await session.commit()
    await session.refresh(template)
    return template


async def list_templates(session: AsyncSession) -> Sequence[models.Template]:
    result = await session.execute(select(models.Template))
    return result.scalars().all()


# ModuleInstance
async def create_module_instance(session: AsyncSession, data: dict) -> models.ModuleInstance:
    instance = models.ModuleInstance(**data)
    session.add(instance)
    await session.commit()
    await session.refresh(instance)
    return instance


async def list_module_instances(session: AsyncSession) -> Sequence[models.ModuleInstance]:
    result = await session.execute(select(models.ModuleInstance))
    return result.scalars().all()


# View
async def create_view(session: AsyncSession, data: dict) -> models.View:
    view = models.View(**data)
    session.add(view)
    await session.commit()
    await session.refresh(view)
    return view


async def list_views(session: AsyncSession) -> Sequence[models.View]:
    result = await session.execute(select(models.View))
    return result.scalars().all()


# Action
async def create_action(session: AsyncSession, data: dict) -> models.Action:
    action = models.Action(**data)
    session.add(action)
    await session.commit()
    await session.refresh(action)
    return action


async def list_actions(session: AsyncSession) -> Sequence[models.Action]:
    result = await session.execute(select(models.Action))
    return result.scalars().all()


# Connection
async def create_connection(session: AsyncSession, data: dict) -> models.Connection:
    connection = models.Connection(**data)
    session.add(connection)
    await session.commit()
    await session.refresh(connection)
    return connection


async def list_connections(session: AsyncSession) -> Sequence[models.Connection]:
    result = await session.execute(select(models.Connection))
    return result.scalars().all()

