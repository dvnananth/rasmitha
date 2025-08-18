from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud, schemas
from .db import get_session


router = APIRouter()


@router.post("/templates", response_model=schemas.TemplateRead)
async def create_template(payload: schemas.TemplateCreate, session: AsyncSession = Depends(get_session)):
    template = await crud.create_template(session, payload.model_dump())
    return template


@router.get("/templates", response_model=list[schemas.TemplateRead])
async def list_templates(session: AsyncSession = Depends(get_session)):
    templates = await crud.list_templates(session)
    return list(templates)


@router.post("/modules", response_model=schemas.ModuleInstanceRead)
async def create_module(payload: schemas.ModuleInstanceCreate, session: AsyncSession = Depends(get_session)):
    instance = await crud.create_module_instance(session, payload.model_dump())
    return instance


@router.get("/modules", response_model=list[schemas.ModuleInstanceRead])
async def list_modules(session: AsyncSession = Depends(get_session)):
    instances = await crud.list_module_instances(session)
    return list(instances)


@router.post("/views", response_model=schemas.ViewRead)
async def create_view(payload: schemas.ViewCreate, session: AsyncSession = Depends(get_session)):
    view = await crud.create_view(session, payload.model_dump())
    return view


@router.get("/views", response_model=list[schemas.ViewRead])
async def list_views(session: AsyncSession = Depends(get_session)):
    views = await crud.list_views(session)
    return list(views)


@router.post("/actions", response_model=schemas.ActionRead)
async def create_action(payload: schemas.ActionCreate, session: AsyncSession = Depends(get_session)):
    action = await crud.create_action(session, payload.model_dump())
    return action


@router.get("/actions", response_model=list[schemas.ActionRead])
async def list_actions(session: AsyncSession = Depends(get_session)):
    actions = await crud.list_actions(session)
    return list(actions)


@router.post("/connections", response_model=schemas.ConnectionRead)
async def create_connection(payload: schemas.ConnectionCreate, session: AsyncSession = Depends(get_session)):
    connection = await crud.create_connection(session, payload.model_dump())
    return connection


@router.get("/connections", response_model=list[schemas.ConnectionRead])
async def list_connections(session: AsyncSession = Depends(get_session)):
    connections = await crud.list_connections(session)
    return list(connections)

