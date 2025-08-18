from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TemplateCreate(BaseModel):
    name: str
    version: str = "0.1.0"
    kind: str
    schema: dict[str, Any]


class TemplateRead(BaseModel):
    id: int
    name: str
    version: str
    kind: str
    schema: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModuleInstanceCreate(BaseModel):
    name: str
    template_id: int
    config: dict[str, Any] = Field(default_factory=dict)


class ModuleInstanceRead(BaseModel):
    id: int
    name: str
    template_id: int
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ViewCreate(BaseModel):
    name: str
    layout: dict[str, Any]


class ViewRead(BaseModel):
    id: int
    name: str
    layout: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ActionCreate(BaseModel):
    name: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    code: str = ""


class ActionRead(BaseModel):
    id: int
    name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    code: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConnectionCreate(BaseModel):
    source_module_id: int
    source_port: str
    target_module_id: int
    target_port: str
    transform: Optional[dict[str, Any]] = None


class ConnectionRead(BaseModel):
    id: int
    source_module_id: int
    source_port: str
    target_module_id: int
    target_port: str
    transform: Optional[dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True

