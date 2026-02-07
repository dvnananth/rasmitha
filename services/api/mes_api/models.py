from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Template(Base):
    __tablename__ = "templates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="0.1.0")
    kind: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., module, view, action
    schema: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class ModuleInstance(Base):
    __tablename__ = "module_instances"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    template_id: Mapped[int] = mapped_column(ForeignKey("templates.id", ondelete="RESTRICT"), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    template: Mapped[Template] = relationship(backref="module_instances")


class View(Base):
    __tablename__ = "views"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    layout: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    input_schema: Mapped[dict] = mapped_column(JSON, nullable=False, default={})
    output_schema: Mapped[dict] = mapped_column(JSON, nullable=False, default={})
    code: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class Connection(Base):
    __tablename__ = "connections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_module_id: Mapped[int] = mapped_column(ForeignKey("module_instances.id", ondelete="CASCADE"), nullable=False)
    source_port: Mapped[str] = mapped_column(String(80), nullable=False)
    target_module_id: Mapped[int] = mapped_column(ForeignKey("module_instances.id", ondelete="CASCADE"), nullable=False)
    target_port: Mapped[str] = mapped_column(String(80), nullable=False)
    transform: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    source_module: Mapped[ModuleInstance] = relationship(foreign_keys=[source_module_id])
    target_module: Mapped[ModuleInstance] = relationship(foreign_keys=[target_module_id])

