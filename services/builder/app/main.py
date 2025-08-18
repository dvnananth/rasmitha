from __future__ import annotations

import json
from typing import Any

import streamlit as st

from .client import ApiClient


st.set_page_config(page_title="MES Builder", layout="wide")
st.title("Composable MES – Builder")

api = ApiClient()


tab_templates, tab_modules, tab_views, tab_actions, tab_connections = st.tabs(
    ["Templates", "Modules", "Views", "Actions", "Connections"]
)


with tab_templates:
    st.subheader("Templates")
    with st.form("create_template"):
        name = st.text_input("Name")
        version = st.text_input("Version", value="0.1.0")
        kind = st.selectbox("Kind", ["module", "view", "action"], index=0)
        schema_text = st.text_area("Schema (JSON)", value="{}", height=160)
        submitted = st.form_submit_button("Create Template")
        if submitted:
            try:
                payload = {
                    "name": name,
                    "version": version,
                    "kind": kind,
                    "schema": json.loads(schema_text or "{}"),
                }
                created = api.create_template(payload)
                st.success(f"Created template #{created['id']}")
            except Exception as e:
                st.error(str(e))

    st.write("Existing templates")
    st.json(api.list_templates())


with tab_modules:
    st.subheader("Modules")
    templates = api.list_templates()
    template_options = {f"{t['name']} ({t['id']})": t["id"] for t in templates if t["kind"] == "module"}
    with st.form("create_module"):
        name = st.text_input("Name", key="mod_name")
        template_label = st.selectbox("Template", list(template_options.keys()))
        config_text = st.text_area("Config (JSON)", value="{}", height=160)
        submitted = st.form_submit_button("Create Module")
        if submitted:
            try:
                payload = {
                    "name": name,
                    "template_id": template_options[template_label],
                    "config": json.loads(config_text or "{}"),
                }
                created = api.create_module(payload)
                st.success(f"Created module #{created['id']}")
            except Exception as e:
                st.error(str(e))

    st.write("Existing modules")
    st.json(api.list_modules())


with tab_views:
    st.subheader("Views")
    with st.form("create_view"):
        name = st.text_input("Name", key="view_name")
        layout_text = st.text_area("Layout (JSON)", value="{\n  \"type\": \"container\",\n  \"children\": []\n}", height=160)
        submitted = st.form_submit_button("Create View")
        if submitted:
            try:
                payload = {"name": name, "layout": json.loads(layout_text)}
                created = api.create_view(payload)
                st.success(f"Created view #{created['id']}")
            except Exception as e:
                st.error(str(e))

    st.write("Existing views")
    st.json(api.list_views())


with tab_actions:
    st.subheader("Actions")
    with st.form("create_action"):
        name = st.text_input("Name", key="action_name")
        input_schema_text = st.text_area("Input Schema (JSON)", value="{}", height=120)
        output_schema_text = st.text_area("Output Schema (JSON)", value="{}", height=120)
        code = st.text_area("Code (Python)", value="", height=160, help="PoC storage only")
        submitted = st.form_submit_button("Create Action")
        if submitted:
            try:
                payload = {
                    "name": name,
                    "input_schema": json.loads(input_schema_text or "{}"),
                    "output_schema": json.loads(output_schema_text or "{}"),
                    "code": code,
                }
                created = api.create_action(payload)
                st.success(f"Created action #{created['id']}")
            except Exception as e:
                st.error(str(e))

    st.write("Existing actions")
    st.json(api.list_actions())


with tab_connections:
    st.subheader("Connections")
    modules = api.list_modules()
    module_options = {f"{m['name']} ({m['id']})": m["id"] for m in modules}
    with st.form("create_connection"):
        source_label = st.selectbox("Source module", list(module_options.keys()))
        source_port = st.text_input("Source port")
        target_label = st.selectbox("Target module", list(module_options.keys()))
        target_port = st.text_input("Target port")
        transform_text = st.text_area("Transform (JSON, optional)", value="", height=120)
        submitted = st.form_submit_button("Create Connection")
        if submitted:
            try:
                payload: dict[str, Any] = {
                    "source_module_id": module_options[source_label],
                    "source_port": source_port,
                    "target_module_id": module_options[target_label],
                    "target_port": target_port,
                }
                if transform_text.strip():
                    payload["transform"] = json.loads(transform_text)
                created = api.create_connection(payload)
                st.success(f"Created connection #{created['id']}")
            except Exception as e:
                st.error(str(e))

    st.write("Existing connections")
    st.json(api.list_connections())

