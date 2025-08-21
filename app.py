import os
from datetime import datetime, date, time
from typing import List, Dict, Any

import pandas as pd
import streamlit as st


###############################################################################
# Page setup
###############################################################################
st.set_page_config(
    page_title="Maintenance Hub",
    page_icon="🛠️",
    layout="wide",
)


###############################################################################
# Constants and simple utilities
###############################################################################
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_store")
WORK_ORDERS_CSV = os.path.join(DATA_DIR, "work_orders.csv")
JOB_PLANS_CSV = os.path.join(DATA_DIR, "job_plans.csv")
SAFETY_PLANS_CSV = os.path.join(DATA_DIR, "safety_plans.csv")


def ensure_data_dir() -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def load_csv(path: str, default_columns: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=default_columns)
    try:
        df = pd.read_csv(path)
        # Guarantee requested columns exist
        for col in default_columns:
            if col not in df.columns:
                df[col] = None
        return df
    except Exception:
        return pd.DataFrame(columns=default_columns)


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_data_dir()
    df.to_csv(path, index=False)


def generate_work_order_number() -> str:
    # Simple unique-ish generator: WO + yymmddHHMMSS
    return f"WO{datetime.now().strftime('%y%m%d%H%M%S')}"


def init_session_state() -> None:
    if "saved_work_orders" not in st.session_state:
        st.session_state.saved_work_orders = load_csv(
            WORK_ORDERS_CSV,
            [
                "work_order_number",
                "site",
                "class",
                "status",
                "location",
                "asset",
                "work_type",
                "work_status",
                "parent_wo",
                "gl_account",
                "status_date",
                "classification",
                "department",
                "inherit_status_changes",
                "description",
                "zone_facility",
                "accepts_charges",
                "unit_building",
                "is_task",
                "failure_class",
                "problem_code",
                "dot_required",
                "planner",
                "oq_tasks",
                "moc_number",
                "ehs_permit_required",
                # Job Details (subset)
                "job_plan",
                "pm",
                "safety_plan",
                "contract",
                "asset_up",
                "asset_location_priority",
                "warranties_exist",
                "sla_applied",
                "priority_justification",
                "charge_to_store",
                "risk_assessment",
                # Scheduling
                "target_start",
                "actual_start",
                "target_finish",
                "actual_finish",
                "scheduled_start",
                "duration_hours",
                "scheduled_finish",
                "time_remaining",
                "has_follow_up",
                "interruptible",
                # Responsibility
                "entered_by",
                "supervisor",
                "owner",
                "reported_by",
                "operations_engineering",
                "owner_group",
                "reported_date",
                "lead_person_craft",
                "on_behalf_of",
                "service",
                "phone",
                "work_group",
                "vendor",
                "service_group",
                # Timestamps
                "created_at",
                "updated_at",
            ],
        )

    if "job_plans" not in st.session_state:
        st.session_state.job_plans = load_csv(
            JOB_PLANS_CSV,
            [
                "job_plan_id",
                "job_plan_name",
                "description",
                "duration_hours",
                "lead",
                "supervisor",
                "status",
                "organization",
                "site",
                "attachment",
                "wo_priority",
                "interruptible",
                "crew",
                "work_group",
                "owner",
                "owner_group",
                "created_at",
                "updated_at",
            ],
        )

    if "safety_plans" not in st.session_state:
        # Seed with a couple of examples for search UX
        default = pd.DataFrame(
            [
                {
                    "safety_plan_id": "SP001",
                    "description": "Confined Space Entry",
                    "site": "Refinery",
                    "status": "Active",
                    "hazard_type": "Physical",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                {
                    "safety_plan_id": "SP002",
                    "description": "Chemical Handling PPE",
                    "site": "Chemical Plant",
                    "status": "Active",
                    "hazard_type": "Chemical",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            ]
        )
        existing = load_csv(
            SAFETY_PLANS_CSV,
            ["safety_plan_id", "description", "site", "status", "hazard_type", "created_at"],
        )
        st.session_state.safety_plans = (
            existing if len(existing) > 0 else default
        )

    if "bookmarked_safety_queries" not in st.session_state:
        st.session_state.bookmarked_safety_queries = []


init_session_state()


###############################################################################
# Option catalogs (mocked)
###############################################################################
SITES = ["Refinery", "Warehouse 1", "Chemical Plant", "Headquarters"]
WO_CLASSES = ["WORKORDER", "Corrective Maintenance", "Preventive", "Inspection"]
WO_STATUSES = ["DRAFT", "WAPPR", "IN PROGRESS", "COMPLETE", "CANCELLED"]
WORK_TYPES = ["REPAIR", "MAINTENANCE", "REPLACEMENT", "INSPECTION"]
WORK_STATUSES = ["Assigned", "Waiting Materials", "In Queue", "In Progress", "Done"]
CLASSIFICATIONS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
DEPARTMENTS = ["MAINTENANCE", "OPERATIONS", "ENGINEERING"]
FAILURE_CLASSES = ["Mechanical Failure", "Electrical Failure", "Hydraulic Failure", "PUMP FAILURE"]
PROBLEM_CODES = ["Steam Leak", "PRESSURE LOSS", "Overheat", "Vibration"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]
SERVICE_GROUPS = ["INTERNAL", "EXTERNAL CONTRACTOR"]
WORK_GROUPS = ["MECHANICAL", "ELECTRICAL", "MAINTENANCE_GROUP"]
ORG_LIST = ["XYZ Corp", "CHS"]
CREWS = ["CREW1", "CREW2"]
JOB_PLAN_STATUSES = ["DRAFT", "ACTIVE", "INPROGRESS", "COMPLETE", "CANCELLED"]


###############################################################################
# Widgets helpers
###############################################################################
def two_col_text(label_left: str, label_right: str, key_left: str, key_right: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.text_input(label_left, key=key_left)
    with c2:
        st.text_input(label_right, key=key_right)


def datetime_inputs(prefix: str, label: str) -> Dict[str, Any]:
    c1, c2 = st.columns(2)
    with c1:
        d: date = st.date_input(f"{label} Date", key=f"{prefix}_date")
    with c2:
        t: time = st.time_input(f"{label} Time", key=f"{prefix}_time")
    dt = datetime.combine(d, t)
    return {"date": d.isoformat(), "time": t.strftime("%H:%M:%S"), "datetime": dt.strftime("%Y-%m-%d %H:%M:%S")}


###############################################################################
# Work Order Tab
###############################################################################
def render_work_order_tab() -> None:
    st.subheader("Work Order Information")

    # Generate WO number once per form open
    if "current_wo_number" not in st.session_state:
        st.session_state.current_wo_number = generate_work_order_number()

    with st.form("work_order_form", clear_on_submit=False):
        st.markdown("Use this form to create or update a Work Order.")

        info = st.container()
        with info:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.text_input("Work Order Number", value=st.session_state.current_wo_number, disabled=True)
                site = st.selectbox("Site", SITES)
                wo_class = st.selectbox("Class", WO_CLASSES)
                status = st.selectbox("Status", WO_STATUSES, index=1)
                location = st.text_input("Location", placeholder="REFINERY-PLANT-A")
                asset = st.text_input("Asset", placeholder="PUMP-24567A")
                work_type = st.selectbox("Work Type", WORK_TYPES, index=0)
                work_status = st.selectbox("Work Status", WORK_STATUSES, index=3)
                parent_wo = st.text_input("Parent WO", placeholder="WO123450 or 78097")
                gl_account = st.text_input("GL Account", placeholder="4005-MAINT-EXP")
            with c2:
                status_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.text_input("Status Date", value=status_dt, disabled=True)
                classification = st.selectbox("Classification", CLASSIFICATIONS, index=0)
                department = st.selectbox("Department", DEPARTMENTS, index=0)
                inherit_status_changes = st.checkbox("Inherit Status Changes?", value=True)
                description = st.text_area("Description", placeholder="Detailed explanation of the work order")
                zone_facility = st.text_input("Zone/Facility", placeholder="ZONE-3")
                accepts_charges = st.checkbox("Accepts Charges?", value=True)
                unit_building = st.text_input("Unit/Bldg.", placeholder="UNIT-12")
                is_task = st.checkbox("Is Task?", value=False)
                failure_class = st.selectbox("Failure Class", FAILURE_CLASSES, index=0)
            with c3:
                problem_code = st.selectbox("Problem Code", PROBLEM_CODES, index=1)
                dot_required = st.checkbox("DOT Documentation Required?", value=False)
                planner = st.text_input("Planner", placeholder="Planner name")
                oq_tasks = st.checkbox("OQ Covered Tasks?", value=False)
                moc_number = st.text_input("MOC Number", placeholder="MOC-2024-002")
                ehs_permit_required = st.checkbox("EHS Permit Required?", value=False)

        st.markdown("---")
        st.subheader("Job Details")
        jd1, jd2, jd3 = st.columns(3)
        with jd1:
            job_plan = st.text_input("Job Plan", placeholder="PUMP-REPAIR-PLAN-A")
            pm = st.text_input("PM", placeholder="PM-2024-001")
            safety_plan = st.text_input("Safety Plan", placeholder="SAFETY-PLAN-A")
            contract = st.text_input("Contract", placeholder="CONTRACT-4567")
        with jd2:
            asset_up = st.checkbox("Asset Up?", value=True)
            asset_location_priority = st.selectbox("Asset/Location Priority", ["LOW", "MEDIUM", "HIGH"], index=2)
            warranties_exist = st.checkbox("Warranties Exist?", value=False)
            sla_applied = st.checkbox("SLA Applied?", value=True)
        with jd3:
            priority_justification = st.text_area("Priority Justification", placeholder="Reason for priority")
            charge_to_store = st.checkbox("Charge to Store?", value=True)
            risk_assessment = st.selectbox("Risk Assessment", RISK_LEVELS, index=1)

        st.markdown("---")
        st.subheader("Scheduling Information")
        ts = datetime_inputs("target_start", "Target Start")
        as_ = datetime_inputs("actual_start", "Actual Start")
        tf = datetime_inputs("target_finish", "Target Finish")
        af = datetime_inputs("actual_finish", "Actual Finish")
        ss = datetime_inputs("scheduled_start", "Scheduled Start")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            duration_hours = st.number_input("Duration (hours)", min_value=0.0, step=0.25, value=8.0)
        with c2:
            sf = datetime_inputs("scheduled_finish", "Scheduled Finish")
        with c3:
            time_remaining = st.text_input("Time Remaining", placeholder="01:30")
        with c4:
            has_follow_up = st.checkbox("Has Follow-up Work?", value=False)
            interruptible = st.checkbox("Interruptible?", value=True)

        st.markdown("---")
        st.subheader("Responsibility")
        r1, r2, r3 = st.columns(3)
        with r1:
            entered_by = st.text_input("Who Entered it?", placeholder="Username")
            supervisor = st.text_input("Supervisor", placeholder="SUPERVISOR-001")
            owner = st.text_input("Owner", placeholder="MAINTENANCE DEPT.")
            reported_by = st.text_input("Who Reported It", placeholder="JOHN DOE")
            operations_engineering = st.text_input("Operations/Engineering", placeholder="ENGINEERING DEPT.")
        with r2:
            owner_group = st.text_input("Owner Group", placeholder="MAINT-TEAM-01")
            reported_date = st.text_input("Reported Date", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
            lead_person_craft = st.text_input("Lead Person/Craft", placeholder="MECHANIC-JIM")
            on_behalf_of = st.text_input("On Behalf Of", placeholder="MAINTENANCE SUPERVISOR")
            service = st.text_input("Service", placeholder="OIL PIPE MAINTENANCE")
        with r3:
            phone = st.text_input("Phone", placeholder="555-1234")
            work_group = st.selectbox("Work Group", WORK_GROUPS, index=0)
            vendor = st.text_input("Vendor", placeholder="PUMP-MANUFACTURER")
            service_group = st.selectbox("Service Group", SERVICE_GROUPS, index=1)

        st.markdown("---")
        b1, b2, b3, b4 = st.columns([1, 1, 1, 6])
        with b1:
            save_clicked = st.form_submit_button("OK (Save)", type="primary")
        with b2:
            clear_clicked = st.form_submit_button("Clear")
        with b3:
            cancel_clicked = st.form_submit_button("Cancel")
        with b4:
            st.caption("Use OK to persist locally (CSV). Clear resets inputs. Cancel discards current unsaved changes.")

        if save_clicked:
            record = {
                "work_order_number": st.session_state.current_wo_number,
                "site": site,
                "class": wo_class,
                "status": status,
                "location": location,
                "asset": asset,
                "work_type": work_type,
                "work_status": work_status,
                "parent_wo": parent_wo,
                "gl_account": gl_account,
                "status_date": status_dt,
                "classification": classification,
                "department": department,
                "inherit_status_changes": inherit_status_changes,
                "description": description,
                "zone_facility": zone_facility,
                "accepts_charges": accepts_charges,
                "unit_building": unit_building,
                "is_task": is_task,
                "failure_class": failure_class,
                "problem_code": problem_code,
                "dot_required": dot_required,
                "planner": planner,
                "oq_tasks": oq_tasks,
                "moc_number": moc_number,
                "ehs_permit_required": ehs_permit_required,
                "job_plan": job_plan,
                "pm": pm,
                "safety_plan": safety_plan,
                "contract": contract,
                "asset_up": asset_up,
                "asset_location_priority": asset_location_priority,
                "warranties_exist": warranties_exist,
                "sla_applied": sla_applied,
                "priority_justification": priority_justification,
                "charge_to_store": charge_to_store,
                "risk_assessment": risk_assessment,
                "target_start": ts["datetime"],
                "actual_start": as_["datetime"],
                "target_finish": tf["datetime"],
                "actual_finish": af["datetime"],
                "scheduled_start": ss["datetime"],
                "duration_hours": duration_hours,
                "scheduled_finish": sf["datetime"],
                "time_remaining": time_remaining,
                "has_follow_up": has_follow_up,
                "interruptible": interruptible,
                "entered_by": entered_by,
                "supervisor": supervisor,
                "owner": owner,
                "reported_by": reported_by,
                "operations_engineering": operations_engineering,
                "owner_group": owner_group,
                "reported_date": reported_date,
                "lead_person_craft": lead_person_craft,
                "on_behalf_of": on_behalf_of,
                "service": service,
                "phone": phone,
                "work_group": work_group,
                "vendor": vendor,
                "service_group": service_group,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.saved_work_orders = pd.concat(
                [st.session_state.saved_work_orders, pd.DataFrame([record])], ignore_index=True
            )
            save_csv(st.session_state.saved_work_orders, WORK_ORDERS_CSV)
            st.success(f"Saved Work Order {record['work_order_number']}")
            st.session_state.current_wo_number = generate_work_order_number()

        if clear_clicked:
            # Reset current WO to a fresh number and rerun to clear ephemeral widget state
            st.session_state.current_wo_number = generate_work_order_number()
            st.rerun()

        if cancel_clicked:
            st.warning("Cancelled. No changes were saved.")

    st.markdown("---")
    st.subheader("Saved Work Orders")
    wo_df = st.session_state.saved_work_orders.copy()
    st.dataframe(wo_df, use_container_width=True, hide_index=True)
    csv_data = wo_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="work_orders.csv",
        mime="text/csv",
    )


###############################################################################
# Plans Tab (Job Plans)
###############################################################################
def render_plans_tab() -> None:
    st.subheader("Job Plan Information")

    with st.form("job_plan_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            job_plan_id = st.text_input("Job Plan ID", placeholder="F302")
            job_plan_name = st.text_input("Job Plan Name", placeholder="TESTINY")
            description = st.text_area("Description", placeholder="Change out F302 carbon filter")
            duration_hours = st.number_input("Duration (hours)", min_value=0.0, step=0.25, value=0.3)
        with c2:
            lead = st.text_input("Lead", placeholder="John Doe")
            supervisor = st.text_input("Supervisor", placeholder="MICHAEL")
            status = st.selectbox("Status", JOB_PLAN_STATUSES, index=0)
            organization = st.selectbox("Organization", ORG_LIST, index=1)
        with c3:
            site = st.selectbox("Site", SITES, index=0)
            attachment = st.text_input("Attachment (path or URL)")
            wo_priority = st.number_input("WO Priority", min_value=0, max_value=10, value=9)
            interruptible = st.checkbox("Interruptible?", value=False)
            crew = st.selectbox("Crew", CREWS, index=0)
        c4, c5, c6 = st.columns(3)
        with c4:
            work_group = st.selectbox("Work Group", WORK_GROUPS, index=0)
        with c5:
            owner = st.text_input("Owner", placeholder="JANE DOE")
        with c6:
            owner_group = st.text_input("Owner Group", placeholder="OPERATIONS_GROUP")

        sub = st.form_submit_button("Save Job Plan", type="primary")
        if sub:
            rec = {
                "job_plan_id": job_plan_id,
                "job_plan_name": job_plan_name,
                "description": description,
                "duration_hours": duration_hours,
                "lead": lead,
                "supervisor": supervisor,
                "status": status,
                "organization": organization,
                "site": site,
                "attachment": attachment,
                "wo_priority": wo_priority,
                "interruptible": interruptible,
                "crew": crew,
                "work_group": work_group,
                "owner": owner,
                "owner_group": owner_group,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.job_plans = pd.concat(
                [st.session_state.job_plans, pd.DataFrame([rec])], ignore_index=True
            )
            save_csv(st.session_state.job_plans, JOB_PLANS_CSV)
            st.success(f"Saved Job Plan {job_plan_id}")

    st.markdown("---")
    st.subheader("Labor (per Job Plan)")
    st.caption("Use the table below to outline craft, quantity, hours, and costs. Download for sharing.")

    default_labor = pd.DataFrame(
        [
            {
                "task": 10,
                "craft": "TECHCRFT",
                "skill_level": "JOURNEYMAN",
                "vendor": "ABC Contracting",
                "quantity": 1,
                "labor_note": "Special Skills Required",
                "regular_hours": 1.0,
                "rate": 20.0,
                "line_cost": 20.0,
                "rate_changed": False,
            }
        ]
    )
    if "labor_editor_df" not in st.session_state:
        st.session_state.labor_editor_df = default_labor

    edited = st.data_editor(
        st.session_state.labor_editor_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "rate_changed": st.column_config.CheckboxColumn("Rate Changed?"),
            "regular_hours": st.column_config.NumberColumn("Regular Hours", step=0.25),
            "rate": st.column_config.NumberColumn("Rate", step=0.5),
            "line_cost": st.column_config.NumberColumn("Line Cost", step=0.5),
        },
        key="labor_editor",
    )

    # Auto-calc line cost
    if len(edited) > 0:
        try:
            edited["line_cost"] = edited["regular_hours"].fillna(0) * edited["rate"].fillna(0)
        except Exception:
            pass
    st.session_state.labor_editor_df = edited

    dl = edited.to_csv(index=False).encode("utf-8")
    st.download_button("Download Labor CSV", dl, file_name="job_plan_labor.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Saved Job Plans")
    st.dataframe(st.session_state.job_plans, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Job Plans CSV",
        st.session_state.job_plans.to_csv(index=False).encode("utf-8"),
        file_name="job_plans.csv",
        mime="text/csv",
    )


###############################################################################
# Safety Tab
###############################################################################
def render_safety_tab() -> None:
    st.subheader("Safety Plans Search")
    q1, q2 = st.columns([3, 1])
    with q1:
        find = st.text_input("Find", placeholder="Confined Space")
    with q2:
        bookmark = st.button("Bookmark this search")
    adv = st.expander("Advanced Search")
    with adv:
        c1, c2, c3 = st.columns(3)
        with c1:
            site = st.selectbox("Site", ["(All)"] + SITES, index=0)
        with c2:
            status = st.selectbox("Status", ["(All)", "Active", "Inactive", "Under Review"], index=1)
        with c3:
            hazard = st.selectbox("Hazard Type", ["(All)", "Chemical", "Biological", "Physical", "Mechanical", "Electrical"], index=0)

    df = st.session_state.safety_plans.copy()
    if find:
        df = df[df["description"].str.contains(find, case=False, na=False) | df["safety_plan_id"].str.contains(find, case=False, na=False)]
    if adv:
        if site and site != "(All)":
            df = df[df["site"] == site]
        if status and status != "(All)":
            df = df[df["status"] == status]
        if hazard and hazard != "(All)":
            df = df[df["hazard_type"].str.contains(hazard, case=False, na=False)]

    st.markdown("---")
    st.caption("Results")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if bookmark and (find or (adv and (site != "(All)" or status != "(All)" or hazard != "(All)"))):
        st.session_state.bookmarked_safety_queries.append({
            "find": find,
            "site": site,
            "status": status,
            "hazard": hazard,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        st.success("Saved current filters as a bookmark.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("All Records")
        st.dataframe(st.session_state.safety_plans, use_container_width=True, hide_index=True)
    with c2:
        st.write("Bookmarks")
        if len(st.session_state.bookmarked_safety_queries) == 0:
            st.info("No bookmarks yet.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.bookmarked_safety_queries), use_container_width=True, hide_index=True)
    with c3:
        st.write("Download")
        st.download_button(
            "Download Selected Plans (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="safety_plans_filtered.csv",
            mime="text/csv",
        )


###############################################################################
# Related Records, Actuals, Logs, Failure Reporting, Reports and Analytics
###############################################################################
def render_related_records_tab() -> None:
    st.subheader("Related Records")
    st.info("Link Work Orders to related WOs, SRs, Assets, or Projects. (Placeholder)")


def render_actuals_tab() -> None:
    st.subheader("Actuals")
    st.info("Capture actual labor, materials, services, and tools. (Placeholder)")


def render_logs_tab() -> None:
    st.subheader("Logs")
    st.info("Chronological activity log and notes. (Placeholder)")


def render_failure_reporting_tab() -> None:
    st.subheader("Failure Reporting")
    st.caption("Log failure codes for analysis.")
    with st.form("failure_code_form"):
        failure_code = st.text_input("Failure Code", placeholder="CAF")
        description = st.text_input("Description", placeholder="Centrifuge Failure")
        organization = st.selectbox("Organization", ORG_LIST, index=0)
        submitted = st.form_submit_button("Save Failure Code")
        if submitted:
            st.success(f"Saved Failure Code {failure_code} - {description} ({organization})")


def render_reports_analytics_tab() -> None:
    st.subheader("Reports & Analytics")
    df = st.session_state.saved_work_orders.copy()
    if len(df) == 0:
        st.info("No data yet. Create Work Orders to see analytics.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.write("Work Orders by Site")
        st.bar_chart(df.groupby("site").size())
    with c2:
        st.write("Work Orders by Status")
        st.bar_chart(df.groupby("status").size())

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.write("Failure Class Distribution")
        st.bar_chart(df.groupby("failure_class").size())
    with c4:
        st.write("Average Duration by Work Type (hours)")
        try:
            st.bar_chart(df.groupby("work_type")["duration_hours"].mean())
        except Exception:
            st.info("Insufficient numeric data for duration.")


###############################################################################
# Main layout
###############################################################################
st.title("🛠️ Maintenance Hub")
st.caption("Work Orders, Plans, Safety, and Analytics")

tabs = st.tabs([
    "Work Order",
    "Plans",
    "Related Records",
    "Actuals",
    "Safety",
    "Logs",
    "Failure Reporting",
    "Reports & Analytics",
])

with tabs[0]:
    render_work_order_tab()
with tabs[1]:
    render_plans_tab()
with tabs[2]:
    render_related_records_tab()
with tabs[3]:
    render_actuals_tab()
with tabs[4]:
    render_safety_tab()
with tabs[5]:
    render_logs_tab()
with tabs[6]:
    render_failure_reporting_tab()
with tabs[7]:
    render_reports_analytics_tab()

