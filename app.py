"""
Streamlit Dashboard for Smart Attendance System
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import time
import subprocess
import sys

from config import MODELS_DIR, KNOWN_FACES_DIR
from core.attendance_logger import AttendanceLogger

st.set_page_config(page_title="Smart Attendance Dashboard", layout="wide", initial_sidebar_state="expanded")

template = "plotly_dark"

st.markdown(
    """
    <style>
    body { background-color: #0e1117; }
    .metric-card { background-color: #1f2937; padding: 20px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_logger():
    """Get attendance logger instance"""
    return AttendanceLogger()


def page_live_overview():
    """Live Overview Page"""
    st.title("📊 Live Overview")

    logger = get_logger()

    col1, col2, col3, col4 = st.columns(4)

    today_df = logger.get_today_attendance()
    today_count = len(today_df[today_df["status"] == "SUCCESS"]) if len(today_df) > 0 else 0
    proxy_count = len(logger.get_proxy_alerts())
    total_persons = len(list(KNOWN_FACES_DIR.glob("*/")))

    with col1:
        st.metric("📅 Today's Date", datetime.now().strftime("%Y-%m-%d"))

    with col2:
        st.metric("✅ Attendance Count", today_count)

    with col3:
        st.metric("⚠️ Proxy Alerts", proxy_count)

    with col4:
        st.metric("👥 Registered Persons", total_persons)

    st.divider()

    st.subheader("📋 Today's Attendance Records")
    if len(today_df) > 0:
        display_df = today_df[
            [
                "person_name",
                "timestamp",
                "face_confidence",
                "behavior_confidence",
                "status",
                "alert_message",
            ]
        ].copy()
        display_df["face_confidence"] = display_df["face_confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")
        display_df["behavior_confidence"] = display_df["behavior_confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("No attendance records today")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Today's Attendance Distribution")

        if len(today_df) > 0:
            status_counts = today_df["status"].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Attendance Status Distribution",
                template=template,
                color_discrete_map={"SUCCESS": "#00ff00", "PROXY": "#ff0000", "BLOCKED": "#ffaa00"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to display")

    with col2:
        st.subheader("👥 Top Attendees (Today)")

        if len(today_df) > 0:
            person_counts = today_df[today_df["status"] == "SUCCESS"]["person_name"].value_counts().head(10)
            fig = px.bar(
                x=person_counts.index,
                y=person_counts.values,
                labels={"x": "Person", "y": "Count"},
                title="Top Attendees",
                template=template,
                color_discrete_sequence=["#00ff00"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attendance data")


def page_attendance_history():
    """Attendance History Page"""
    st.title("📅 Attendance History")

    logger = get_logger()

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))

    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    with col3:
        person_filter = st.text_input("Filter by Person Name", "")

    attendance_df = logger.get_attendance_range(start_date, end_date)

    if person_filter:
        attendance_df = attendance_df[attendance_df["person_name"].str.contains(person_filter, case=False, na=False)]

    st.subheader(f"📊 Records: {len(attendance_df)}")

    if len(attendance_df) > 0:
        display_df = attendance_df[
            [
                "person_name",
                "timestamp",
                "face_confidence",
                "behavior_confidence",
                "status",
                "alert_message",
                "attempts",
            ]
        ].copy()
        display_df["face_confidence"] = display_df["face_confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")
        display_df["behavior_confidence"] = display_df["behavior_confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")

        st.dataframe(display_df, use_container_width=True, height=500)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Daily Attendance Trend")
            daily_counts = (
                attendance_df[attendance_df["status"] == "SUCCESS"].groupby(pd.to_datetime(attendance_df["timestamp"]).dt.date).size()
            )
            fig = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                labels={"x": "Date", "y": "Attendance Count"},
                title="Daily Attendance Trend",
                template=template,
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Status Distribution")
            status_counts = attendance_df["status"].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Overall Status Distribution",
                template=template,
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No records found for the selected date range")


def page_proxy_alerts():
    """Proxy Alerts Page"""
    st.title("🚨 Proxy Alerts")

    logger = get_logger()

    proxy_alerts = logger.get_proxy_alerts()

    st.subheader(f"⚠️ Total Proxy Alerts: {len(proxy_alerts)}")

    if len(proxy_alerts) > 0:
        display_df = proxy_alerts[
            [
                "person_name",
                "timestamp",
                "face_confidence",
                "behavior_confidence",
                "alert_message",
                "attempts",
            ]
        ].copy()
        display_df["face_confidence"] = display_df["face_confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")
        display_df["behavior_confidence"] = display_df["behavior_confidence"].apply(lambda x: f"{x*100:.1f}%" if x else "-")

        st.dataframe(display_df, use_container_width=True, height=500)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚠️ Proxy Attempts per Person")
            proxy_by_person = proxy_alerts["person_name"].value_counts()
            fig = px.bar(
                x=proxy_by_person.index,
                y=proxy_by_person.values,
                labels={"x": "Person", "y": "Alert Count"},
                title="Proxy Alerts by Person",
                template=template,
                color_discrete_sequence=["#ff0000"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Proxy Alerts Over Time")
            daily_alerts = proxy_alerts.groupby(pd.to_datetime(proxy_alerts["timestamp"]).dt.date).size()
            fig = px.line(
                x=daily_alerts.index,
                y=daily_alerts.values,
                labels={"x": "Date", "y": "Alert Count"},
                title="Daily Proxy Alerts",
                template=template,
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.success("✅ No proxy alerts detected!")


def page_registered_persons():
    """Registered Persons Page"""
    st.title("👥 Registered Persons")

    logger = get_logger()

    persons_dir = Path(KNOWN_FACES_DIR)
    registered_persons = sorted([p.name for p in persons_dir.glob("*") if p.is_dir()])

    st.subheader(f"📋 Total Registered Persons: {len(registered_persons)}")

    st.divider()

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🔄 Retrain All Models"):
            st.info("Training models... This may take a few minutes.")
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/train_behavior_models.py"],
                    cwd=Path(__file__).parent,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    st.success("✅ Models retrained successfully!")
                else:
                    st.error(f"❌ Training failed: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    if len(registered_persons) > 0:
        persons_stats = logger.get_persons_stats()

        for idx, person_name in enumerate(registered_persons):
            with st.expander(f"👤 {person_name}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)

                person_data = persons_stats[persons_stats["name"] == person_name]

                if len(person_data) > 0:
                    row = person_data.iloc[0]
                    registered_at = row["registered_at"]
                    total_attendances = row["total_attendances"]
                    blocked = row["blocked"]
                    proxy_count = row["proxy_count"]

                    with col1:
                        st.metric("📅 Registered", registered_at[:10])

                    with col2:
                        st.metric("✅ Total Attendances", int(total_attendances))

                    with col3:
                        st.metric("⚠️ Proxy Alerts", int(proxy_count) if proxy_count else 0)

                    with col4:
                        if blocked:
                            st.metric("🚫 Status", "BLOCKED")
                        else:
                            st.metric("✅ Status", "ACTIVE")

                    person_dir = persons_dir / person_name
                    face_images = list(person_dir.glob("*.jpg"))
                    st.write(f"**Face Images**: {len(face_images)}")

                    person_attendance = logger.get_person_attendance(person_name)
                    st.write(f"**Total Records**: {len(person_attendance)}")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"🔓 Enable Re-attendance", key=f"enable_{idx}"):
                            result = logger.mark_person_for_reattendance(person_name)
                            st.success(f"✅ {result['message']}")

                    with col2:
                        if st.button(f"🚫 Block Person", key=f"block_{idx}"):
                            result = logger.block_person(person_name)
                            st.warning(f"⚠️ {result['message']}")

    else:
        st.info("No registered persons found. Register using: python scripts/register_person.py --name 'Your Name'")


def page_live_monitoring():
    """Live Monitoring Page"""
    st.title("📡 Live Monitoring")

    logger = get_logger()

    col1, col2 = st.columns([2, 1])

    with col2:
        refresh_rate = st.selectbox("Refresh Rate (seconds)", [2, 3, 5, 10])

    st.divider()

    placeholder = st.empty()

    while True:
        col1, col2, col3, col4, col5 = st.columns(5)

        today_df = logger.get_today_attendance()
        active_persons = today_df["person_name"].unique() if len(today_df) > 0 else []

        with col1:
            st.metric("👥 Active Persons", len(active_persons))

        with col2:
            latest_record = today_df.iloc[0] if len(today_df) > 0 else None
            if latest_record is not None:
                st.metric("👤 Latest", latest_record["person_name"])
            else:
                st.metric("👤 Latest", "None")

        with col3:
            successful = len(today_df[today_df["status"] == "SUCCESS"]) if len(today_df) > 0 else 0
            st.metric("✅ Successful", successful)

        with col4:
            proxy = len(today_df[today_df["status"] == "PROXY"]) if len(today_df) > 0 else 0
            st.metric("⚠️ Proxy Attempts", proxy)

        with col5:
            blocked = len(today_df[today_df["status"] == "BLOCKED"]) if len(today_df) > 0 else 0
            st.metric("🚫 Blocked", blocked)

        st.divider()

        st.subheader("📊 Live Activity Feed")

        if len(today_df) > 0:
            display_df = today_df[[
                "person_name",
                "timestamp",
                "status",
                "behavior_confidence",
                "alert_message",
            ]].head(20).copy()
            display_df["behavior_confidence"] = display_df["behavior_confidence"].apply(
                lambda x: f"{x*100:.1f}%" if x else "-"
            )
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No live activity yet")

        time.sleep(refresh_rate)


def main():
    st.sidebar.title("🎯 Smart Attendance System")

    page = st.sidebar.radio(
        "Select Page",
        [
            "📊 Live Overview",
            "📅 Attendance History",
            "🚨 Proxy Alerts",
            "👥 Registered Persons",
            "📡 Live Monitoring",
        ],
    )

    if page == "📊 Live Overview":
        page_live_overview()
    elif page == "📅 Attendance History":
        page_attendance_history()
    elif page == "🚨 Proxy Alerts":
        page_proxy_alerts()
    elif page == "👥 Registered Persons":
        page_registered_persons()
    elif page == "📡 Live Monitoring":
        page_live_monitoring()

    st.sidebar.divider()
    st.sidebar.info("🔧 Smart Attendance with Behavior Pattern Recognition")


if __name__ == "__main__":
    main()
