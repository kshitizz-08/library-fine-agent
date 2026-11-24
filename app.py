"""
Library Fine Calculator — Clean & Simple Helpdesk UI
Provides a clear, user-friendly interface for students and library staff.
Hides complex technical traces inside optional review expanders.
"""

import os
import sys
import uuid
from datetime import date, timedelta
import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.calculator_agent import calculate_library_fine
from agents.policy_agent import PolicyAgent
from agents.supervisor import LibrarySupervisor
from approvals.waiver_approval import get_waiver_manager
from memory.sqlite_memory import MemberProfileStore, get_sqlite_checkpointer
from rag.rag_engine import get_policy_rag
from reminders.reminder_engine import get_reminder_manager, LOAN_PERIODS, DAILY_FINE_RATES

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
BOOKS_ICON_PATH = os.path.join(os.path.dirname(__file__), "static", "books_icon.png")
LOGO_PATH = os.path.join(os.path.dirname(__file__), "static", "mit_adt_logo.png")

st.set_page_config(
    page_title="Library Fine Calculator — MIT ADT",
    page_icon=BOOKS_ICON_PATH if os.path.exists(BOOKS_ICON_PATH) else "📚",
    layout="wide",
)

# Custom minimal styling to make text and cards crisp and modern
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1050px;
    }
    .rate-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .rate-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.25rem;
    }
    .rate-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
    }
    .rate-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .chat-bubble-user {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #94a3b8;
    }
    .chat-bubble-assistant {
        background: #f0fdf4;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #16a34a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# LOAD BACKEND (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_backend():
    checkpointer = get_sqlite_checkpointer()
    supervisor = LibrarySupervisor(checkpointer=checkpointer, max_hops=4)
    profile_store = MemberProfileStore()
    waiver_manager = get_waiver_manager()
    policy_rag = get_policy_rag()
    reminder_manager = get_reminder_manager()
    return supervisor, profile_store, waiver_manager, policy_rag, reminder_manager

supervisor, profile_store, waiver_manager, policy_rag, reminder_manager = load_backend()

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:6]}"
if "member_id" not in st.session_state:
    st.session_state.member_id = "MEM-001"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefilled_question" not in st.session_state:
    st.session_state.prefilled_question = ""
if "is_staff" not in st.session_state:
    st.session_state.is_staff = False

# ---------------------------------------------------------
# SIDEBAR (CLEAN & MINIMAL)
# ---------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        col_sb1, col_sb2 = st.columns([1, 2.5])
        with col_sb1:
            st.image(LOGO_PATH, width=65)
        with col_sb2:
            st.markdown(
                "<div style='margin-top: 6px;'><strong>MIT ADT</strong><br><span style='color: #64748b; font-size: 0.85rem;'>Library Desk</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.title("📚 Library Desk")
        st.caption("Central Library Services")

    st.markdown("---")
    st.subheader("Your Profile")
    member_input = st.text_input("Student / Member ID", value=st.session_state.member_id)
    if member_input != st.session_state.member_id:
        st.session_state.member_id = member_input
        st.rerun()

    prof = profile_store.get_profile(st.session_state.member_id)
    st.write(f"**Name:** {prof.get('name', 'Library Member')}")
    st.write(f"**Tier:** {prof.get('tier', 'Student')}")

    if st.button("🔄 Start New Chat", use_container_width=True):
        st.session_state.thread_id = f"session-{uuid.uuid4().hex[:6]}"
        st.session_state.chat_history = []
        st.session_state.prefilled_question = ""
        st.success("New conversation started!")
        st.rerun()

    st.markdown("---")
    st.caption("Local AI: Ollama `qwen2.5:3b` + `nomic-embed-text`")

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
hdr_col1, hdr_col2 = st.columns([1, 8])
with hdr_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=95)
    elif os.path.exists(BOOKS_ICON_PATH):
        st.image(BOOKS_ICON_PATH, width=80)
with hdr_col2:
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
            <h1 style="margin: 0; font-size: 2.1rem; color: #1e293b;">
                Library Fine Calculator & Helpdesk
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 1rem;">
                <strong>MIT ADT University</strong> • Check overdue fines, loan rules, and submit fee waiver requests.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.write("")

# Simple Rate Cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="rate-card">
            <div class="rate-title">📖 Novels / Fiction</div>
            <div class="rate-value">₹5 <span style="font-size:0.9rem; font-weight:normal; color:#64748b;">/ day</span></div>
            <div class="rate-sub">Loan period: 14 days</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="rate-card">
            <div class="rate-title">📘 Textbooks / Academic</div>
            <div class="rate-value">₹10 <span style="font-size:0.9rem; font-weight:normal; color:#64748b;">/ day</span></div>
            <div class="rate-sub">Loan period: 7 days</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="rate-card">
            <div class="rate-title">📕 Reference Works</div>
            <div class="rate-value">₹15 <span style="font-size:0.9rem; font-weight:normal; color:#64748b;">/ day</span></div>
            <div class="rate-sub">Loan period: 2 days</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# MAIN TABS (SIMPLIFIED TO 3 TABS)
# ---------------------------------------------------------
tab_chat, tab_waiver, tab_books = st.tabs([
    "💬 Ask Question & Calculate Fine",
    "📝 Fine Waiver Request",
    "📖 My Borrowed Books & Due Dates",
])

# =========================================================
# TAB 1: CHAT & FINE CALCULATOR
# =========================================================
with tab_chat:
    col_t1, col_t2 = st.columns([0.6, 9.4])
    with col_t1:
        if os.path.exists(BOOKS_ICON_PATH):
            st.image(BOOKS_ICON_PATH, width=38)
    with col_t2:
        st.write("### Ask a Question")
    st.caption("Select a sample question below or type your own question in plain English:")

    # Quick Click Buttons for common tests
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("Novel 5 Days Late", use_container_width=True):
            st.session_state.prefilled_question = "A novel was returned 5 days late. How much is the library fine?"
    with btn_col2:
        if st.button("Reference Follow-up", use_container_width=True):
            st.session_state.prefilled_question = "And if it were a reference book?"
    with btn_col3:
        if st.button("Damaged Book Policy?", use_container_width=True):
            st.session_state.prefilled_question = "Does the library provide free replacement books for damaged reference books?"
    with btn_col4:
        if st.button("Negative Days Guard", use_container_width=True):
            st.session_state.prefilled_question = "Calculate the fine for a novel that is -3 days overdue."

    user_query = st.text_input(
        "Type your question here:",
        value=st.session_state.prefilled_question,
        placeholder="e.g. My novel is 4 days overdue, what is my fine?",
    )

    if st.button("Submit Question", type="primary"):
        if not user_query.strip():
            st.warning("Please enter or select a question first.")
        else:
            with st.spinner("Calculating fine based on library rules..."):
                try:
                    result = supervisor.invoke(
                        question=user_query.strip(),
                        member_id=st.session_state.member_id,
                        thread_id=st.session_state.thread_id,
                        max_hops=4,
                    )
                    st.session_state.chat_history.append({
                        "question": user_query.strip(),
                        "answer": result.get("final_response", ""),
                        "policy_status": result.get("policy_status"),
                        "trace": result.get("execution_trace", []),
                        "policy": result.get("retrieved_policy"),
                    })
                    st.session_state.prefilled_question = ""
                except Exception as ex:
                    st.error(f"Error processing question: {ex}")

    st.markdown("---")

    # Conversation Display
    if not st.session_state.chat_history:
        st.info("No questions asked yet in this session. Click any sample button above to try it out!")
    else:
        st.write("### Conversation History")
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            # User Message
            st.markdown(
                f"""
                <div class="chat-bubble-user">
                    <strong>👤 You asked:</strong><br>
                    {item['question']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Assistant Message
            st.markdown(
                f"""
                <div class="chat-bubble-assistant">
                    <strong>🤖 Library Assistant:</strong><br>
                    {item['answer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Collapsed Expander for Technical Evidence & Trace (Hidden by default!)
            with st.expander("🛠️ View Agent Details & Technical Trace (For Reviewers)", expanded=False):
                if item.get("policy"):
                    pol = item["policy"]
                    st.write("**Policy Grounding:**")
                    st.write(f"- Source Document: `{pol.get('source', 'library_rules.txt')}`")
                    st.write(f"- Rate Verified: ₹{pol.get('daily_rate')} per day")
                    st.code(pol.get("supporting_text", ""), language="text")

                if item.get("trace"):
                    st.write("**Supervisor Multi-Agent Execution Hops:**")
                    for step in item["trace"]:
                        hop = step.get("hop", "?")
                        agent_name = step.get("agent", "").upper()
                        action = step.get("action", "")
                        st.write(f"- **[Hop {hop}] {agent_name}**: {action}")

            st.write("")

# =========================================================
# TAB 2: FINE WAIVER REQUESTS
# =========================================================
with tab_waiver:
    st.write("### Fine Waiver Portal")
    st.caption("Submit an official request to waive library fines due to medical emergency, hospitalization, or sanctioned university travel.")

    w_col1, w_col2 = st.columns([1, 1], gap="large")

    with w_col1:
        st.write("#### 📝 Submit a New Request")
        with st.form("waiver_form"):
            w_book = st.selectbox("Book Type", ["novel", "textbook", "reference"], format_func=lambda x: x.capitalize())
            w_days = st.number_input("Days Overdue", min_value=1, max_value=180, value=5)

            rates = {"novel": 5, "textbook": 10, "reference": 15}
            fine_est = w_days * rates[w_book]
            st.info(f"Total fine amount to waive: **₹{fine_est}** ({w_days} days × ₹{rates[w_book]}/day)")

            w_reason = st.text_area(
                "Reason for Exemption",
                placeholder="e.g. Hospitalized from Nov 10 to Nov 15 with medical certificate attached.",
            )
            submit_waiver_btn = st.form_submit_button("Submit Waiver Request", type="primary", use_container_width=True)

            if submit_waiver_btn:
                if not w_reason.strip():
                    st.warning("Please enter a valid reason.")
                else:
                    new_w = waiver_manager.create_waiver_request(
                        member_id=st.session_state.member_id,
                        thread_id=st.session_state.thread_id,
                        book_type=w_book,
                        days_overdue=w_days,
                        fine_amount=fine_est,
                        reason=w_reason,
                        policy_evidence="Library fine waivers require review by authorized library staff.",
                    )
                    st.success(f"Waiver submitted successfully! Request ID: `{new_w['request_id']}` (Status: PENDING)")
                    st.rerun()

    with w_col2:
        st.write("#### 👨‍💼 Staff Review Desk")
        is_staff = st.toggle(
            "👨‍💼 Librarian Authorization Mode",
            value=st.session_state.get("is_staff", False),
            key="staff_mode_toggle",
            help="Turn ON to approve or reject student fee waivers as staff.",
        )
        st.session_state.is_staff = is_staff
        staff_id = "librarian_staff" if is_staff else "student"

        if not is_staff:
            st.warning("⚠️ **Staff approval is locked for students** (to prevent self-approval). Turn ON the toggle above to authorize.")
        else:
            st.success("✅ **Librarian Mode Active:** You are authorized to approve or reject waivers.")

        pending_list = waiver_manager.list_waivers(status="PENDING")
        if not pending_list:
            st.write("✅ No pending waiver requests waiting for review.")
        else:
            w_options = {f"{w['request_id']} — {w['member_id']} (₹{w['fine_amount']})": w["request_id"] for w in pending_list}
            chosen_label = st.selectbox("Select Pending Request", list(w_options.keys()))
            chosen_id = w_options[chosen_label]
            req = waiver_manager.get_waiver_request(chosen_id)

            if req:
                st.write(f"**Member ID:** `{req['member_id']}` | **Book:** {req['book_type'].capitalize()} | **Fine:** ₹{req['fine_amount']}")
                st.write(f"**Reason:** {req['reason']}")

                staff_note = st.text_input("Reviewer Notes", value="Verified documentation")

                app_col, rej_col = st.columns(2)
                with app_col:
                    if st.button("✅ Approve Waiver", type="primary", use_container_width=True):
                        if not is_staff:
                            st.error("🔒 Action Blocked: Please turn ON '👨‍💼 Librarian Authorization Mode' toggle above first!")
                        else:
                            try:
                                waiver_manager.decide_waiver(
                                    request_id=chosen_id,
                                    reviewer_id=staff_id,
                                    decision="APPROVED",
                                    decision_reason=staff_note,
                                    is_authorized=True,
                                strangers_error=None)
                            except TypeError:
                                waiver_manager.decide_waiver(
                                    request_id=chosen_id,
                                    reviewer_id=staff_id,
                                    decision="APPROVED",
                                    decision_reason=staff_note,
                                    is_authorized=True,
                                )
                            st.success(f"Request {chosen_id} APPROVED successfully!")
                            st.rerun()

                with rej_col:
                    if st.button("❌ Reject Waiver", use_container_width=True):
                        if not is_staff:
                            st.error("🔒 Action Blocked: Please turn ON '👨‍💼 Librarian Authorization Mode' toggle above first!")
                        else:
                            try:
                                waiver_manager.decide_waiver(
                                    request_id=chosen_id,
                                    reviewer_id=staff_id,
                                    decision="REJECTED",
                                    decision_reason=staff_note,
                                    is_authorized=True,
                                )
                                st.error(f"Request {chosen_id} REJECTED.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

    st.markdown("---")
    st.write("#### 📋 All Waiver Records")
    all_w = waiver_manager.list_waivers()
    if all_w:
        # Show simple clean table
        st.dataframe(all_w, use_container_width=True)
    else:
        st.caption("No waivers on record.")

# =========================================================
# TAB 3: BORROWED BOOKS & DUE DATES
# =========================================================
with tab_books:
    st.write("### My Active Book Loans")
    st.caption("View return deadlines and avoid late fine accumulation.")

    loans = reminder_manager.get_member_loans(st.session_state.member_id, status_filter="ACTIVE")

    b_col1, b_col2 = st.columns([3, 2], gap="large")

    with b_col1:
        if not loans:
            st.info(f"No active borrowed books for Member `{st.session_state.member_id}`.")
        else:
            for l in loans:
                urg = l["urgency"]
                title = l["book_title"]
                due = l["due_date"]
                rate = l["daily_rate"]
                lid = l["loan_id"]

                if urg == "OVERDUE":
                    status_str = f"🔴 OVERDUE ({abs(l['days_remaining'])} days late) — Fine accrued: ₹{l['fine_accrued']}"
                elif urg == "DUE_TODAY":
                    status_str = f"🟡 DUE TODAY ({due}) — Return today to avoid ₹{rate}/day fine!"
                else:
                    status_str = f"🟢 Due in {l['days_remaining']} days ({due})"

                st.write(f"**{title}** ({l['book_type'].capitalize()})")
                st.write(status_str)

                c_ret, _ = st.columns([1, 2])
                with c_ret:
                    if st.button(f"📥 Return Book", key=f"ret_{lid}"):
                        ret = reminder_manager.return_loan(lid)
                        if ret.get("days_overdue", 0) > 0:
                            st.warning(f"Book returned late! Fine assessed: ₹{ret['fine_accrued']}.")
                        else:
                            st.success("Book returned on time! Overdue fine: ₹0.")
                        st.rerun()
                st.markdown("---")

    with b_col2:
        st.write("#### Issue New Book")
        with st.form("new_loan_form"):
            new_title = st.text_input("Book Title", placeholder="e.g. Introduction to Algorithms")
            new_type = st.selectbox("Category", ["novel", "textbook", "reference"], format_func=lambda x: x.capitalize())
            new_btn = st.form_submit_button("Borrow Book", type="primary", use_container_width=True)

            if new_btn:
                if not new_title.strip():
                    st.warning("Please enter a book title.")
                else:
                    new_rec = reminder_manager.issue_loan(
                        member_id=st.session_state.member_id,
                        book_title=new_title.strip(),
                        book_type=new_type,
                        borrow_date_str=date.today().strftime("%Y-%m-%d"),
                    )
                    st.success(f"Book '{new_title}' issued! Due date: {new_rec['due_date']}.")
                    st.rerun()
