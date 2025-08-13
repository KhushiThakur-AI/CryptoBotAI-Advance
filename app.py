import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import pandas as pd
import datetime

# --- Firestore Initialization (should be robust for Streamlit's re-runs) ---
FIREBASE_CRED_PATH = os.path.join(os.path.dirname(__file__), "firebase_credentials.json")
db = None # Initialize db as None

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        st.success("✅ Firebase initialized.")
        db = firestore.client()
    except Exception as e:
        st.error(f"❌ Firebase init failed: {e}")
else:
    st.info("⚠️ Firebase already initialized.")
    db = firestore.client() # Get the client if already initialized

# --- Configuration (must match main.py) ---
APP_ID = "myApp"
USER_ID = "user123"

# --- Streamlit Dashboard Layout ---
st.set_page_config(layout="wide")
st.title("🤖 Crypto Trading Bot Dashboard")

if db:
    st.header("📊 Bot State & Performance")

    # --- Fetch Bot State ---
    @st.cache_data(ttl=60) # Cache data for 60 seconds to avoid excessive Firestore reads
    def get_bot_state():
        try:
            # Path: artifacts/myApp/users/user123/settings/bot_state
            doc_ref = db.collection(f"artifacts/{APP_ID}/users/{USER_ID}/settings").document("bot_state")
            doc = doc_ref.get()
            if doc.exists:
                state_data = doc.to_dict()
                # paper_positions is stored as a JSON string, so parse it
                state_data["paper_positions"] = json.loads(state_data.get("paper_positions", "{}"))
                st.sidebar.success("Bot state loaded from Firestore.")
                return state_data
            else:
                st.sidebar.warning("No bot state found in Firestore.")
                return None
        except Exception as e:
            st.sidebar.error(f"Error loading bot state: {e}")
            return None

    bot_state = get_bot_state()

    if bot_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Paper Balance (USD)", f"{bot_state.get('paper_balance', 0.0):,.2f}")
        with col2:
            st.metric("Initial Paper Balance (USD)", f"{bot_state.get('initial_paper_balance', 0.0):,.2f}")
        with col3:
            total_pnl = bot_state.get('paper_balance', 0.0) - bot_state.get('initial_paper_balance', 0.0)
            st.metric("Total Paper PnL (USD)", f"{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}") # Show PnL as delta

        st.subheader("Current Open Paper Positions")
        paper_positions = bot_state.get("paper_positions", {})
        if paper_positions:
            # Convert positions dict to a more display-friendly list of dicts
            positions_data = []
            for symbol, pos in paper_positions.items():
                positions_data.append({
                    "Symbol": symbol,
                    "Side": pos.get("side"),
                    "Quantity": pos.get("quantity"),
                    "Entry Price": f"{pos.get('entry_price', 0.0):.2f}",
                    "Stop Loss": f"{pos.get('stop_loss', 0.0):.2f}",
                    "Take Profit": f"{pos.get('take_profit', 0.0):.2f}",
                    "Highest Price Since Entry": f"{pos.get('highest_price_since_entry', 0.0):.2f}",
                    "Current Trailing Stop": f"{pos.get('current_trailing_stop_price', 0.0):.2f}"
                })
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("No open paper positions.")

    # --- Fetch Trade History ---
    st.header("📜 Trade History")

    @st.cache_data(ttl=60) # Cache data for 60 seconds
    def get_trade_history():
        try:
            # Path: artifacts/myApp/users/user123/trades
            trades_ref = db.collection(f"artifacts/{APP_ID}/users/{USER_ID}/trades")
            # Order by server timestamp to get the latest trades first
            docs = trades_ref.order_by("timestamp_server", direction=firestore.Query.DESCENDING).limit(100).stream()
            trade_list = []
            for doc in docs:
                trade_data = doc.to_dict()
                # Convert timestamp string to datetime object for better handling
                trade_data['timestamp'] = datetime.datetime.strptime(trade_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                trade_list.append(trade_data)
            st.sidebar.success(f"Loaded {len(trade_list)} trades from Firestore.")
            return pd.DataFrame(trade_list)
        except Exception as e:
            st.sidebar.error(f"Error loading trade history: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

    trade_df = get_trade_history()

    if not trade_df.empty:
        st.subheader("Recent Trades")
        # Select relevant columns for display
        display_df = trade_df[[
            'timestamp', 'symbol', 'type', 'price', 'quantity', 'pnl', 'reason', 'real_trade', 'entry_price',
            'sl_price_at_entry', 'tp_price_at_entry', 'tsl_price_at_hit'
        ]].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') # Format for display
        st.dataframe(display_df, use_container_width=True)

        st.subheader("PnL Over Time")
        # Ensure 'pnl' is numeric and sum it up over time
        pnl_data = trade_df[trade_df['pnl'] != 'N/A'].copy() # Filter out 'N/A' PnL entries
        pnl_data['pnl'] = pd.to_numeric(pnl_data['pnl'])
        pnl_data.sort_values('timestamp', inplace=True)
        pnl_data['cumulative_pnl'] = pnl_data['pnl'].cumsum()

        if not pnl_data.empty and not pnl_data['cumulative_pnl'].isnull().all():
            # Filter out invalid values before plotting
            pnl_data_clean = pnl_data[pnl_data['cumulative_pnl'].notna() & 
                                     pnl_data['cumulative_pnl'] != float('inf') & 
                                     pnl_data['cumulative_pnl'] != float('-inf')]
            if not pnl_data_clean.empty:
                # Create a line chart of cumulative PnL
                st.line_chart(pnl_data_clean.set_index('timestamp')['cumulative_pnl'])
            else:
                st.info("No valid PnL data available to plot.")
        else:
            st.info("No PnL data available to plot.")

    else:
        st.info("No trade history available.")

    st.markdown("---")
    st.markdown("Dashboard last updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

else:
    st.warning("Firestore client not initialized. Please check your `firebase_credentials.json` and ensure Firebase is set up correctly.")

st.sidebar.header("Dashboard Controls")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear() # Clear the cache
    st.rerun() # Rerun the app to fetch fresh data

st.sidebar.info(
    "This dashboard reads data from your Firestore database. "
    "Ensure your bot (`main.py`) is running and writing data to Firestore."
)