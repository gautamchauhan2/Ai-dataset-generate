import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import tempfile
import os
import google.generativeai as genai

# Configure Google AI API
genai.configure(api_key="AIzaSyDp6sMaVDOCva53AA_yzdGZ9vb2fSDAq-8")  # Replace with your actual API Key

# Initialize session state for AI beliefs, desires, intentions, and rewards
if "beliefs" not in st.session_state:
    st.session_state.beliefs = {}
if "desires" not in st.session_state:
    st.session_state.desires = {}
if "intentions" not in st.session_state:
    st.session_state.intentions = {}
if "rewards" not in st.session_state:
    st.session_state.rewards = {}

# 🎯 RL-BDI Agent (Belief-Desire-Intention Model)
class RLBDIAgent:
    def update_beliefs(self, feature, dependencies):
        """ Update AI beliefs when new dependencies are added. """
        st.session_state.beliefs[feature] = dependencies

    def refine_desires(self, feature):
        """ Adjust AI desires based on new dependencies. """
        st.session_state.desires[feature] = f"The AI wants to refine and expand dependencies for {feature}."

    def update_intentions(self, feature, selected):
        """ Update AI intentions when dependencies are selected. """
        st.session_state.intentions[feature] = f"The AI intends to analyze the selected dependencies for {feature}."

    def reward(self, feature, success=True):
        """ Reinforcement Learning Feedback for AI Learning. """
        if feature not in st.session_state.rewards:
            st.session_state.rewards[feature] = {"success": 0, "penalty": 0}
        if success:
            st.session_state.rewards[feature]["success"] += 1
        else:
            st.session_state.rewards[feature]["penalty"] += 1

# Initialize AI Agent
agent = RLBDIAgent()

import re

def normalize_text(text):
    """ Normalize AI response by converting inconsistent spaces/tabs into a standard format. """
    return re.sub(r"\*\s{2,}", "* ", text)  # Replace extra spaces after asterisks with a single space
def get_ai_dependencies(feature):
    """ 
    Fetch AI-generated dependencies using Gemini AI while ensuring meaningful suggestions.
    This version ensures that only Primary dependencies (10-20) are extracted.
    """
    prompt = (
        f"List at least 10 to 20 primary dependencies for '{feature}', without secondary or tertiary categories. "
        "Each dependency should be formatted as:\n"
        "* **feature_name** (reason why it is a primary dependency)\n"
        "Focus only on Primary dependencies—no secondary or tertiary ones."
    )
    
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        raw_output = response.text if response.text else "EMPTY RESPONSE"

        # ✅ Normalize text before parsing
        normalized_output = normalize_text(raw_output)

        # ✅ Debugging: Print raw AI response
        print("\n=== AI RAW RESPONSE ===\n", raw_output, "\n====================\n")

        if raw_output == "EMPTY RESPONSE":
            return {"Primary": []}, {}

        # ✅ Extract Primary dependencies only
        primary_dependencies = []
        explanations = {}

        for line in normalized_output.split("\n"):
            line = line.strip()
            
            if line.startswith("* **"):  # ✅ Ensure correct parsing of Primary dependencies
                match = re.match(r"\*\s*\*\*([^*]+)\*\*\s*\(([^)]+)\)", line)
                if match:
                    feature_name, reason = match.groups()
                    primary_dependencies.append(feature_name.strip())
                    explanations[feature_name.strip()] = reason.strip()

        # ✅ Ensure at least 10 and at most 20 dependencies
        if len(primary_dependencies) < 10:
            print("⚠️ Warning: AI generated fewer than 10 Primary dependencies. Consider refining the prompt.")
        elif len(primary_dependencies) > 20:
            primary_dependencies = primary_dependencies[:20]  # Trim to 20 max

        # 🚨 Debugging: Print extracted dependencies
        print("\n=== EXTRACTED DEPENDENCIES ===\n", primary_dependencies, "\n====================\n")

        # ✅ Keeping the structure unchanged (storing only Primary in 'dependencies' key)
        return {"Primary": primary_dependencies}, explanations

    except Exception as e:
        st.error(f"⚠️ AI Error: {e}")
        return {"Primary": []}, {}



# Initialize session state for dependencies
if "dependencies" not in st.session_state:
    st.session_state.dependencies = {}
if "explanations" not in st.session_state:
    st.session_state.explanations = {}
if "selected_dependencies" not in st.session_state:
    st.session_state.selected_dependencies = {}
if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()

# Sidebar: AI Thought Process
st.sidebar.title("🧠 AI Thought Process")
st.sidebar.subheader("📌 AI's Current Knowledge (Beliefs)")
for feature, deps in st.session_state.beliefs.items():
    st.sidebar.write(f"For **{feature}**, the AI believes these dependencies exist.")

st.sidebar.subheader("🎯 AI's Goal (Desires)")
for feature, desire in st.session_state.desires.items():
    st.sidebar.write(f"For **{feature}**, {desire}")

st.sidebar.subheader("🚀 AI's Next Step (Intentions)")
for feature, intention in st.session_state.intentions.items():
    st.sidebar.write(f"For **{feature}**, {intention}")

# Main App
st.title("🤖 AI-Powered Dynamic Dependency Analyzer")

# Step 1: Enter Target Feature
st.subheader("🟢 Step 1: Enter a Target Feature")
target_feature = st.text_input("Enter the Target Feature (e.g., Car Performance):")

if target_feature and target_feature not in st.session_state.dependencies:
    deps, explanations = get_ai_dependencies(target_feature)
    st.session_state.dependencies[target_feature] = deps
    st.session_state.explanations[target_feature] = explanations
    st.session_state.selected_dependencies[target_feature] = []

    # ✅ Update BDI
    agent.update_beliefs(target_feature, deps)
    agent.refine_desires(target_feature)

# Step 2: Select & Confirm Dependencies
st.subheader("🟡 Step 2: Select & Expand Dependencies")
for parent, children in list(st.session_state.dependencies.items()):
    st.write(f"### Dependencies for: {parent}")

    for category, items in children.items():
        if items:
            st.markdown(f"**🔹 {category} Dependencies:**")
            for item in items:
                explanation = st.session_state.explanations[parent].get(item, "No explanation provided.")
                st.markdown(f"- **{item}**: {explanation}")

    selected = st.multiselect(
        f"Select dependencies for {parent}:",
        sum(children.values(), []),
        default=st.session_state.selected_dependencies.get(parent, []),
    )

    if st.button(f"✅ Confirm & Expand {parent}", key=f"confirm_{parent}"):
        st.session_state.selected_dependencies[parent] = selected
        st.session_state.expanded_nodes.add(parent)

        # Expand AI Dependencies
        for item in selected:
            if item not in st.session_state.dependencies:
                deps, explanations = get_ai_dependencies(item)
                st.session_state.dependencies[item] = deps
                st.session_state.explanations[item] = explanations

                # ✅ Update BDI
                agent.update_beliefs(item, deps)
                agent.refine_desires(item)

# Step 3: Generate Graph
st.subheader("📊 Generate Dependency Graph")
if st.button("🔄 Generate Graph"):
    G = nx.DiGraph()
    for parent, children in st.session_state.selected_dependencies.items():
        for child in children:
            G.add_edge(parent, child)

    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_file.name)
    st.components.v1.html(open(temp_file.name, "r").read(), height=600)
    os.unlink(temp_file.name)

import random

# Step 4: Generate Synthetic Dataset
st.subheader("📊 Step 4: Generate Synthetic Dataset")

if st.button("📄 Generate Dataset"):
    if not st.session_state.selected_dependencies:
        st.warning("⚠️ No dependencies selected. Please expand some dependencies first.")
    else:
        # Extract all features and determine depth
        feature_levels = {}  # {feature: depth}
        def assign_depth(feature, depth=1):
            if feature in feature_levels:
                feature_levels[feature] = min(feature_levels[feature], depth)  # Store the smallest depth found
            else:
                feature_levels[feature] = depth
            for dep in st.session_state.selected_dependencies.get(feature, []):
                assign_depth(dep, depth + 1)

        # Get all selected root features
        root_features = list(st.session_state.selected_dependencies.keys())
        for root in root_features:
            assign_depth(root, 1)

        # Extract all unique features
        all_features = list(feature_levels.keys())

        if not all_features:
            st.warning("⚠️ No features available for dataset generation.")
        else:
            # First feature is the target variable
            target_feature = root_features[0]

            # Dictionary to store generated feature values
            data = []

            # Generate 100 rows of logical synthetic data
            for _ in range(100):
                row = {}

                # Step 1: Generate Base Feature Values
                base_values = {}
                for feature in all_features:
                    base_values[feature] = random.randint(50, 100)  # Initial random value (adjusted later)

                # Step 2: Apply Dependency-Based Adjustments with Exponential Decay
                for feature, dependencies in st.session_state.selected_dependencies.items():
                    for dependent_feature in dependencies:
                        if dependent_feature in base_values:
                            depth = feature_levels.get(dependent_feature, 1)
                            influence_factor = 1 / (1.5 ** (depth - 1))  # Exponential decay (higher depth = lower impact)
                            base_values[dependent_feature] = max(0, min(100, base_values[feature] * influence_factor + random.randint(-5, 5)))

                # Step 3: Assign values to dataset row
                for feature in all_features:
                    row[feature] = base_values[feature]

                # Step 4: Compute Target Variable with Decayed Influence from Dependencies
                if target_feature in all_features:
                    relevant_features = [f for f in all_features if f in st.session_state.selected_dependencies.get(target_feature, [])]
                    if relevant_features:
                        row[target_feature] = sum(row[f] * (1 / (1.5 ** (feature_levels[f] - 1))) for f in relevant_features)  # Weighted avg with decay

                data.append(row)

            # Convert to DataFrame
            df = pd.DataFrame(data)
            st.write("### 📝 Generated Dataset")
            st.dataframe(df)

            # Provide CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="synthetic_dataset.csv",
                mime="text/csv"
            )
