# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import string # For TF-IDF preprocessing
import os     # For path manipulation
from datetime import date # For footer date

# Imports for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# --- SET PAGE CONFIG FIRST ---
# This MUST be the first Streamlit command executed
st.set_page_config(layout="wide")

# --- NLTK Stopwords Setup ---
try:
    stop_words = stopwords.words('english')
except LookupError:
    with st.spinner("Downloading NLTK stopwords data..."):
        nltk.download('stopwords', quiet=True)
    try:
        stop_words = stopwords.words('english')
        # st.success("NLTK stopwords downloaded.") # Can be commented out after first run
    except Exception as e:
        st.error(f"Failed to download NLTK stopwords after attempt: {e}")
        stop_words = [] # Fallback to empty list


# --- Define relative paths to CSV files ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CS_COURSE_PATH = os.path.join(APP_DIR, 'cs_courses.csv')
GENERAL_COURSE_PATH = os.path.join(APP_DIR, 'general_courses.csv')

# =============================================================================
# Data Loading (Cached)
# =============================================================================
@st.cache_data # Cache the data loading result
def load_all_data():
    """Loads CS and General course data from CSV files, normalizes, and caches."""
    try:
        # --- Load CS Courses ---
        cs_df = pd.read_csv(CS_COURSE_PATH)
        string_cols_cs = cs_df.select_dtypes(include=['object']).columns
        for col in string_cols_cs:
            cs_df.loc[:, col] = cs_df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else x)
        if 'prerequisite(s)' not in cs_df.columns:
            cs_df['prerequisite(s)'] = ''
        cs_df.loc[:, 'prerequisite(s)'] = cs_df['prerequisite(s)'].fillna('')
        required_cs_cols = {'course number', 'course title', 'prerequisite(s)'}
        if not required_cs_cols.issubset(cs_df.columns):
                 st.error(f"CS Course data is missing required columns: {required_cs_cols - set(cs_df.columns)}")
                 return None, None

        # --- Load General Courses ---
        general_df = pd.read_csv(GENERAL_COURSE_PATH)
        string_cols_gen = general_df.select_dtypes(include=['object']).columns
        for col in string_cols_gen:
            general_df.loc[:, col] = general_df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else x)
        if 'semester' in general_df.columns:
                 general_df['semester_norm'] = general_df['semester'].astype(str).str.strip().str.lower()
                 required_general_cols = {'semester_norm', 'course number', 'course title'}
        else:
                 st.error(f"General Course data is missing the 'semester' column.")
                 return None, None
        if not required_general_cols.issubset(general_df.columns):
                 missing_cols = required_general_cols - set(general_df.columns)
                 st.error(f"General Course data is missing required columns: {missing_cols}")
                 return None, None

        print("Data loaded and normalized successfully.") # Terminal feedback
        return cs_df, general_df

    except FileNotFoundError as e:
        st.error(f"❌ Error: Data file not found! Details: {e}")
        st.error(f"Please ensure '{os.path.basename(CS_COURSE_PATH)}' and '{os.path.basename(GENERAL_COURSE_PATH)}' exist in the directory: {APP_DIR}")
        return None, None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during data loading:")
        st.exception(e)
        return None, None

# =============================================================================
# Helper Functions
# =============================================================================

def get_allowed_range(level_choice_num):
    """Maps academic level number (1-8) to min/max course number range."""
    ranges = {
        1: (100, 199), 2: (100, 199), 3: (200, 299), 4: (200, 299),
        5: (300, 399), 6: (300, 399), 7: (400, 499), 8: (400, 499)
    }
    return ranges.get(level_choice_num, (0, 999))

def extract_course_number(course_id):
    """Extracts the main number from a course ID string (assumes lowercase input)."""
    try:
        match = re.search(r'[a-z]{2,}\s*(\d{1,3})[a-z]?', str(course_id))
        if match: return int(match.group(1))
        num_part = re.search(r'\d+', str(course_id))
        return int(num_part.group()) if num_part else 0
    except: return 0

def filter_courses_by_level(df, min_level, max_level, id_column='course number'):
    """Adds 'course_num_int' column and filters DataFrame by level range."""
    if id_column not in df.columns: return pd.DataFrame()
    if df.empty: return df
    df_copy = df.copy()
    df_copy['course_num_int'] = df_copy[id_column].astype(str).apply(extract_course_number)
    filtered = df_copy[(df_copy['course_num_int'] >= min_level) & (df_copy['course_num_int'] <= max_level)]
    return filtered.drop(columns=['course_num_int'], errors='ignore')

def check_prerequisites_v2(course, taken_courses_list, cs_df, level_choice, selected_courses_list=None, id_column='course number'):
    """
    Checks if prerequisites for a course are met using restricted eval.
    *** WARNING: Uses eval(), review security implications. ***
    """
    if selected_courses_list is None: selected_courses_list = []
    available_courses = set(str(c).strip().lower() for c in taken_courses_list + selected_courses_list if c is not None)
    course_lower = str(course).strip().lower()

    if cs_df is None or cs_df.empty: return False
    if id_column not in cs_df.columns or 'prerequisite(s)' not in cs_df.columns: return False

    prereq_row = cs_df[cs_df[id_column] == course_lower]
    if prereq_row.empty: return True # Assume met if not in CS list

    prereq_data = prereq_row['prerequisite(s)'].iloc[0]
    if not prereq_data or prereq_data == 'none': return True

    original_prereq_str = prereq_data
    expression = original_prereq_str.lower()
    course_codes = set(re.findall(r'([a-z]{2,}\s*\d{1,3}[a-z]?)', expression))

    if 'math 110' in course_codes and level_choice <= 2 and 'math 141' in available_courses:
        available_courses.add('math 110')

    eval_context = {}
    all_prereqs_mentioned_individually_met = True
    for code in course_codes:
        is_met = code in available_courses
        if not is_met: all_prereqs_mentioned_individually_met = False
        safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', code)
        if safe_key.isidentifier() and not safe_key.startswith('_') and safe_key not in {'and', 'or', 'not', 'true', 'false'}:
            eval_context[safe_key] = is_met

    eval_expression = expression
    eval_expression = re.sub(r'\band\b', ' and ', eval_expression, flags=re.IGNORECASE)
    eval_expression = re.sub(r'\bor\b', ' or ', eval_expression, flags=re.IGNORECASE)
    eval_expression = re.sub(r'\bnot\b', ' not ', eval_expression, flags=re.IGNORECASE)
    eval_expression = eval_expression.replace(',', ' and ').replace('&', ' and ').replace('/', ' or ')

    sorted_course_codes = sorted(list(course_codes), key=len, reverse=True)
    for code in sorted_course_codes:
        safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', code)
        if safe_key.isidentifier() and not safe_key.startswith('_') and safe_key not in {'and', 'or', 'not', 'true', 'false'}:
            escaped_code = re.escape(code)
            eval_expression = re.sub(r'\b' + escaped_code + r'\b', safe_key, eval_expression)

    eval_expression = ' '.join(eval_expression.split())
    if eval_expression.count('(') != eval_expression.count(')'):
         return all_prereqs_mentioned_individually_met

    try:
        potential_leftovers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eval_expression)
        valid_tokens = set(eval_context.keys()) | {'and', 'or', 'not', 'True', 'False'}
        unknown_tokens = [token for token in potential_leftovers if token.lower() not in valid_tokens and token not in {'True', 'False'}]
        if unknown_tokens:
             return all_prereqs_mentioned_individually_met

        allowed_builtins = {'True': True, 'False': False}
        result = eval(eval_expression, {"__builtins__": allowed_builtins}, eval_context)
        return bool(result)

    except Exception as e:
        has_logic_keywords = any(kw in expression for kw in [' and ', ' or ', ' not ', '(', ')'])
        if not has_logic_keywords and len(course_codes) > 0:
            return all_prereqs_mentioned_individually_met
        return False

# --- This list includes ALL courses relevant for requirements (Required + Options) ---
def get_required_courses():
    """
    Returns the list of ALL courses potentially satisfying core requirements (lowercase).
    Used for checking remaining categories and finding available options.
    """
    return [
        'bio 244', 'cmsc 141', 'cmsc 145', 'cmsc 201',
        'cmsc 226', 'cmsc 275', 'cmsc 301', 'cmsc 305',
        'cmsc 326', 'econ 229', 'math 110',
        'math 141', 'psy 203'
    ]

# --- This maps requirement categories to the course(s) that fulfill them ---
def get_remaining_requirements(taken_courses_list):
    """Calculates remaining requirement CATEGORIES based on taken courses (lowercase)."""
    req_map = {
        'MATH 141 (or placement)': ['math 141'],
        'CMSC 141': ['cmsc 141'],
        'CMSC 145': ['cmsc 145'],
        'CMSC 201': ['cmsc 201'],
        'Statistics Requirement': ['cmsc 275', 'bio 244', 'psy 203', 'econ 229'],
        'CMSC 301': ['cmsc 301'],
        'CMSC 305': ['cmsc 305'],
        'Systems Requirement': ['cmsc 226', 'cmsc 326'],
    }
    taken_set = set(str(c).strip().lower() for c in taken_courses_list if c is not None)
    if 'math 141' in taken_set: taken_set.add('math 110')
    remaining = []
    for category, courses_in_category in req_map.items():
        if not courses_in_category: continue
        is_met = any(course_id in taken_set for course_id in courses_in_category)
        if not is_met: remaining.append(category)
    return sorted(list(set(remaining)))

# --- This lists courses available FOR a specific requirement CATEGORY ---
def list_available_courses_for_requirement(
    cs_df, general_df, requirement_category, taken_courses,
    min_level, max_level, semester_norm, level_choice, selected_courses=None):
    """Filters and lists available courses for a specific requirement category."""
    if selected_courses is None: selected_courses = []
    if cs_df is None or general_df is None or cs_df.empty or general_df.empty: return pd.DataFrame()

    cs_id_column = 'course number'; general_id_column = 'course number'; cs_title_column = 'course title'
    req_map = { # Map category name back to course IDs
        'math 141 (or placement)': ['math 141'], 'CMSC 141': ['cmsc 141'], 'CMSC 145': ['cmsc 145'],
        'CMSC 201': ['cmsc 201'], 'Statistics Requirement': ['cmsc 275', 'bio 244', 'psy 203', 'econ 229'],
        'CMSC 301': ['cmsc 301'], 'CMSC 305': ['cmsc 305'], 'Systems Requirement': ['cmsc 226', 'cmsc 326'],
    }
    requirement_category_lower = requirement_category.lower(); target_ids = []
    if requirement_category_lower in req_map: target_ids = req_map[requirement_category_lower]
    else: # Fallback if category name *is* the course ID (less likely now)
        all_valid_course_ids = set(cs_df[cs_id_column].unique()) | set(general_df[general_id_column].unique())
        if requirement_category_lower in all_valid_course_ids: target_ids = [requirement_category_lower]
    if not target_ids: return pd.DataFrame()

    cs_essentials = cs_df[[cs_id_column, cs_title_column, 'prerequisite(s)']].drop_duplicates(subset=[cs_id_column])
    gen_essentials = general_df[[general_id_column, cs_title_column, 'semester_norm']].rename(columns={general_id_column: cs_id_column})
    gen_essentials = gen_essentials.drop_duplicates(subset=[cs_id_column, 'semester_norm'])
    offered_in_semester_df = gen_essentials[gen_essentials['semester_norm'] == semester_norm]
    if offered_in_semester_df.empty: return pd.DataFrame()
    offered_ids_this_semester = set(offered_in_semester_df[cs_id_column].unique())
    potential_candidates_ids = set(target_ids) & offered_ids_this_semester
    if not potential_candidates_ids: return pd.DataFrame()

    base_candidates_df = offered_in_semester_df[offered_in_semester_df[cs_id_column].isin(potential_candidates_ids)].copy()
    base_candidates_df = base_candidates_df.merge(cs_essentials, on=cs_id_column, how='left', suffixes=('_gen', '_cs'))
    base_candidates_df[cs_title_column] = base_candidates_df['course title_cs'].fillna(base_candidates_df['course title_gen'])
    if 'prerequisite(s)' not in base_candidates_df.columns: base_candidates_df['prerequisite(s)'] = ''
    base_candidates_df['prerequisite(s)'] = base_candidates_df['prerequisite(s)'].fillna('')
    base_candidates_df = base_candidates_df[[cs_id_column, cs_title_column, 'prerequisite(s)', 'semester_norm']].drop_duplicates()

    level_filtered_df = filter_courses_by_level(base_candidates_df, min_level, max_level, cs_id_column)
    if level_filtered_df.empty: return pd.DataFrame()

    available_mask = level_filtered_df[cs_id_column].apply(lambda course_id: check_prerequisites_v2(course=course_id, taken_courses_list=taken_courses, cs_df=cs_courses_df, level_choice=level_choice, selected_courses_list=selected_courses))
    final_available_df = level_filtered_df[available_mask]
    if final_available_df.empty: return pd.DataFrame()

    display_columns = [cs_id_column, cs_title_column]; existing_display_columns = [col for col in display_columns if col in final_available_df.columns and not final_available_df[col].isnull().all()]
    return final_available_df[existing_display_columns].drop_duplicates().reset_index(drop=True)

def get_next_semester(current_semester_str):
    """Calculates the next semester string."""
    parts = str(current_semester_str).lower().strip().split();
    if len(parts) != 2: return None
    season, year_str = parts
    try: year = int(year_str)
    except ValueError: return None
    if season == 'spring': return f"fall {year}"
    elif season == 'fall': return f"spring {year + 1}"
    else: return None

def parse_prereqs_to_list(prereq_str):
    """Extracts potential course codes mentioned in a prerequisite string."""
    if pd.isna(prereq_str) or not prereq_str: return []
    course_codes = re.findall(r'([a-z]{2,}\s*\d{1,3}[a-z]?)', str(prereq_str).lower())
    return sorted(list(set(course_codes)))

def parse_distribution_str(distribution_str):
    """Parses comma/space/plus separated distribution codes."""
    if pd.isna(distribution_str) or not distribution_str: return []
    codes = re.split(r'[,\s\+]+', str(distribution_str))
    return [code.strip().upper() for code in codes if code.strip()]

def preprocess_text(text):
    """Cleans text for TF-IDF."""
    if pd.isna(text): return ""
    text = str(text).lower(); text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split();
    if stop_words: words = [word for word in words if word not in stop_words]
    return " ".join(words)

def perform_topic_search_tfidf_st(filtered_df_input, query, cs_df, general_df, N=10, similarity_threshold=0.05):
    """ Performs TF-IDF search on the provided pre-filtered DataFrame. Returns results DF."""
    if filtered_df_input.empty: return pd.DataFrame()
    cs_ref = cs_df[['course number', 'course title', 'course description']].rename(columns={'course description': 'cs_desc'}).drop_duplicates(subset=['course number'])
    courses_df = filtered_df_input.copy()
    courses_df = courses_df.merge(cs_ref[['course number', 'cs_desc']], on='course number', how='left')
    gen_desc_col = 'course description' if 'course description' in courses_df.columns else None
    courses_df['text_for_search'] = courses_df['course title'].fillna('') + ' ' + courses_df['cs_desc'].fillna('') + ' ' + (courses_df[gen_desc_col].fillna('') if gen_desc_col else '')
    if courses_df['text_for_search'].isnull().all() or courses_df['text_for_search'].str.strip().eq('').all(): return pd.DataFrame()
    with st.spinner("Preprocessing text..."): courses_df['preprocessed_text'] = courses_df['text_for_search'].apply(preprocess_text)
    if courses_df['preprocessed_text'].str.strip().eq('').all(): return pd.DataFrame()
    try:
        with st.spinner("Calculating similarities..."):
            vectorizer = TfidfVectorizer(); tfidf_matrix = vectorizer.fit_transform(courses_df['preprocessed_text'])
            preprocessed_query = preprocess_text(query)
            if not preprocessed_query: return pd.DataFrame()
            query_vector = vectorizer.transform([preprocessed_query])
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_n_indices_local = np.argsort(cosine_similarities)[::-1]
        relevant_indices_local = [i for i in top_n_indices_local if cosine_similarities[i] > similarity_threshold][:N]
        if not relevant_indices_local: return pd.DataFrame()
        results_df = courses_df.iloc[relevant_indices_local].copy()
        results_df['similarity_score'] = cosine_similarities[relevant_indices_local]
        return results_df.sort_values(by='similarity_score', ascending=False)
    except Exception as e: st.error("TF-IDF search error."); st.exception(e); return pd.DataFrame()

# =============================================================================
# Streamlit App Main Logic
# =============================================================================

st.title("🎓 University Course Planner & Search")

cs_courses_df, general_courses_df = load_all_data()
if cs_courses_df is None or general_courses_df is None: st.error("Data loading failed."); st.stop()

levels = ['Freshman I', 'Freshman II', 'Sophomore I', 'Sophomore II', 'Junior I', 'Junior II', 'Senior I', 'Senior II']
level_map = {name: i + 1 for i, name in enumerate(levels)}

if 'selected_courses_T' not in st.session_state: st.session_state.selected_courses_T = {}
if 'selected_topic_search_courses' not in st.session_state: st.session_state.selected_topic_search_courses = []
if 'run_phase2' not in st.session_state: st.session_state.run_phase2 = False
if 'target_course_count' not in st.session_state: st.session_state.target_course_count = 4
if 'show_final_schedule' not in st.session_state: st.session_state.show_final_schedule = False
if 'planner_semester_value' not in st.session_state: st.session_state.planner_semester_value = None
if 'planner_level_name' not in st.session_state: st.session_state.planner_level_name = None
if 'planner_level_choice' not in st.session_state: st.session_state.planner_level_choice = None
if 'planner_taken_ids' not in st.session_state: st.session_state.planner_taken_ids = []

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode:", ["📅 Program Planner", "🔍 Course Search & Selection"])
st.sidebar.markdown("---")
st.sidebar.subheader("Final Plan Review")
if st.sidebar.button("Show Final Combined Schedule", key='show_final_schedule_button'):
    if st.session_state.planner_semester_value: st.session_state.show_final_schedule = True; st.session_state.run_phase2 = False
    else: st.sidebar.warning("Set semester/level in Planner first.")

# =============================================================================
# Mode 1: Program Planner (Focus on Required Courses)
# =============================================================================
if app_mode == "📅 Program Planner":
    st.header("📅 Program Planner")
    st.info("Select strictly required courses taken (Step 1), then fulfill remaining requirement categories (Step 2). Use Recommendations (Step 4) or Search tab for electives.")

    all_required_ids_list = get_required_courses() # Gets ALL relevant courses (Required + Options)
    available_cs_ids = set(cs_courses_df['course number'].dropna().unique()); available_gen_ids = set(general_courses_df['course number'].dropna().unique())
    all_available_ids = available_cs_ids.union(available_gen_ids)
    missing_required_from_data = [req_id for req_id in all_required_ids_list if req_id not in all_available_ids]
    if missing_required_from_data: st.warning(f"⚠️ Data Warning: Courses `{', '.join(missing_required_from_data)}` needed for requirements not found in data. Check definitions and CSV files.")

    st.subheader("Step 1: Your Information & STRICTLY Required Courses Taken")
    unique_semesters = sorted(general_courses_df['semester_norm'].dropna().unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        default_sem_index = len(unique_semesters) - 1 if unique_semesters else 0; default_sem = st.session_state.get('planner_semester_value', unique_semesters[default_sem_index] if unique_semesters else None)
        if unique_semesters:
            try: sem_index = unique_semesters.index(default_sem) if default_sem in unique_semesters else default_sem_index
            except ValueError: sem_index = default_sem_index
            chosen_semester_norm = st.selectbox("Select Target Semester:", options=unique_semesters, index=sem_index, key='planner_semester_input')
        else: st.error("No semesters found!"); chosen_semester_norm = None; st.stop()
    with col2:
        default_level = st.session_state.get('planner_level_name', 'Junior I');
        try: level_index = levels.index(default_level)
        except ValueError: level_index = 4
        level_name = st.selectbox("Select Academic Level:", options=levels, index=level_index, key='planner_level_input')
        level_choice = level_map[level_name]; min_level_T, max_level_T = get_allowed_range(level_choice)
    with col3:
        target_course_count = st.number_input("Target courses this semester?", min_value=1, max_value=8, value=st.session_state.get('target_course_count', 4), step=1, key='planner_target_count_input')
    st.session_state.planner_semester_value = chosen_semester_norm; st.session_state.planner_level_name = level_name; st.session_state.planner_level_choice = level_choice; st.session_state.target_course_count = target_course_count
    st.write(f"Planning for: **{chosen_semester_norm.upper()}** | Level: **{level_name} ({min_level_T}-{max_level_T})** | Target Courses: **{target_course_count}**")

    # --- Input STRICTLY Required Courses Taken (Using corrected logic) --- #
    all_courses_lookup_df = pd.concat([cs_courses_df[['course number', 'course title']], general_courses_df[['course number', 'course title']]], ignore_index=True).drop_duplicates(subset=['course number'], keep='first')
    strictly_required_course_ids = ['cmsc 141', 'cmsc 145', 'cmsc 201', 'cmsc 301', 'cmsc 305', 'math 141']
    taken_course_options = {}
    for req_id in strictly_required_course_ids:
         if req_id in all_available_ids:
             course_info = all_courses_lookup_df[all_courses_lookup_df['course number'] == req_id]
             title_val = course_info['course title'].iloc[0] if not course_info.empty else None # Use iloc[0] safely
             title = str(title_val).title() if pd.notna(title_val) and str(title_val).strip() else '(Title Unknown)'
             taken_course_options[req_id] = f"{req_id.upper()} - {title}"
         else: st.warning(f"Strictly required course '{req_id}' missing from data.")
    valid_default_taken = [tid for tid in st.session_state.get('planner_taken_ids', []) if tid in taken_course_options]
    taken_courses_ids = st.multiselect(f"Select STRICTLY REQUIRED courses ALREADY TAKEN (before {chosen_semester_norm}):", options=list(taken_course_options.keys()), format_func=lambda x: taken_course_options.get(x, x.upper()), default=valid_default_taken, key='planner_taken_input_strict')
    st.session_state.planner_taken_ids = taken_courses_ids
    # --- End Input STRICTLY Required ---
    st.markdown("---")

    # --- Step 2: Select Courses for Remaining Requirement CATEGORIES ---
    st.subheader(f"Step 2: Fulfill Remaining Requirement Categories for {chosen_semester_norm.upper()}")
    remaining_requirements = get_remaining_requirements(taken_courses_ids)
    if not remaining_requirements:
        st.success("All core requirement categories appear satisfied!"); st.info("Use 'Course Search & Selection' tab for electives.")
        st.session_state.selected_courses_T = {}
    else:
        st.write("Select one course for each unmet requirement category below:")
        current_selections_this_run = {}
        valid_prev_selections_T = {k: v for k, v in st.session_state.selected_courses_T.items() if v}
        courses_for_prereq_check = taken_courses_ids + st.session_state.selected_topic_search_courses + list(valid_prev_selections_T.values())
        for req_category in remaining_requirements:
            st.markdown(f"##### Requirement Category: {req_category.replace('_', ' ').upper()}") # Nicer display
            available_df = list_available_courses_for_requirement(cs_courses_df, general_courses_df, req_category, taken_courses_ids, min_level_T, max_level_T, chosen_semester_norm, level_choice, courses_for_prereq_check)
            if available_df.empty: st.info(f"No suitable courses found for '{req_category}'.")
            else:
                options_dict = {"Skip": "Skip this category for now"}
                for _, row in available_df.iterrows(): options_dict[row['course number']] = f"{row['course number'].upper()} - {row['course title'].title()}"
                default_selection_id = st.session_state.selected_courses_T.get(req_category, "Skip")
                if default_selection_id not in options_dict: default_selection_id = "Skip"
                selected_course_id = st.radio(f"Select one course for '{req_category.replace('_', ' ').upper()}':", options=list(options_dict.keys()), format_func=lambda x: options_dict[x], key=f"select_{req_category}_{chosen_semester_norm}_{level_name}", index=list(options_dict.keys()).index(default_selection_id))
                if selected_course_id != "Skip": current_selections_this_run[req_category] = selected_course_id
        st.session_state.selected_courses_T = current_selections_this_run.copy()
        current_total_selected_count_check = len(st.session_state.selected_courses_T) + len(st.session_state.selected_topic_search_courses)
        if current_total_selected_count_check > st.session_state.target_course_count: st.warning(f"⚠️ {current_total_selected_count_check} total selected, exceeding target of {st.session_state.target_course_count}.")
    st.markdown("---")

    # --- Step 3: Summary of Courses Selected for Requirements ---
    st.subheader(f"Step 3: Summary of Courses Selected for Requirements")
    required_selected_ids = list(st.session_state.selected_courses_T.values())
    if not required_selected_ids: st.info("No courses selected for requirement categories this semester.")
    else:
        selected_req_df = all_courses_lookup_df[all_courses_lookup_df['course number'].isin(required_selected_ids)][['course number', 'course title']]
        if not selected_req_df.empty:
             selected_req_df = selected_req_df.drop_duplicates(); selected_req_df['course title'] = selected_req_df['course title'].fillna('(Title Unknown)').str.title()
             st.dataframe(selected_req_df.rename(columns={'course number': 'Course ID', 'course title': 'Title'}).reset_index(drop=True)); st.caption(f"{len(required_selected_ids)} courses selected.")
        else: st.info("Details missing for selected courses.")
    st.markdown("➡️ Go to **Course Search & Selection** tab for electives.")
    st.markdown("---")

    # --- Step 4: Preliminary Prerequisite Check & Recommendations ---
    next_semester_str = get_next_semester(chosen_semester_norm)
    if not next_semester_str: st.warning("Could not determine next semester.")
    else:
        st.subheader(f"Step 4: Prerequisite Check & Recommendations for {next_semester_str.upper()}")
        st.caption(f"(Analyzes potential requirements next semester. Use recommendations below to add prerequisites to **{chosen_semester_norm.upper()}** plan.)")
        current_req_selected_ids = list(st.session_state.selected_courses_T.values()); current_elective_selected_ids = st.session_state.selected_topic_search_courses
        prelim_combined_ids = sorted(list(set(current_req_selected_ids + current_elective_selected_ids)))
        if st.button(f"Analyze Prerequisites Needed for {next_semester_str.upper()} (Preliminary)", key='run_phase2_prelim'):
            st.session_state.run_phase2 = True
            if not (taken_courses_ids or prelim_combined_ids): st.warning(f"Select courses taken/planned first."); st.session_state.run_phase2 = False
            else:
                 with st.spinner(f"Analyzing prerequisites..."): # Calculation Logic
                     next_level_choice = min(level_choice + 1, 8); min_next_level, max_next_level = get_allowed_range(next_level_choice)
                     courses_met_by_end_of_T = set(taken_courses_ids + prelim_combined_ids); required_cs_courses_all = get_required_courses()
                     target_t_plus_1_courses = [req for req in required_cs_courses_all if req not in courses_met_by_end_of_T and min_next_level <= extract_course_number(req) <= max_next_level]
                     st.session_state.phase2_target_courses = target_t_plus_1_courses; missing_prereqs_map = {}
                     for target_course in target_t_plus_1_courses:
                         if not check_prerequisites_v2(target_course, taken_courses_ids, cs_courses_df, next_level_choice, prelim_combined_ids):
                             prereq_row = cs_courses_df[cs_courses_df['course number'] == target_course]
                             if prereq_row.empty: prereq_row = general_courses_df[general_courses_df['course number'] == target_course]
                             if not prereq_row.empty and 'prerequisite(s)' in prereq_row.columns:
                                 prereq_data = prereq_row['prerequisite(s)'].iloc[0]; mentioned_prereqs = parse_prereqs_to_list(prereq_data)
                                 for prereq in mentioned_prereqs:
                                     if prereq not in courses_met_by_end_of_T:
                                         if prereq not in missing_prereqs_map: missing_prereqs_map[prereq] = set()
                                         missing_prereqs_map[prereq].add(target_course.upper())
                     missing_prereqs_for_T_set = set(missing_prereqs_map.keys()); st.session_state.phase2_missing_prereqs = missing_prereqs_for_T_set
                     phase2_recommendations = []
                     if missing_prereqs_for_T_set:
                          offered_in_T_df = general_courses_df[general_courses_df['semester_norm'] == chosen_semester_norm]; offered_in_T_ids = set(offered_in_T_df['course number'].unique())
                          for prereq_needed in missing_prereqs_for_T_set:
                              is_offered = prereq_needed in offered_in_T_ids; prereq_num = extract_course_number(prereq_needed); is_level_appropriate = min_level_T <= prereq_num <= max_level_T
                              can_take_prereq = check_prerequisites_v2(prereq_needed, taken_courses_ids, cs_courses_df, level_choice, None)
                              if is_offered and is_level_appropriate and can_take_prereq and prereq_needed not in prelim_combined_ids:
                                   needed_for_list_str = ", ".join(sorted(list(missing_prereqs_map.get(prereq_needed, set())))); phase2_recommendations.append({'course_needed': prereq_needed, 'needed_for': needed_for_list_str})
                     st.session_state.phase2_recommendations = phase2_recommendations

        if st.session_state.get('run_phase2', False): # Display Logic for Step 4
            target_courses_disp = st.session_state.get('phase2_target_courses', []); missing_prereqs_disp = st.session_state.get('phase2_missing_prereqs', set()); recommendations_disp = st.session_state.get('phase2_recommendations', [])
            st.markdown("---")
            st.write(f"Potential unmet required courses for **{next_semester_str.upper()}**: `{', '.join(target_courses_disp) or 'None'}`")
            if not missing_prereqs_disp: st.success(f"All prerequisites seem met!")
            else:
                st.warning(f"Potential prerequisites needed for {next_semester_str.upper()} courses: `{', '.join(missing_prereqs_disp)}`")
                if not recommendations_disp: st.info(f"No suitable *additional* prerequisites identified for recommendation in {chosen_semester_norm.upper()}.")
                else: # Interactive Recommendation Block
                    st.subheader(f"❗️ Recommended Prerequisites for {chosen_semester_norm.upper()} ❗️")
                    st.warning(f"Select courses below to add to plan for **{chosen_semester_norm.upper()}**.")
                    all_courses_ref_df_step4 = all_courses_lookup_df; recs_to_offer = []
                    current_all_selected_ids_before_widget = set(list(st.session_state.selected_courses_T.values()) + st.session_state.selected_topic_search_courses)
                    for rec_item in recommendations_disp:
                        course_id = rec_item['course_needed']
                        if course_id not in current_all_selected_ids_before_widget:
                            rec_detail = all_courses_ref_df_step4[all_courses_ref_df_step4['course number'] == course_id]
                            title = rec_detail['course title'].iloc[0].title() if not rec_detail.empty else '(Title Unknown)'; needed_for = rec_item['needed_for']
                            recs_to_offer.append({'id': course_id, 'display': f"{course_id.upper()} - {title} (Needed for: {needed_for})"})
                    if not recs_to_offer: st.info("All recommended prerequisites appear already selected.")
                    else:
                        recommendation_options = {rec['id']: rec['display'] for rec in recs_to_offer}; available_rec_ids_in_view = set(recommendation_options.keys())
                        valid_defaults_recs = [sel for sel in st.session_state.selected_topic_search_courses if sel in available_rec_ids_in_view]
                        selected_in_recommendation_widget = st.multiselect(f"Select recommended courses to add:", options=list(recommendation_options.keys()), format_func=lambda x: recommendation_options.get(x, x.upper()), default=valid_defaults_recs, key=f"recommendation_multiselect_{chosen_semester_norm}_{level_name}")
                        selections_outside_this_rec_view = [sel for sel in st.session_state.selected_topic_search_courses if sel not in available_rec_ids_in_view]
                        updated_total_elective_selections = sorted(list(set(selections_outside_this_rec_view + selected_in_recommendation_widget)))
                        if updated_total_elective_selections != st.session_state.selected_topic_search_courses: st.session_state.selected_topic_search_courses = updated_total_elective_selections
            st.markdown("---") # Metric Display
            current_total_selected_count_planner = len(st.session_state.selected_courses_T) + len(st.session_state.selected_topic_search_courses)
            target = st.session_state.target_course_count; 
            delta_color = "normal" # Default color
            delta_val = current_total_selected_count_planner - target
            if delta_val > 0:
                delta_color = "inverse" # More than target
            elif delta_val < 0:
                delta_color = "off"     # Less than target
            st.metric(label=f"Total Planned Courses ({chosen_semester_norm.upper()})", value=f"{current_total_selected_count_planner} / {target}", delta=f"{delta_val} vs Target", delta_color=delta_color)
            if current_total_selected_count_planner > target: st.warning(f"⚠️ Exceeding target of {target} courses.")

# =============================================================================
# Mode 2: Course Search & Selection
# =============================================================================
elif app_mode == "🔍 Course Search & Selection":
    st.header("🔍 Course Search & Selection")
    st.info("Find courses by filtering or topic to add as electives or for distribution requirements.")

    st.subheader("Step 1: Initial Filters")
    unique_semesters_search = sorted(general_courses_df['semester_norm'].dropna().unique())
    levels_search = ['Freshman I', 'Freshman II', 'Sophomore I', 'Sophomore II', 'Junior I', 'Junior II', 'Senior I', 'Senior II', 'All Levels']
    col1_search, col2_search = st.columns(2)
    with col1_search:
        default_sem_index_search = len(unique_semesters_search) - 1 if unique_semesters_search else 0
        planner_sem = st.session_state.get('planner_semester_value', unique_semesters_search[default_sem_index_search] if unique_semesters_search else None)
        if unique_semesters_search:
             try: sem_index = unique_semesters_search.index(planner_sem) if planner_sem in unique_semesters_search else default_sem_index_search
             except ValueError: sem_index = default_sem_index_search
             chosen_semester_search = st.selectbox("Select Semester:", options=unique_semesters_search, index=sem_index, key='search_semester_input')
        else: st.error("No semesters found!"); chosen_semester_search = None; st.stop()
    with col2_search:
        planner_level_name_search = st.session_state.get('planner_level_name', 'All Levels')
        if planner_level_name_search not in levels_search: planner_level_name_search = 'All Levels'
        try: level_index = levels_search.index(planner_level_name_search)
        except ValueError: level_index = len(levels_search)-1
        level_name_search = st.selectbox("Select Academic Level:", options=levels_search, index=level_index, key='search_level_input')

    with st.spinner("Applying initial filters..."):
        semester_filtered_df = general_courses_df[general_courses_df["semester_norm"] == chosen_semester_search].copy()
        if level_name_search != 'All Levels':
            level_choice_search = level_map[level_name_search] # Uses global level_map
            min_level_search, max_level_search = get_allowed_range(level_choice_search)
            final_filtered_df = filter_courses_by_level(semester_filtered_df, min_level_search, max_level_search)
            level_desc = f"Level: **{level_name_search} ({min_level_search}-{max_level_search})**"
        else: final_filtered_df = semester_filtered_df; level_desc = "Level: **All Levels**"
    st.success(f"Found **{len(final_filtered_df)}** courses in **{chosen_semester_search.upper()}** matching {level_desc}.")
    st.markdown("---")

    if final_filtered_df.empty: st.warning("No courses match selected Semester/Level.")
    else:
        st.subheader("Step 2: Find Courses")
        search_action = st.radio("How to find courses?", options=["Filter by Program / Distribution Area", "Search by Topic Keywords"], key='search_action_choice_v2', horizontal=True)
        st.markdown("---")
        courses_to_display_df = pd.DataFrame()

        if search_action == "Filter by Program / Distribution Area": # Filter Logic
            st.subheader("Step 2a: Apply Additional Filters")
            program_col = 'program'; dist_area_col = 'dist area'; program_filtered_df = final_filtered_df; dist_filtered_df = pd.DataFrame()
            if program_col in final_filtered_df.columns:
                program_filtered_df.loc[:, f"{program_col}_norm"] = program_filtered_df[program_col].astype(str).str.strip().str.lower()
                program_options = ["All Programs"] + sorted(program_filtered_df[f"{program_col}_norm"].dropna().unique())
                chosen_program = st.selectbox("Filter by Program:", options=program_options, key='filter_program_v2')
                if chosen_program != "All Programs": program_filtered_df = program_filtered_df[program_filtered_df[f"{program_col}_norm"] == chosen_program].copy()
            if not program_filtered_df.empty and dist_area_col in program_filtered_df.columns:
                relevant_dist_codes_set = set()
                for area_str in program_filtered_df[dist_area_col].dropna(): relevant_dist_codes_set.update(parse_distribution_str(area_str))
                relevant_dist_codes_set.discard(''); relevant_dist_codes_set.discard('NONE'); dist_area_options = ["All Areas"] + sorted(list(relevant_dist_codes_set))
                chosen_dist_area = st.selectbox("Filter by Distribution Area:", options=dist_area_options, key='filter_dist_area_v2')
                if chosen_dist_area != "All Areas": dist_filtered_df = program_filtered_df[program_filtered_df[dist_area_col].apply(lambda x: chosen_dist_area in parse_distribution_str(x))].copy()
                else: dist_filtered_df = program_filtered_df
            elif program_filtered_df.empty: dist_filtered_df = program_filtered_df
            st.subheader("Filtered Course List");
            if dist_filtered_df.empty: st.info("No courses match filter criteria.")
            else: courses_to_display_df = dist_filtered_df.copy()
        elif search_action == "Search by Topic Keywords": # Search Logic
            st.subheader("Step 2a: Enter Topic Keywords")
            topic_query = st.text_input("Enter keywords:", key='search_query_v2')
            if topic_query:
                 with st.spinner("Performing topic search..."):
                     tfidf_results_df = perform_topic_search_tfidf_st(filtered_df_input=final_filtered_df, query=topic_query, cs_df=cs_courses_df, general_df=general_courses_df)
                     if not tfidf_results_df.empty: courses_to_display_df = tfidf_results_df.copy()

        st.markdown("---")
        # --- Step 3: Select Courses from Filter/Search Results ---
        if not courses_to_display_df.empty:
            st.subheader("Step 3: Select Courses to Add to Plan")
            display_select_df = courses_to_display_df[['course number', 'course title']].copy(); display_select_df.rename(columns={'course number':'ID', 'course title':'Title'}, inplace=True)
            display_select_df['Title'] = display_select_df['Title'].fillna('(Title Unknown)').str.title()
            st.dataframe(display_select_df.drop_duplicates().reset_index(drop=True))

            topic_search_options_dict = {} # Fixed block start
            for _, row in courses_to_display_df.drop_duplicates(subset=['course number']).iterrows():
                course_id = row['course number']; title_val = row['course title']
                if pd.notna(title_val) and str(title_val).strip(): display_title = str(title_val).strip().title()
                else: display_title = '(Title Unknown)'
                topic_search_options_dict[str(course_id)] = f"{str(course_id).upper()} - {display_title}"
            # Fixed block end

            selectable_options = {k: v for k, v in topic_search_options_dict.items() if k not in st.session_state.selected_courses_T.values()}
            if not selectable_options: st.info("All found courses are selected as required in Planner tab.")
            else:
                 current_option_ids_in_view = set(selectable_options.keys()); state_elective_selections = st.session_state.selected_topic_search_courses
                 valid_defaults_search = [sel for sel in state_elective_selections if sel in current_option_ids_in_view]
                 selected_in_this_view = st.multiselect("Select/deselect courses from list above:", options=list(selectable_options.keys()), format_func=lambda x: selectable_options.get(x, x.upper()), default=valid_defaults_search, key='topic_search_multiselect')
                 selections_outside_this_view = [sel for sel in state_elective_selections if sel not in current_option_ids_in_view]
                 updated_total_selections = sorted(list(set(selections_outside_this_view + selected_in_this_view)))
                 if updated_total_selections != st.session_state.selected_topic_search_courses: st.session_state.selected_topic_search_courses = updated_total_selections

                 # --- Metric Display (Corrected Syntax) ---
                 current_total_selected_count_topic = len(st.session_state.selected_courses_T) + len(st.session_state.selected_topic_search_courses)
                 target = st.session_state.target_course_count
                 delta_color = "normal"; delta_val = current_total_selected_count_topic - target
                 if delta_val > 0:
                     delta_color = "inverse"
                 elif delta_val < 0:
                     delta_color = "off"
                 st.metric(label="Total Planned Courses", value=f"{current_total_selected_count_topic} / {target}", delta=f"{delta_val} vs Target", delta_color=delta_color)
                 if current_total_selected_count_topic > target: st.warning(f"⚠️ Exceeding target of {target} courses.")
                 # --- End Metric Display ---
        else:
             st.info("No courses to display/select based on filters/search.")
             # --- Metric Display (Corrected Syntax - Also in Else block) ---
             current_total_selected_count_topic = len(st.session_state.selected_courses_T) + len(st.session_state.selected_topic_search_courses)
             target = st.session_state.target_course_count
             delta_color = "normal"; delta_val = current_total_selected_count_topic - target
             if delta_val > 0:
                 delta_color = "inverse"
             elif delta_val < 0:
                 delta_color = "off"
             st.metric(label="Total Planned Courses", value=f"{current_total_selected_count_topic} / {target}", delta=f"{delta_val} vs Target", delta_color=delta_color)
             # --- End Metric Display ---

# =============================================================================
# Final Combined Schedule Display Area
# =============================================================================
if st.session_state.show_final_schedule:
     st.divider(); st.header("✅ Final Combined Schedule")
     planner_semester_value = st.session_state.get('planner_semester_value', None)
     if not planner_semester_value: st.error("Set target semester in 'Program Planner' first."); st.stop()
     st.subheader(f"Plan for: {planner_semester_value.upper()}")
     final_combined_ids = sorted(list(set(list(st.session_state.selected_courses_T.values()) + st.session_state.selected_topic_search_courses)))
     if not final_combined_ids: st.warning("No courses selected.")
     else:
          target = st.session_state.target_course_count; actual = len(final_combined_ids)
          delta_color = "normal"; delta_val = actual - target
          if delta_val > 0: delta_color = "inverse"; elif delta_val < 0: delta_color = "off"
          st.metric(label="Planned Courses", value=f"{actual} / {target}", delta=f"{delta_val} vs Target", delta_color=delta_color)
          try: all_courses_lookup_df # Check if lookup df exists from Planner mode
          except NameError: # Define it if not (e.g., if only Search mode was visited)
               all_courses_lookup_df = pd.concat([cs_courses_df[['course number', 'course title']], general_courses_df[['course number', 'course title']]], ignore_index=True).drop_duplicates(subset=['course number'], keep='first')
          final_summary_df = all_courses_lookup_df[all_courses_lookup_df['course number'].isin(final_combined_ids)][['course number', 'course title']]
          if not final_summary_df.empty:
               final_summary_df = final_summary_df.drop_duplicates(); final_summary_df['course title'] = final_summary_df['course title'].fillna('(Title Unknown)').str.title()
               st.dataframe(final_summary_df.rename(columns={'course number': 'Course ID', 'course title': 'Title'}).reset_index(drop=True))
          else: st.warning("Could not retrieve details for selected courses.")

          st.markdown("---"); st.subheader("Final Prerequisite Check for Next Semester")
          level_choice_final = st.session_state.get('planner_level_choice', None); taken_ids_final = st.session_state.get('planner_taken_ids', [])
          next_semester_final = get_next_semester(planner_semester_value) if planner_semester_value else None
          if not next_semester_final: st.warning("Could not determine next semester.")
          elif level_choice_final is None: st.warning("Planner level not set.")
          else:
               if st.button(f"Analyze Prerequisites for {next_semester_final.upper()} (Based on Final Plan)", key="final_phase2_check"):
                    with st.spinner(f"Analyzing prerequisites..."): # Final Prereq Calculation
                         next_level_choice = min(level_choice_final + 1, 8); min_next_level, max_next_level = get_allowed_range(next_level_choice)
                         courses_met_by_end_of_T = set(taken_ids_final + final_combined_ids); required_cs_courses_all = get_required_courses()
                         target_t_plus_1_courses = [req for req in required_cs_courses_all if req not in courses_met_by_end_of_T and min_next_level <= extract_course_number(req) <= max_next_level]
                         st.write(f"Potential unmet required courses for **{next_semester_final.upper()}**: `{', '.join(target_t_plus_1_courses) or 'None'}`")
                         missing_prereqs_map = {}
                         for target_course in target_t_plus_1_courses:
                             if not check_prerequisites_v2(target_course, taken_ids_final, cs_courses_df, next_level_choice, final_combined_ids):
                                  prereq_row = cs_courses_df[cs_courses_df['course number'] == target_course]
                                  if prereq_row.empty: prereq_row = general_courses_df[general_courses_df['course number'] == target_course]
                                  if not prereq_row.empty and 'prerequisite(s)' in prereq_row.columns:
                                       prereq_data = prereq_row['prerequisite(s)'].iloc[0]; mentioned_prereqs = parse_prereqs_to_list(prereq_data)
                                       for prereq in mentioned_prereqs:
                                           if prereq not in courses_met_by_end_of_T:
                                                if prereq not in missing_prereqs_map: missing_prereqs_map[prereq] = set()
                                                missing_prereqs_map[prereq].add(target_course.upper())
                         missing_prereqs_for_T_set_final = set(missing_prereqs_map.keys())
                         # Display Final Prereq Results
                         if not missing_prereqs_for_T_set_final: st.success(f"All prerequisites for potential required courses seem met!")
                         else:
                              st.warning(f"Potential prerequisites needed for {next_semester_final.upper()} courses: `{', '.join(missing_prereqs_for_T_set_final)}`")
                              phase2_recommendations_final = []
                              offered_in_T_df_final = general_courses_df[general_courses_df['semester_norm'] == planner_semester_value]; offered_in_T_ids_final = set(offered_in_T_df_final['course number'].unique())
                              planner_min_level_final, planner_max_level_final = get_allowed_range(level_choice_final)
                              for prereq_needed in missing_prereqs_for_T_set_final:
                                   is_offered = prereq_needed in offered_in_T_ids_final; prereq_num = extract_course_number(prereq_needed); is_level_appropriate = planner_min_level_final <= prereq_num <= planner_max_level_final
                                   can_take_prereq = check_prerequisites_v2(prereq_needed, taken_ids_final, cs_courses_df, level_choice_final, None)
                                   if is_offered and is_level_appropriate and can_take_prereq and prereq_needed not in final_combined_ids:
                                        needed_for_list_str = ", ".join(sorted(list(missing_prereqs_map.get(prereq_needed, set())))); phase2_recommendations_final.append({'course_needed': prereq_needed, 'needed_for': needed_for_list_str})
                              if phase2_recommendations_final:
                                   st.subheader(f"❗️ Recommended Prerequisites to Consider Adding to {planner_semester_value.upper()} ❗️")
                                   st.warning(f"(Use Planner/Search tabs to adjust plan, then click 'Show Final Combined Schedule' again).")
                                   rec_df_final = pd.DataFrame(phase2_recommendations_final)
                                   rec_df_final = rec_df_final.merge(all_courses_lookup_df[['course number', 'course title']], left_on='course_needed', right_on='course number', how='left')
                                   rec_df_final['course title'] = rec_df_final['course title'].fillna('(Title Unknown)').str.title()
                                   st.dataframe(rec_df_final[['course number', 'course title', 'needed_for']].rename(columns={'course number':'Course ID', 'course title':'Title', 'needed_for':'Helps Prereqs For'}).reset_index(drop=True))
                              else: st.info("No suitable *additional* prerequisites identified for recommendation.")

# --- Footer ---
st.markdown("---")
data_date_placeholder = "[Enter Data Source Date]" # TODO: Replace with actual date
today = date.today().strftime("%B %d, %Y")
st.caption(f"Course Planner & Search Tool | Data Date: {data_date_placeholder} | Accessed: {today}")